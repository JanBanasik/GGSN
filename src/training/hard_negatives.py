"""
Hard negative mining for InfoNCE contrastive training.

Standard MoCo / SimCLR trick: every N epochs, compute embeddings on the entire
training set, and for each anchor find the K most-similar negatives (hardest
to distinguish). Then build batches as (1 anchor + K-1 hard negatives) instead
of fully-random in-batch negatives.

LEAKAGE GUARDS (critical for contrastive on patient data):
- Exclude the anchor itself.
- Exclude any sample sharing the anchor's ``subject_id`` — otherwise the model
  is asked to push apart two notes from the same patient, which is exactly
  the same-patient signal we accidentally treated as a negative pair.
  (Two different notes from the same stay are still "the same person", so
  pushing them apart corrupts the embedding semantics.)

API
---
    table = compute_hard_negative_table(
        text_emb, signal_emb, subject_ids, k_per_anchor=128,
    )
    sampler = HardNegativeBatchSampler(
        table, anchor_pool=train_idx, batch_size=64, generator=rng,
    )
    DataLoader(..., batch_sampler=sampler)
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import torch


def compute_hard_negative_table(
    text_emb: torch.Tensor,
    signal_emb: torch.Tensor,
    subject_ids: Sequence[int],
    *,
    k_per_anchor: int = 128,
    chunk_size: int = 1024,
) -> torch.Tensor:
    """
    Build a (N, K) tensor of hard negative INDICES per anchor.

    For anchor ``i``:
        scores_i[j] = text_emb[i] · signal_emb[j]            (cosine similarity)
        valid_j     = (j != i) AND (subject_ids[j] != subject_ids[i])
        hard_negs_i = top-K valid_j by descending scores_i

    Args:
        text_emb:    (N, D) float on CPU or CUDA
        signal_emb:  (N, D) float on same device as text_emb
        subject_ids: length N, one int per row
        k_per_anchor: how many hard negatives to cache per anchor
        chunk_size:  rows-of-text processed per matmul; bounds peak VRAM at
                     chunk_size * N * 4 bytes for the score matrix.

    Returns:
        Tensor (N, k_per_anchor) of int64 indices into [0, N).
        If a row has fewer valid candidates than k_per_anchor, it is padded
        by repeating the last valid index.
    """
    if text_emb.shape != signal_emb.shape:
        raise ValueError(f"shape mismatch: {text_emb.shape} vs {signal_emb.shape}")
    n, _ = text_emb.shape
    if len(subject_ids) != n:
        raise ValueError(f"subject_ids len {len(subject_ids)} != N {n}")
    if k_per_anchor <= 0 or k_per_anchor > n - 1:
        raise ValueError(f"k_per_anchor must be in (0, N-1]; got {k_per_anchor}")

    device = text_emb.device
    sid = torch.as_tensor(list(subject_ids), dtype=torch.long, device=device)
    out = torch.empty((n, k_per_anchor), dtype=torch.long, device="cpu")

    # Need at least k+1 valid candidates per row (k negatives + safety).
    # If the patient has more than N-k notes, top-k might fall short — pad below.
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk_text = text_emb[start:end]  # (B, D)
        scores = chunk_text @ signal_emb.T  # (B, N)

        # Mask out invalid candidates (self + same subject) by setting -inf
        chunk_sid = sid[start:end].unsqueeze(1)  # (B, 1)
        same_subject = sid.unsqueeze(0) == chunk_sid  # (B, N)
        idx_self = torch.arange(start, end, device=device).unsqueeze(1)
        is_self = torch.arange(n, device=device).unsqueeze(0) == idx_self
        invalid = same_subject | is_self
        scores = scores.masked_fill(invalid, float("-inf"))

        # top-k indices per row
        # If a row has < k_per_anchor valid (-inf) entries, topk still returns
        # k indices, but some scores will be -inf — pad those by repeating the
        # last valid index (best effort).
        top_scores, top_idx = scores.topk(k_per_anchor, dim=1)  # (B, K)

        # Find rows where any -inf appeared in top-k → fix by replacing -inf
        # entries with the last finite index in that row.
        bad_mask = torch.isinf(top_scores)
        if bad_mask.any():
            for local_i in range(top_idx.size(0)):
                row_bad = bad_mask[local_i]
                if not row_bad.any():
                    continue
                good = top_idx[local_i][~row_bad]
                if good.numel() == 0:
                    # Worst case: no valid candidates. Use fallback: shift index
                    # by 1 (will be filtered later or fall back to random sampling).
                    fallback = (torch.arange(k_per_anchor, device=device) + 1) % n
                    top_idx[local_i] = fallback
                else:
                    top_idx[local_i, row_bad] = good[-1]

        out[start:end] = top_idx.cpu()

    return out


class HardNegativeBatchSampler:
    """
    DataLoader-compatible batch_sampler that yields **CLIP-style batches**:
    each batch contains ``M`` core anchors + their union of hard negatives,
    totaling ``batch_size`` (= effective InfoNCE pool).

    All ``batch_size`` items in a batch act simultaneously as "their own
    positive" (diagonal of the similarity matrix) and as in-batch negatives
    for the others. The hard-neg trick guarantees that the in-batch negatives
    are HARD (semantically close, anchor-disjoint subjects), not random.

    Constraints enforced per batch:
      - All ``batch_size`` items have unique dataset indices (no duplicates).
      - All M core anchors come from disjoint subjects (no two notes from the
        same patient compete with each other in the same batch).
      - Hard negatives are subject-disjoint from any anchor (already
        guaranteed by ``compute_hard_negative_table``'s leakage filter).

    Args:
        hard_neg_table: (N_dataset, K) tensor where row i (dataset index) lists
            top-K hard negative dataset indices. Built by
            ``compute_hard_negative_table`` and expanded to N_dataset rows.
        anchor_pool: dataset indices that may serve as core anchors (= train_idx).
        batch_size: total batch size (= effective InfoNCE pool, e.g. 64).
        anchors_per_batch: M = how many core anchors per batch (default 8).
            Must satisfy ``M <= batch_size``. Each batch builds out the
            remaining ``batch_size - M`` items from the anchors' hard-neg pools.
        subject_ids: full-dataset list of subject ids; required for
            cross-anchor subject-disjointness check.
        generator: torch.Generator for reproducibility.
    """

    def __init__(
        self,
        hard_neg_table: torch.Tensor,
        anchor_pool: Sequence[int],
        batch_size: int,
        anchors_per_batch: int = 8,
        subject_ids: Sequence[int] | None = None,
        generator: torch.Generator | None = None,
    ) -> None:
        if hard_neg_table.dim() != 2:
            raise ValueError(f"hard_neg_table must be (N, K); got {tuple(hard_neg_table.shape)}")
        if anchors_per_batch < 1 or anchors_per_batch > batch_size:
            raise ValueError(
                f"anchors_per_batch must be in [1, batch_size]; got {anchors_per_batch}"
            )
        self.hard_neg_table = hard_neg_table
        self.anchor_pool = list(anchor_pool)
        self.batch_size = int(batch_size)
        self.anchors_per_batch = int(anchors_per_batch)
        self.k_per_anchor = hard_neg_table.size(1)
        self.subject_ids = list(subject_ids) if subject_ids is not None else None
        self.generator = generator

    def __len__(self) -> int:
        # Each batch consumes M anchors from anchor_pool (without replacement)
        return len(self.anchor_pool) // self.anchors_per_batch

    def _next_int(self, n: int) -> int:
        if self.generator is not None:
            return int(torch.randint(n, (1,), generator=self.generator).item())
        return int(torch.randint(n, (1,)).item())

    def _perm(self, n: int) -> list[int]:
        if self.generator is not None:
            return torch.randperm(n, generator=self.generator).tolist()
        return torch.randperm(n).tolist()

    def __iter__(self) -> Iterator[list[int]]:
        order = self._perm(len(self.anchor_pool))
        cursor = 0
        n_extra = self.batch_size - self.anchors_per_batch  # negatives per batch

        while cursor + self.anchors_per_batch <= len(order):
            # 1) Pick M core anchors with disjoint subjects
            anchors: list[int] = []
            anchor_subjects: set[int] = set()
            scan = cursor
            while len(anchors) < self.anchors_per_batch and scan < len(order):
                cand = self.anchor_pool[order[scan]]
                if self.subject_ids is None or self.subject_ids[cand] not in anchor_subjects:
                    anchors.append(cand)
                    if self.subject_ids is not None:
                        anchor_subjects.add(self.subject_ids[cand])
                scan += 1
            cursor = scan
            if len(anchors) < self.anchors_per_batch:
                break  # not enough disjoint-subject anchors left this epoch

            # 2) Build pool of hard negatives: union over anchors,
            #    deduplicate, exclude anchors themselves, exclude any sample
            #    sharing subject with any anchor.
            anchor_set = set(anchors)
            negs_set: set[int] = set()
            for a in anchors:
                row = self.hard_neg_table[a].tolist()
                for n in row:
                    if n in anchor_set or n in negs_set:
                        continue
                    if self.subject_ids is not None:
                        if self.subject_ids[n] in anchor_subjects:
                            continue
                    negs_set.add(n)
                    if len(negs_set) >= n_extra * 3:  # over-collect, then sample
                        break
                if len(negs_set) >= n_extra * 3:
                    break

            negs_list = list(negs_set)

            # 3) Sample exactly n_extra without replacement, supplementing from
            #    random anchor_pool elements if hard pool is too small.
            if len(negs_list) >= n_extra:
                idx = self._perm(len(negs_list))[:n_extra]
                chosen_negs = [negs_list[i] for i in idx]
            else:
                chosen_negs = list(negs_list)
                n_extra - len(chosen_negs)
                avoid = anchor_set | negs_set
                # walk a permutation of anchor_pool to fill the rest
                fillers_pool = self._perm(len(self.anchor_pool))
                for fp in fillers_pool:
                    cand = self.anchor_pool[fp]
                    if cand in avoid:
                        continue
                    if self.subject_ids is not None and self.subject_ids[cand] in anchor_subjects:
                        continue
                    chosen_negs.append(cand)
                    avoid.add(cand)
                    if len(chosen_negs) >= n_extra:
                        break

            yield anchors + chosen_negs


def diagnose_hard_neg_table(
    table: torch.Tensor,
    subject_ids: Sequence[int],
    *,
    sample_n: int = 256,
    rng: np.random.Generator | None = None,
) -> dict:
    """
    Sanity-check a hard negative table:
      - Should have NO same-subject indices (leakage check)
      - Should have NO self-references
      - Should have unique indices per anchor (within K)
    Returns a dict of stats; does NOT raise — caller decides what to do.
    """
    rng = rng or np.random.default_rng(0)
    n = table.size(0)
    sid = np.asarray(list(subject_ids))
    sample = rng.choice(n, size=min(sample_n, n), replace=False)

    leak_count = 0
    self_count = 0
    dup_count = 0
    for i in sample:
        negs = table[i].numpy()
        if (negs == i).any():
            self_count += 1
        if (sid[negs] == sid[i]).any():
            leak_count += 1
        if len(set(negs.tolist())) != len(negs):
            dup_count += 1

    return {
        "sampled": int(len(sample)),
        "self_refs": int(self_count),
        "same_subject_leakage": int(leak_count),
        "rows_with_dup_idx": int(dup_count),
        "k_per_anchor": int(table.size(1)),
    }
