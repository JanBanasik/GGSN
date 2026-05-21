"""
Phase 1 snapshot and evaluation helpers.

Extracted from train_contrastive.py for modularity.

Public API:
    compute_all_embeddings             — {note_id -> Tensor} from text_tower
    compute_text_signal_embeddings_arr — aligned (text_emb, sig_emb, subject_ids) arrays
    compute_similarity_matrix          — NxN cosine-similarity matrix
    evaluate_val_loss                  — val InfoNCE (macro or full-batch mode)
    take_snapshot                      — save per-epoch artifacts for animation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data_prep.contrastive_dataset import ContrastivePairsDataset
from src.models.towers import SignalTower, TextTower
from src.training.loss import info_nce_loss


@torch.no_grad()
def compute_all_embeddings(
    text_tower: TextTower,
    dataset: ContrastivePairsDataset,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run text_tower over the given indices. Returns {note_id: Tensor(D,)}."""
    text_tower.eval()
    out: dict[str, torch.Tensor] = {}
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        items = [dataset[i] for i in chunk]
        input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
        attention_mask = torch.stack([it["attention_mask"] for it in items]).to(device)
        z = text_tower(input_ids, attention_mask).cpu()
        for it, emb in zip(items, z, strict=False):
            out[it["note_id"]] = emb
    return out


@torch.no_grad()
def compute_text_signal_embeddings_arr(
    text_tower: TextTower,
    signal_tower: SignalTower,
    dataset: ContrastivePairsDataset,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """
    Compute (text_emb, signal_emb, subject_ids) as parallel tensors.
    Used by hard-negative mining (needs both modalities + subject_id for leakage filter).
    """
    text_tower.eval()
    signal_tower.eval()
    text_chunks: list[torch.Tensor] = []
    sig_chunks: list[torch.Tensor] = []
    subject_ids: list[int] = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        items = [dataset[i] for i in chunk]
        input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
        attention_mask = torch.stack([it["attention_mask"] for it in items]).to(device)
        item_ids = torch.stack([it["item_ids"] for it in items]).to(device)
        values = torch.stack([it["values"] for it in items]).to(device)
        signal_mask = torch.stack([it["signal_mask"] for it in items]).to(device)
        hours = torch.stack([it["hours"] for it in items]).to(device)
        deltas = torch.stack([it["deltas"] for it in items]).to(device)
        text_chunks.append(text_tower(input_ids, attention_mask).cpu())
        sig_chunks.append(signal_tower(item_ids, values, signal_mask, hours, deltas).cpu())
        subject_ids.extend(it["subject_id"] for it in items)
    return torch.cat(text_chunks, dim=0), torch.cat(sig_chunks, dim=0), subject_ids


@torch.no_grad()
def compute_similarity_matrix(
    text_tower: TextTower,
    signal_tower: SignalTower,
    dataset: ContrastivePairsDataset,
    indices: list[int],
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """
    NxN cosine-similarity matrix between text and signal embeddings.
    Returns (matrix, note_ids).
    """
    text_tower.eval()
    signal_tower.eval()
    items = [dataset[i] for i in indices]
    input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
    attention_mask = torch.stack([it["attention_mask"] for it in items]).to(device)
    item_ids = torch.stack([it["item_ids"] for it in items]).to(device)
    values = torch.stack([it["values"] for it in items]).to(device)
    signal_mask = torch.stack([it["signal_mask"] for it in items]).to(device)
    hours = torch.stack([it["hours"] for it in items]).to(device)
    deltas = torch.stack([it["deltas"] for it in items]).to(device)
    z_text = text_tower(input_ids, attention_mask)
    z_sig = signal_tower(item_ids, values, signal_mask, hours, deltas)
    sim = (z_text @ z_sig.T).cpu().numpy()
    return sim, [it["note_id"] for it in items]


@torch.no_grad()
def evaluate_val_loss(
    text_tower: TextTower,
    signal_tower: SignalTower,
    val_loader: DataLoader,
    temperature: float,
    device: torch.device,
    grad_accum_steps: int,
    amp_enabled: bool,
    val_loss_mode: str = "macro",
) -> tuple[float, dict]:
    """
    Val InfoNCE aligned with training when val_loss_mode='macro': concatenate
    micro-batches every grad_accum_steps (same N as effective_batch during train).

    Returns (mean_loss, stats).
    """
    text_tower.eval()
    signal_tower.eval()

    if val_loss_mode == "full":
        return _eval_val_loss_full_batch(
            text_tower, signal_tower, val_loader, temperature, device, amp_enabled
        )
    if val_loss_mode != "macro":
        raise ValueError(f"val_loss_mode must be 'macro' or 'full', got {val_loss_mode!r}")

    losses: list[float] = []
    chunk_sizes: list[int] = []
    z_text_buf: list[torch.Tensor] = []
    z_sig_buf: list[torch.Tensor] = []

    def _flush() -> None:
        if not z_text_buf:
            return
        z_text_cat = torch.cat(z_text_buf, dim=0)
        z_sig_cat = torch.cat(z_sig_buf, dim=0)
        z_text_buf.clear()
        z_sig_buf.clear()
        if z_text_cat.size(0) < 2:
            return
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            loss = info_nce_loss(z_text_cat, z_sig_cat, temperature)
        losses.append(loss.item())
        chunk_sizes.append(int(z_text_cat.size(0)))

    n_batches = len(val_loader)
    for step, batch in enumerate(val_loader, 1):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        item_ids = batch["item_ids"].to(device, non_blocking=True)
        values = batch["values"].to(device, non_blocking=True)
        signal_mask = batch["signal_mask"].to(device, non_blocking=True)
        hours = batch["hours"].to(device, non_blocking=True)
        deltas = batch["deltas"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            z_text = text_tower(input_ids, attention_mask)
            z_sig = signal_tower(item_ids, values, signal_mask, hours, deltas)
        z_text_buf.append(z_text)
        z_sig_buf.append(z_sig)
        if (step % grad_accum_steps == 0) or (step == n_batches):
            _flush()

    if not losses:
        return float("nan"), {"val_infonce_n_mean": 0, "val_infonce_macro_chunks": 0}
    return float(np.mean(losses)), {
        "val_infonce_n_mean": float(np.mean(chunk_sizes)),
        "val_infonce_macro_chunks": len(losses),
    }


@torch.no_grad()
def _eval_val_loss_full_batch(
    text_tower: TextTower,
    signal_tower: SignalTower,
    val_loader: DataLoader,
    temperature: float,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict]:
    """Single InfoNCE over all validation embeddings (largest possible negative set)."""
    text_tower.eval()
    signal_tower.eval()
    z_text_chunks, z_sig_chunks = [], []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        item_ids = batch["item_ids"].to(device, non_blocking=True)
        values = batch["values"].to(device, non_blocking=True)
        signal_mask = batch["signal_mask"].to(device, non_blocking=True)
        hours = batch["hours"].to(device, non_blocking=True)
        deltas = batch["deltas"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            z_text_chunks.append(text_tower(input_ids, attention_mask))
            z_sig_chunks.append(signal_tower(item_ids, values, signal_mask, hours, deltas))
    if not z_text_chunks:
        return float("nan"), {"val_infonce_n_used": 0}
    z_text = torch.cat(z_text_chunks, dim=0)
    z_sig = torch.cat(z_sig_chunks, dim=0)
    n = int(z_text.size(0))
    if n < 2:
        return float("nan"), {"val_infonce_n_used": n}
    with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
        loss = info_nce_loss(z_text, z_sig, temperature)
    return loss.item(), {"val_infonce_n_used": n}


def take_snapshot(
    run_dir: Path,
    epoch: int,
    text_tower: TextTower,
    signal_tower: SignalTower,
    dataset: ContrastivePairsDataset,
    val_idx: list[int],
    sim_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> None:
    """Write per-epoch artifacts for animation: embeddings + similarity matrix."""
    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    val_embeds = compute_all_embeddings(text_tower, dataset, val_idx, batch_size, device)
    torch.save(val_embeds, epoch_dir / "val_embeddings.pt")

    labels = {dataset.note_ids[i]: dataset.mortality[i] for i in val_idx}
    (epoch_dir / "val_labels.json").write_text(json.dumps(labels))

    sim, note_ids = compute_similarity_matrix(
        text_tower, signal_tower, dataset, sim_indices, device
    )
    np.save(epoch_dir / "similarity_matrix.npy", sim)
    (epoch_dir / "similarity_note_ids.json").write_text(json.dumps(note_ids))
