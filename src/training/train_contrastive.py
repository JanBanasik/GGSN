"""
Phase 1: Multimodal Contrastive Pre-training (Self-Supervised).

Pipeline:
  cardio_pairs.csv  →  CardiacPairsDataset  →  Two-Tower (BERT + CNN)
                    →  InfoNCE loss  →  trained encoders
                    →  per-epoch snapshots (embeddings + similarity matrix)
                    →  final node_embeddings.pt for Phase 2 (GNN)

Usage:
    uv run python -m src.training.train_contrastive
    uv run python -m src.training.train_contrastive --epochs 25 --batch-size 16 --grad-accum-steps 4
    uv run python -m src.training.train_contrastive --preset low-data-bert
    uv run python -m src.training.train_contrastive --val-loss-mode full
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer

from src.models.towers import (
    BERT_MODEL,
    EMBED_DIM,
    NUM_SIGNAL_TYPES,
    SignalTower,
    TextTower,
)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 surowego pliku (np. cardio_pairs.csv) — ten sam plik = ten sam hash między osobami."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _grad_scaler_cuda(enabled: bool):
    """torch.amp.GradScaler exists from ~2.4; torch 2.2 ma tylko torch.cuda.amp.GradScaler."""
    amp_gs = getattr(torch.amp, "GradScaler", None)
    if amp_gs is not None:
        return amp_gs("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Checkpoint helpers (resume training across sessions)
# ---------------------------------------------------------------------------
CHECKPOINT_NAME = "checkpoint.pt"
# Hyperparams that, if changed at resume time, invalidate the optimizer/scheduler
# state (different param shapes or different LR-schedule trajectory). Other fields
# (epochs, max_time_hours, early_stop_patience) can change freely.
CKPT_LOCKED_KEYS = (
    "batch_size", "grad_accum_steps", "embed_dim", "freeze_bert",
    "freeze_bottom_layers", "proj_dropout", "lr_bert", "lr_head",
    "temperature", "seed", "val_frac",
)


def save_checkpoint(path: Path, **state) -> None:
    """Atomic-ish: write to .tmp then rename, so a Ctrl-C mid-write doesn't corrupt."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load on CPU first then move per-tensor to avoid VRAM spike (see plan, edge cases)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def restore_rng(ckpt: dict) -> None:
    torch.set_rng_state(ckpt["torch_rng"])
    if ckpt.get("torch_cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    random.setstate(ckpt["python_rng"])


def collect_rng_state() -> dict:
    return {
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def validate_resume_config(ckpt_config: dict, current_config: dict, force: bool) -> None:
    """
    At resume time: compare locked hyperparams. If any differ → raise unless --force.

    Locked = anything that would invalidate optimizer/scheduler state (param shapes,
    LR trajectory). Free to change between sessions: epochs, max_time_hours,
    early_stop_patience, sim_matrix_n.
    """
    diffs = {
        k: (ckpt_config.get(k), current_config.get(k))
        for k in CKPT_LOCKED_KEYS
        if ckpt_config.get(k) != current_config.get(k)
    }
    if not diffs:
        return
    msg_lines = ["[resume] Locked hyperparam mismatch vs checkpoint:"]
    for k, (old, new) in diffs.items():
        msg_lines.append(f"  {k}: ckpt={old!r}  cli={new!r}")
    msg_lines.append("Pass --force to override (optimizer/scheduler may be invalid).")
    msg = "\n".join(msg_lines)
    if not force:
        raise RuntimeError(msg)
    print(f"[resume] WARNING (--force):\n{msg}")


# ---------------------------------------------------------------------------
# Hyper-parameters (overridable via CLI)
# ---------------------------------------------------------------------------
EPOCHS = 25
BATCH_SIZE = 16
MAX_TEXT_LEN = 256
SEQ_LEN = 64                 # max signal events per note
LR_BERT = 5e-6               # was 2e-5; lowered after observed overfit (epoch 2 was best)
LR_HEAD = 2e-4
TEMPERATURE = 0.07
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
VAL_FRAC = 0.10              # 10% of subjects held out
SEED = 42
SIM_MATRIX_N = 64            # NxN similarity snapshot for animation
FREEZE_BOTTOM_LAYERS = 8     # 0=fine-tune all 12 BERT layers; 8=fine-tune only top 4
PROJ_DROPOUT = 0.2           # dropout in TextTower projection head
GRAD_ACCUM_STEPS = 4         # effective batch = batch_size * grad_accum_steps (align val InfoNCE to this N)
FREEZE_BERT = False          # if True, freeze all 12 BERT layers (overrides FREEZE_BOTTOM_LAYERS)

# Val InfoNCE: "macro" = buffer micro-batches like train (same N as effective_batch when aligned);
#            "full" = one InfoNCE over all val embeddings (max negative pool; needs VRAM for matmul).
VAL_LOSS_MODE = "macro"


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_sha() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CardiacPairsDataset(Dataset):
    """
    Reads cardio_pairs.csv (one row per note × signal pair) and collapses to
    one sample per unique note_id.

    Each sample:
        input_ids       (MAX_TEXT_LEN,) long
        attention_mask  (MAX_TEXT_LEN,) long
        item_ids        (SEQ_LEN,)      long  – signal type indices
        values          (SEQ_LEN,)      float – normalised values in [0,1]
        signal_mask     (SEQ_LEN,)      float – 1 for real event, 0 for pad
        hours           (SEQ_LEN,)      float – event_hours_from_intime / 24 ∈ [0,1]
                                                 (zeros when CSV doesn't have this column)
        note_id         str
        subject_id      int
        mortality       int
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer,
        max_text_len: int = MAX_TEXT_LEN,
        seq_len: int = SEQ_LEN,
    ) -> None:
        df = pl.read_csv(csv_path)
        has_hours = "event_hours_from_intime" in df.columns

        # Collect ordered events per note
        per_note: dict[str, dict] = {}
        for row in df.sort("event_time").iter_rows(named=True):
            nid = row["note_id"]
            entry = per_note.setdefault(nid, {
                "text": row["text"],
                "subject_id": int(row["subject_id"]),
                "mortality": int(row["mortality"]),
                "items": [],
                "values": [],
                "hours": [],
            })
            entry["items"].append(int(row["item_type_id"]))
            entry["values"].append(float(row["norm_value"]))
            if has_hours:
                # Normalize to [0, 1] by /24 — extractor already clips to [0, 24]
                entry["hours"].append(float(row["event_hours_from_intime"]) / 24.0)
            else:
                entry["hours"].append(0.0)

        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.seq_len = seq_len
        self.has_hours = has_hours

        self.note_ids: list[str] = []
        self.encodings: list[dict] = []
        self.item_ids: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.hours: list[torch.Tensor] = []
        self.signal_masks: list[torch.Tensor] = []
        self.subject_ids: list[int] = []
        self.mortality: list[int] = []

        print(f"  Tokenising {len(per_note)} unique notes… (hours column: {has_hours})")
        for nid, entry in per_note.items():
            enc = tokenizer(
                entry["text"],
                max_length=max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            items = entry["items"][:seq_len]
            values = entry["values"][:seq_len]
            hours_seq = entry["hours"][:seq_len]
            real_n = len(items)
            pad_n = seq_len - real_n
            if pad_n > 0:
                items = items + [0] * pad_n
                values = values + [0.0] * pad_n
                hours_seq = hours_seq + [0.0] * pad_n

            mask = [1.0] * real_n + [0.0] * pad_n

            self.note_ids.append(nid)
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            })
            self.item_ids.append(torch.tensor(items, dtype=torch.long))
            self.values.append(torch.tensor(values, dtype=torch.float32))
            self.hours.append(torch.tensor(hours_seq, dtype=torch.float32))
            self.signal_masks.append(torch.tensor(mask, dtype=torch.float32))
            self.subject_ids.append(entry["subject_id"])
            self.mortality.append(entry["mortality"])

    def __len__(self) -> int:
        return len(self.note_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings[idx]["input_ids"],
            "attention_mask": self.encodings[idx]["attention_mask"],
            "item_ids": self.item_ids[idx],
            "values": self.values[idx],
            "hours": self.hours[idx],
            "signal_mask": self.signal_masks[idx],
            "note_id": self.note_ids[idx],
            "subject_id": self.subject_ids[idx],
            "mortality": self.mortality[idx],
        }


def split_indices_by_subject(
    dataset: CardiacPairsDataset, val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """
    Train/val split that keeps all notes from a subject in the SAME split
    (prevents subject-level leakage during contrastive training).
    """
    rng = np.random.default_rng(seed)
    subjects = sorted(set(dataset.subject_ids))
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * val_frac))
    val_subjects = set(subjects[:n_val])

    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, sid in enumerate(dataset.subject_ids):
        (val_idx if sid in val_subjects else train_idx).append(i)

    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def info_nce_loss(z_text: torch.Tensor, z_signal: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent) loss over in-batch positives.
    z_text, z_signal are L2-normalised (N, D).
    """
    logits = torch.matmul(z_text, z_signal.T) / temperature  # (N, N)
    labels = torch.arange(z_text.size(0), device=z_text.device)
    loss_t2s = F.cross_entropy(logits, labels)
    loss_s2t = F.cross_entropy(logits.T, labels)
    return (loss_t2s + loss_s2t) / 2.0


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_all_embeddings(
    text_tower: TextTower,
    dataset: CardiacPairsDataset,
    indices: list[int],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Run text_tower over the given indices, return {note_id: tensor(D,)}."""
    text_tower.eval()
    out: dict[str, torch.Tensor] = {}
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        items = [dataset[i] for i in chunk]
        input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
        attention_mask = torch.stack([it["attention_mask"] for it in items]).to(device)
        z = text_tower(input_ids, attention_mask).cpu()
        for it, emb in zip(items, z):
            out[it["note_id"]] = emb
    return out


@torch.no_grad()
def compute_similarity_matrix(
    text_tower: TextTower,
    signal_tower: SignalTower,
    dataset: CardiacPairsDataset,
    indices: list[int],
    device: torch.device,
) -> tuple[np.ndarray, list[str]]:
    """
    For a fixed list of `indices` (size N), compute NxN cos-sim matrix between
    text and signal embeddings. Returns (matrix, note_ids).
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

    z_text = text_tower(input_ids, attention_mask)              # (N, D)
    z_sig = signal_tower(item_ids, values, signal_mask, hours)  # (N, D)
    sim = (z_text @ z_sig.T).cpu().numpy()
    note_ids = [it["note_id"] for it in items]
    return sim, note_ids


@torch.no_grad()
def evaluate_val_loss(
    text_tower: TextTower,
    signal_tower: SignalTower,
    val_loader: DataLoader,
    temperature: float,
    device: torch.device,
    grad_accum_steps: int,
    amp_enabled: bool,
    val_loss_mode: str = VAL_LOSS_MODE,
) -> tuple[float, dict[str, float | int]]:
    """
    Val InfoNCE aligned with training when val_loss_mode='macro': concatenate
    micro-batches every grad_accum_steps (and remainder at end), same as train.

    Returns (mean_loss, stats) where stats includes val_infonce_n_used for logging.
    """
    text_tower.eval()
    signal_tower.eval()

    if val_loss_mode == "full":
        return _evaluate_val_loss_full_batch(
            text_tower, signal_tower, val_loader, temperature, device, amp_enabled
        )

    if val_loss_mode != "macro":
        raise ValueError(f"val_loss_mode must be 'macro' or 'full', got {val_loss_mode!r}")

    losses: list[float] = []
    chunk_sizes: list[int] = []
    z_text_buf: list[torch.Tensor] = []
    z_sig_buf: list[torch.Tensor] = []

    def flush() -> None:
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
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            z_text = text_tower(input_ids, attention_mask)
            z_sig = signal_tower(item_ids, values, signal_mask, hours)
        z_text_buf.append(z_text)
        z_sig_buf.append(z_sig)
        is_step = (step % grad_accum_steps == 0) or (step == n_batches)
        if is_step:
            flush()

    if not losses:
        stats = {"val_infonce_n_mean": 0, "val_infonce_macro_chunks": 0}
        return float("nan"), stats

    mean_loss = float(np.mean(losses))
    stats = {
        "val_infonce_n_mean": float(np.mean(chunk_sizes)),
        "val_infonce_macro_chunks": len(losses),
    }
    return mean_loss, stats


@torch.no_grad()
def _evaluate_val_loss_full_batch(
    text_tower: TextTower,
    signal_tower: SignalTower,
    val_loader: DataLoader,
    temperature: float,
    device: torch.device,
    amp_enabled: bool,
) -> tuple[float, dict[str, float | int]]:
    """Single InfoNCE over all validation embeddings (largest possible negative set)."""
    text_tower.eval()
    signal_tower.eval()
    z_text_chunks: list[torch.Tensor] = []
    z_sig_chunks: list[torch.Tensor] = []
    for batch in val_loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        item_ids = batch["item_ids"].to(device, non_blocking=True)
        values = batch["values"].to(device, non_blocking=True)
        signal_mask = batch["signal_mask"].to(device, non_blocking=True)
        hours = batch["hours"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
            z_text_chunks.append(text_tower(input_ids, attention_mask))
            z_sig_chunks.append(signal_tower(item_ids, values, signal_mask, hours))
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


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    csv_path: Path,
    snapshots_root: Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr_bert: float = LR_BERT,
    lr_head: float = LR_HEAD,
    embed_dim: int = EMBED_DIM,
    temperature: float = TEMPERATURE,
    seed: int = SEED,
    val_frac: float = VAL_FRAC,
    use_amp: bool = True,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    sim_matrix_n: int = SIM_MATRIX_N,
    freeze_bottom_layers: int = FREEZE_BOTTOM_LAYERS,
    proj_dropout: float = PROJ_DROPOUT,
    grad_accum_steps: int = GRAD_ACCUM_STEPS,
    freeze_bert: bool = FREEZE_BERT,
    val_loss_mode: str = VAL_LOSS_MODE,
    training_preset: str = "none",
    cli_argv: list[str] | None = None,
    resume_from: str | None = None,
    max_time_hours: float | None = None,
    force: bool = False,
) -> Path:
    """
    Main training entry. Returns path to the run directory.

    Args:
        resume_from: path to existing run dir; loads checkpoint.pt and continues.
            Run continues IN THE SAME directory (log.csv appended).
        max_time_hours: stop cleanly after this many hours wall-clock; checkpoint
            written before exit. Resume command printed for next session.
        force: allow resume despite locked-hyperparam mismatch.
    """
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"AMP    : {use_amp and device.type == 'cuda'}")

    # Run directory: new run OR continuation of existing one
    is_resume = resume_from is not None
    if is_resume:
        run_dir = Path(resume_from).resolve()
        if not (run_dir / CHECKPOINT_NAME).exists():
            raise FileNotFoundError(f"No {CHECKPOINT_NAME} in {run_dir}")
        run_id = run_dir.name.removeprefix("run_")
        print(f"Run dir: {run_dir}  (RESUME)")
    else:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = snapshots_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"Run dir: {run_dir}")

    # --- Data ---
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    dataset = CardiacPairsDataset(csv_path, tokenizer)
    train_idx, val_idx = split_indices_by_subject(dataset, val_frac, seed)
    print(f"  Train notes: {len(train_idx)} | Val notes: {len(val_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    # drop_last=False: remainder is merged into last macro-batch (same as train's end-of-epoch flush)
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Fixed indices for similarity-matrix snapshots (NxN, sorted note_ids for stability)
    sim_pool = sorted(val_idx, key=lambda i: dataset.note_ids[i])
    sim_indices = sim_pool[:sim_matrix_n] if len(sim_pool) >= sim_matrix_n else sim_pool
    print(f"  Similarity-matrix snapshot N: {len(sim_indices)}")

    # --- Models ---
    text_tower = TextTower(
        embed_dim=embed_dim,
        freeze_bert=freeze_bert,
        freeze_bottom_layers=freeze_bottom_layers if not freeze_bert else 0,
        proj_dropout=proj_dropout,
    ).to(device)
    signal_tower = SignalTower(num_item_types=NUM_SIGNAL_TYPES, embed_dim=embed_dim).to(device)

    n_bert_total = sum(p.numel() for p in text_tower.bert.parameters())
    n_bert_train = sum(p.numel() for p in text_tower.bert.parameters() if p.requires_grad)
    print(f"  BERT trainable: {n_bert_train / 1e6:.1f}M / {n_bert_total / 1e6:.1f}M params "
          f"(freeze_bottom_layers={freeze_bottom_layers}, proj_dropout={proj_dropout})")

    bert_trainable = [p for p in text_tower.bert.parameters() if p.requires_grad]
    optimizer_groups = [
        {"params": text_tower.proj.parameters(), "lr": lr_head},
        {"params": signal_tower.parameters(), "lr": lr_head},
    ]
    if bert_trainable:
        optimizer_groups.insert(0, {"params": bert_trainable, "lr": lr_bert})

    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * max(1, len(train_loader))
    )

    amp_enabled = use_amp and device.type == "cuda"
    scaler = _grad_scaler_cuda(amp_enabled)

    # --- Run config (saved upfront so a crashed run is still self-describing) ---
    config = {
        "run_id": run_id,
        "git_sha": git_sha(),
        "csv_path": str(csv_path),
        "n_total_notes": len(dataset),
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "n_subjects_total": len(set(dataset.subject_ids)),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr_bert": lr_bert,
        "lr_head": lr_head,
        "embed_dim": embed_dim,
        "temperature": temperature,
        "seed": seed,
        "val_frac": val_frac,
        "max_text_len": MAX_TEXT_LEN,
        "seq_len": SEQ_LEN,
        "num_signal_types": NUM_SIGNAL_TYPES,
        "amp": amp_enabled,
        "freeze_bottom_layers": freeze_bottom_layers,
        "freeze_bert": freeze_bert,
        "proj_dropout": proj_dropout,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch": int(batch_size * grad_accum_steps),
        "bert_trainable_params": int(n_bert_train),
        "bert_total_params": int(n_bert_total),
        "infonce_baseline_ln_train": float(np.log(batch_size * grad_accum_steps)),
        "val_loss_mode": val_loss_mode,
        "val_macro_accum_steps": grad_accum_steps,
        "infonce_baseline_ln_val_macro_aligned": float(np.log(batch_size * grad_accum_steps)),
        "training_preset": training_preset,
        "cli_argv": list(cli_argv) if cli_argv else [],
        "cardio_pairs_sha256": _sha256_file(csv_path) if csv_path.is_file() else None,
    }
    # --- Resume vs new run: write/load config + checkpoint ---
    log_path = run_dir / "log.csv"
    start_epoch = 1
    best_val = float("inf")
    bad_epochs = 0

    if is_resume:
        ckpt_path = run_dir / CHECKPOINT_NAME
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path, device)

        # Validate that we're not changing locked hyperparams
        validate_resume_config(ckpt["config"], config, force=force)

        # Update config in-place: keep checkpoint's locked fields, but allow
        # epochs / max_time / patience to come from current CLI
        config = {**ckpt["config"], "epochs": epochs, "training_preset": training_preset,
                  "cli_argv": list(cli_argv) if cli_argv else ckpt["config"].get("cli_argv", [])}
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))

        text_tower.load_state_dict(ckpt["text_tower"])
        signal_tower.load_state_dict(ckpt["signal_tower"])
        text_tower.to(device)
        signal_tower.to(device)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        restore_rng(ckpt)

        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt["best_val"])
        bad_epochs = int(ckpt["bad_epochs"])

        if start_epoch > epochs:
            raise RuntimeError(
                f"Checkpoint already at epoch {ckpt['epoch']} >= target {epochs}. "
                f"Increase --epochs or use a different run dir."
            )
        print(f"  Resumed from epoch {ckpt['epoch']}; continuing {start_epoch}..{epochs} "
              f"(best_val={best_val:.4f}, bad_epochs={bad_epochs})")

        if cli_argv:
            quoted = " ".join(shlex.quote(a) for a in cli_argv)
            with (run_dir / "RUN_COMMAND.txt").open("a") as f:
                f.write(f"\n# Resume session: {quoted}\n")
    else:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        if cli_argv:
            quoted = " ".join(shlex.quote(a) for a in cli_argv)
            (run_dir / "RUN_COMMAND.txt").write_text(
                f"{quoted}\n\n"
                "# Powiel na innej maszynie (ten sam repo, ten sam cardio_pairs.csv po sha256).\n"
            )
        log_path.write_text(
            "epoch,train_loss,val_loss,val_infonce_n_mean,lr,elapsed_s\n"
        )

        # Snapshot epoch 0 (pre-training baseline) — only for fresh run
        print("\nSnapshot epoch 0 (pre-training)…")
        _take_snapshot(
            run_dir, 0, text_tower, signal_tower, dataset, val_idx, sim_indices,
            batch_size, device,
        )

    # --- Training ---
    effective_batch = batch_size * grad_accum_steps
    print(f"\nTraining epochs {start_epoch}..{epochs} | train={len(train_idx)} val={len(val_idx)} "
          f"| batch={batch_size} × accum={grad_accum_steps} = effective {effective_batch} "
          f"| InfoNCE baseline = ln({effective_batch}) = {np.log(effective_batch):.3f}\n"
          f"| val_loss_mode={val_loss_mode} (macro aligns val InfoNCE batching with train)\n")

    # Wall-clock budget for partial training across sessions
    session_start = time.time()
    max_seconds = float(max_time_hours) * 3600.0 if max_time_hours else float("inf")
    stopped_by_max_time = False

    # When grad_accum > 1, we collect z_text/z_signal across micro-batches and
    # compute InfoNCE on the concatenated tensor — that is what makes
    # gradient accumulation a real "larger batch" for contrastive losses
    # (naive accumulation of per-microbatch InfoNCE would NOT enlarge the
    # negative pool, only smooth the gradient).
    for epoch in range(start_epoch, epochs + 1):
        text_tower.train()
        signal_tower.train()

        epoch_loss = 0.0
        n_optim_steps = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        z_text_buf: list[torch.Tensor] = []
        z_sig_buf: list[torch.Tensor] = []

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            item_ids = batch["item_ids"].to(device, non_blocking=True)
            values = batch["values"].to(device, non_blocking=True)
            signal_mask = batch["signal_mask"].to(device, non_blocking=True)
            hours = batch["hours"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                z_text = text_tower(input_ids, attention_mask)
                z_signal = signal_tower(item_ids, values, signal_mask, hours)

            z_text_buf.append(z_text)
            z_sig_buf.append(z_signal)

            is_optim_step = (step % grad_accum_steps == 0) or (step == len(train_loader))
            if not is_optim_step:
                continue

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                z_text_cat = torch.cat(z_text_buf, dim=0)
                z_sig_cat = torch.cat(z_sig_buf, dim=0)
                loss = info_nce_loss(z_text_cat, z_sig_cat, temperature)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(text_tower.parameters()) + list(signal_tower.parameters()),
                GRAD_CLIP,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            n_optim_steps += 1
            z_text_buf.clear()
            z_sig_buf.clear()

            if n_optim_steps % 25 == 0 or step == len(train_loader):
                print(f"  Epoch {epoch}/{epochs} | optim_step {n_optim_steps:4d} "
                      f"(micro {step}/{len(train_loader)}) | loss {loss.item():.4f}")

        train_loss = epoch_loss / max(1, n_optim_steps)
        val_loss, val_stats = evaluate_val_loss(
            text_tower,
            signal_tower,
            val_loader,
            temperature,
            device,
            grad_accum_steps=grad_accum_steps,
            amp_enabled=amp_enabled,
            val_loss_mode=val_loss_mode,
        )
        if "val_infonce_n_used" in val_stats:
            val_n = float(val_stats["val_infonce_n_used"])
        else:
            val_n = float(val_stats.get("val_infonce_n_mean", 0.0))
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| val_N≈{val_n:.1f} | lr {current_lr:.2e} | {elapsed:.1f}s\n")

        with log_path.open("a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{float(val_n):.2f},"
                f"{current_lr:.6e},{elapsed:.2f}\n"
            )

        # Per-epoch snapshot
        _take_snapshot(
            run_dir, epoch, text_tower, signal_tower, dataset, val_idx, sim_indices,
            batch_size, device,
        )

        # Early stopping (must update best_val/bad_epochs BEFORE saving checkpoint)
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            torch.save(text_tower.state_dict(), run_dir / "best_text_tower.pt")
            torch.save(signal_tower.state_dict(), run_dir / "best_signal_tower.pt")
        else:
            bad_epochs += 1

        # Save full resume-checkpoint (overwrites previous epoch's checkpoint)
        save_checkpoint(
            run_dir / CHECKPOINT_NAME,
            text_tower=text_tower.state_dict(),
            signal_tower=signal_tower.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(),
            epoch=epoch,
            best_val=best_val,
            bad_epochs=bad_epochs,
            config=config,
            **collect_rng_state(),
        )

        if bad_epochs >= early_stop_patience:
            print(f"Early stopping at epoch {epoch} (no val improvement for {early_stop_patience} epochs).")
            break

        # Wall-clock budget check (graceful stop, checkpoint already saved)
        if time.time() - session_start >= max_seconds:
            stopped_by_max_time = True
            elapsed_h = (time.time() - session_start) / 3600.0
            print(f"\n[max-time] Stopped after {elapsed_h:.2f}h (limit {max_time_hours}h).")
            print(f"[max-time] Resume next session with:")
            print(f"           uv run python -m src.training.train_contrastive "
                  f"--resume {run_dir} --epochs {epochs}"
                  + (f" --max-time-hours {max_time_hours}" if max_time_hours else ""))
            break

    # --- Final artifacts (skip if stopped by --max-time-hours; resume + finish later) ---
    if stopped_by_max_time:
        print(f"\n[max-time] Skipping final node_embeddings export (incomplete training).")
        print(f"[max-time] Resume to finish — final artifacts will be exported then.")
        return run_dir

    torch.save(text_tower.state_dict(), run_dir / "final_text_tower.pt")
    torch.save(signal_tower.state_dict(), run_dir / "final_signal_tower.pt")

    # Export node embeddings for ALL notes (Phase 2 input)
    print("Exporting full node_embeddings.pt for all notes…")
    all_idx = list(range(len(dataset)))
    full_embeds = compute_all_embeddings(text_tower, dataset, all_idx, batch_size, device)
    torch.save(full_embeds, run_dir / "node_embeddings.pt")

    # Also copy to canonical Phase-2 location
    canonical = csv_path.parent.parent / "embeddings"
    canonical.mkdir(parents=True, exist_ok=True)
    torch.save(full_embeds, canonical / "node_embeddings.pt")
    torch.save(text_tower.state_dict(), canonical / "text_tower.pt")
    torch.save(signal_tower.state_dict(), canonical / "signal_tower.pt")
    print(f"Final artifacts → {run_dir} | canonical copies → {canonical}")

    return run_dir


def _take_snapshot(
    run_dir: Path,
    epoch: int,
    text_tower: TextTower,
    signal_tower: SignalTower,
    dataset: CardiacPairsDataset,
    val_idx: list[int],
    sim_indices: list[int],
    batch_size: int,
    device: torch.device,
) -> None:
    """Write per-epoch artifacts for animation: embeddings + similarity matrix."""
    epoch_dir = run_dir / f"epoch_{epoch:03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # Full validation embeddings (used by animate_umap.py)
    val_embeds = compute_all_embeddings(text_tower, dataset, val_idx, batch_size, device)
    torch.save(val_embeds, epoch_dir / "val_embeddings.pt")

    # Validation labels (mortality) — needed for UMAP coloring
    labels = {dataset.note_ids[i]: dataset.mortality[i] for i in val_idx}
    (epoch_dir / "val_labels.json").write_text(json.dumps(labels))

    # Fixed-N similarity matrix (used by animate_similarity.py)
    sim, note_ids = compute_similarity_matrix(
        text_tower, signal_tower, dataset, sim_indices, device
    )
    np.save(epoch_dir / "similarity_matrix.npy", sim)
    (epoch_dir / "similarity_note_ids.json").write_text(json.dumps(note_ids))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def apply_training_preset(ns: argparse.Namespace) -> None:
    """Tune defaults for small radiology cohort vs full BERT (roadmap: freeze + accum)."""
    if ns.preset != "low-data-bert":
        return
    ns.freeze_bert = True
    if ns.grad_accum_steps == GRAD_ACCUM_STEPS:
        ns.grad_accum_steps = 4
    if ns.lr_head == LR_HEAD:
        ns.lr_head = 2e-4
    if ns.lr_bert == LR_BERT:
        ns.lr_bert = 0.0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: Contrastive pre-training")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr-bert", type=float, default=LR_BERT)
    p.add_argument("--lr-head", type=float, default=LR_HEAD)
    p.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--freeze-bottom-layers", type=int, default=FREEZE_BOTTOM_LAYERS,
                   help="Freeze first N of 12 BERT layers (default 8 = train top 4)")
    p.add_argument("--freeze-bert", action="store_true",
                   help="Freeze ALL BERT layers — train only proj head + signal tower")
    p.add_argument("--proj-dropout", type=float, default=PROJ_DROPOUT)
    p.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS,
                   help="Gradient accumulation steps. Effective batch = batch_size × this. "
                        "InfoNCE is computed on the concatenated effective batch.")
    p.add_argument(
        "--val-loss-mode",
        type=str,
        choices=("macro", "full"),
        default=VAL_LOSS_MODE,
        help="macro: buffer val batches like train (aligned N); "
             "full: one InfoNCE over entire val set (max negatives; higher VRAM).",
    )
    p.add_argument(
        "--preset",
        type=str,
        choices=("none", "low-data-bert"),
        default="none",
        help="low-data-bert: freeze BERT, lr_bert=0, lr_head=2e-4, accum=4 if defaults unchanged.",
    )
    p.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Override input pairs CSV (default: data/processed/cardio_pairs.csv). "
             "Use e.g. data/processed/pairs_all-icus_note_level.csv for full MIMIC.",
    )
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint.pt in this run dir (continues in same dir).",
    )
    p.add_argument(
        "--max-time-hours",
        type=float,
        default=None,
        help="Stop cleanly after this many wall-clock hours. Resume command printed.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="On --resume: allow continuing despite locked-hyperparam mismatch.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.preset == "low-data-bert":
        apply_training_preset(args)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    csv_path = (
        Path(args.csv_path).resolve()
        if args.csv_path
        else PROJECT_ROOT / "data" / "processed" / "cardio_pairs.csv"
    )

    train(
        csv_path=csv_path,
        snapshots_root=PROJECT_ROOT / "data" / "snapshots",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_bert=args.lr_bert,
        lr_head=args.lr_head,
        embed_dim=args.embed_dim,
        temperature=args.temperature,
        seed=args.seed,
        val_frac=args.val_frac,
        use_amp=not args.no_amp,
        early_stop_patience=args.early_stop_patience,
        freeze_bottom_layers=args.freeze_bottom_layers,
        freeze_bert=args.freeze_bert,
        proj_dropout=args.proj_dropout,
        grad_accum_steps=args.grad_accum_steps,
        val_loss_mode=args.val_loss_mode,
        training_preset=args.preset,
        cli_argv=sys.argv,
        resume_from=args.resume,
        max_time_hours=args.max_time_hours,
        force=args.force,
    )
