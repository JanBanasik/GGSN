"""
Phase 1: Multimodal Contrastive Pre-training (Self-Supervised).

Pipeline:
  pairs_*.csv  →  ContrastivePairsDataset  →  Two-Tower (BERT + CNN)
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.data_prep.contrastive_dataset import ContrastivePairsDataset, split_indices_by_subject
from src.models.towers import BERT_MODEL, EMBED_DIM, NUM_SIGNAL_TYPES, SignalTower, TextTower
from src.training.hard_negatives import (
    HardNegativeBatchSampler,
    compute_hard_negative_table,
    diagnose_hard_neg_table,
)
from src.training.loss import info_nce_loss
from src.training.snapshot import (
    compute_all_embeddings,
    compute_text_signal_embeddings_arr,
    evaluate_val_loss,
    take_snapshot,
)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def _grad_scaler_cuda(enabled: bool):
    amp_gs = getattr(torch.amp, "GradScaler", None)
    if amp_gs is not None:
        return amp_gs("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Checkpoint helpers (resume training across sessions)
# ---------------------------------------------------------------------------
CHECKPOINT_NAME = "checkpoint.pt"
CKPT_LOCKED_KEYS = (
    "batch_size",
    "grad_accum_steps",
    "embed_dim",
    "freeze_bert",
    "freeze_bottom_layers",
    "proj_dropout",
    "lr_bert",
    "lr_head",
    "temperature",
    "seed",
    "val_frac",
    "max_text_len",
    "seq_len",
)


def save_checkpoint(path: Path, **state) -> None:
    """Atomic write: write to .tmp then rename, so Ctrl-C can't corrupt."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
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
# Hyper-parameter defaults
# ---------------------------------------------------------------------------
EPOCHS = 25
BATCH_SIZE = 16
MAX_TEXT_LEN = 256
SEQ_LEN = 64
LR_BERT = 5e-6
LR_HEAD = 2e-4
TEMPERATURE = 0.07
GRAD_CLIP = 1.0
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 5
VAL_FRAC = 0.10
SEED = 42
SIM_MATRIX_N = 64
FREEZE_BOTTOM_LAYERS = 8
PROJ_DROPOUT = 0.2
GRAD_ACCUM_STEPS = 4
FREEZE_BERT = False
HARD_NEG_REFRESH_EVERY = 3
HARD_NEG_POOL_SIZE = 256
HARD_NEG_ANCHORS_PER_BATCH = 8
VAL_LOSS_MODE = "macro"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_sha() -> str:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


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
    hard_negatives: bool = False,
    hard_neg_refresh_every: int = HARD_NEG_REFRESH_EVERY,
    hard_neg_pool_size: int = HARD_NEG_POOL_SIZE,
    hard_neg_anchors_per_batch: int = HARD_NEG_ANCHORS_PER_BATCH,
    init_from_weights: str | None = None,
    max_text_len: int = MAX_TEXT_LEN,
    seq_len: int = SEQ_LEN,
) -> Path:
    """
    Main training entry. Returns path to the run directory.

    resume_from: path to an existing run dir; loads checkpoint.pt and continues.
    max_time_hours: stop cleanly after N wall-clock hours; checkpoint written first.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
    dataset = ContrastivePairsDataset(
        csv_path, tokenizer, max_text_len=max_text_len, seq_len=seq_len
    )
    train_idx, val_idx = split_indices_by_subject(dataset, val_frac, seed)
    print(f"  Train notes: {len(train_idx)} | Val notes: {len(val_idx)}")

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    effective_batch = batch_size * grad_accum_steps
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
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
    print(f"  BERT trainable: {n_bert_train / 1e6:.1f}M / {n_bert_total / 1e6:.1f}M params")

    if init_from_weights and not is_resume:
        init_dir = Path(init_from_weights).resolve()
        text_path, sig_path = init_dir / "best_text_tower.pt", init_dir / "best_signal_tower.pt"
        if not text_path.exists() or not sig_path.exists():
            raise FileNotFoundError(f"--init-from-weights expects best_*.pt in {init_dir}")
        text_tower.load_state_dict(torch.load(text_path, map_location=device, weights_only=False))
        signal_tower.load_state_dict(torch.load(sig_path, map_location=device, weights_only=False))
        print("  → towers initialized from BEST checkpoint; optimizer/scheduler are FRESH.")

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

    # --- Config ---
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
        "max_text_len": int(max_text_len),
        "seq_len": int(seq_len),
        "num_signal_types": NUM_SIGNAL_TYPES,
        "amp": amp_enabled,
        "freeze_bottom_layers": freeze_bottom_layers,
        "freeze_bert": freeze_bert,
        "proj_dropout": proj_dropout,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch": int(effective_batch),
        "bert_trainable_params": int(n_bert_train),
        "bert_total_params": int(n_bert_total),
        "infonce_baseline_ln_train": float(np.log(effective_batch)),
        "val_loss_mode": val_loss_mode,
        "val_macro_accum_steps": grad_accum_steps,
        "infonce_baseline_ln_val_macro_aligned": float(np.log(effective_batch)),
        "training_preset": training_preset,
        "cli_argv": list(cli_argv) if cli_argv else [],
        "pairs_csv_sha256": _sha256_file(csv_path) if csv_path.is_file() else None,
        "hard_negatives": hard_negatives,
        "hard_neg_refresh_every": hard_neg_refresh_every,
        "hard_neg_pool_size": hard_neg_pool_size,
        "hard_neg_anchors_per_batch": hard_neg_anchors_per_batch,
        "init_from_weights": str(Path(init_from_weights).resolve()) if init_from_weights else None,
    }

    log_path = run_dir / "log.csv"
    start_epoch = 1
    best_val = float("inf")
    bad_epochs = 0

    if is_resume:
        ckpt = load_checkpoint(run_dir / CHECKPOINT_NAME, device)
        validate_resume_config(ckpt["config"], config, force=force)
        config = {
            **ckpt["config"],
            "epochs": epochs,
            "training_preset": training_preset,
            "cli_argv": list(cli_argv) if cli_argv else ckpt["config"].get("cli_argv", []),
        }
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
        print(f"  Resumed from epoch {ckpt['epoch']}; continuing {start_epoch}..{epochs}")
        if cli_argv:
            with (run_dir / "RUN_COMMAND.txt").open("a") as f:
                f.write(f"\n# Resume: {' '.join(shlex.quote(a) for a in cli_argv)}\n")
    else:
        (run_dir / "config.json").write_text(json.dumps(config, indent=2))
        if cli_argv:
            (run_dir / "RUN_COMMAND.txt").write_text(
                f"{' '.join(shlex.quote(a) for a in cli_argv)}\n"
            )
        log_path.write_text("epoch,train_loss,val_loss,val_infonce_n_mean,lr,elapsed_s\n")
        print("\nSnapshot epoch 0 (pre-training)…")
        take_snapshot(
            run_dir, 0, text_tower, signal_tower, dataset, val_idx, sim_indices, batch_size, device
        )

    # --- Hard negative state ---
    hard_neg_table: torch.Tensor | None = None
    hard_neg_seed = seed * 1000 + 7

    def rebuild_hard_neg_loader() -> DataLoader:
        gen = torch.Generator().manual_seed(hard_neg_seed + epoch)
        sampler = HardNegativeBatchSampler(
            hard_neg_table,
            anchor_pool=train_idx,
            batch_size=effective_batch,
            anchors_per_batch=hard_neg_anchors_per_batch,
            subject_ids=dataset.subject_ids,
            generator=gen,
        )
        return DataLoader(
            dataset, batch_sampler=sampler, num_workers=2, pin_memory=(device.type == "cuda")
        )

    # --- Training loop ---
    print(
        f"\nTraining {start_epoch}..{epochs} | train={len(train_idx)} val={len(val_idx)} "
        f"| batch={batch_size} × accum={grad_accum_steps} = effective {effective_batch} "
        f"| InfoNCE baseline = ln({effective_batch}) = {np.log(effective_batch):.3f}\n"
        f"| val_loss_mode={val_loss_mode}"
        + (
            f"\n| hard_negatives=ON, refresh_every={hard_neg_refresh_every}"
            if hard_negatives
            else ""
        )
        + "\n"
    )
    session_start = time.time()
    max_seconds = float(max_time_hours) * 3600.0 if max_time_hours else float("inf")
    stopped_by_max_time = False

    for epoch in range(start_epoch, epochs + 1):
        text_tower.train()
        signal_tower.train()

        # Hard negatives: refresh table + swap loader
        epoch_grad_accum = grad_accum_steps
        if hard_negatives and epoch >= 2:
            need_refresh = hard_neg_table is None or ((epoch - 2) % hard_neg_refresh_every == 0)
            if need_refresh:
                print(
                    f"  [hard-neg] epoch {epoch}: rebuilding table over {len(train_idx)} train samples…"
                )
                t_hn = time.time()
                text_emb, sig_emb, train_subj_ids = compute_text_signal_embeddings_arr(
                    text_tower,
                    signal_tower,
                    dataset,
                    train_idx,
                    batch_size,
                    device,
                )
                hard_neg_table_in_train = compute_hard_negative_table(
                    text_emb,
                    sig_emb,
                    train_subj_ids,
                    k_per_anchor=hard_neg_pool_size,
                )
                stats = diagnose_hard_neg_table(
                    hard_neg_table_in_train, train_subj_ids, sample_n=128
                )
                if stats["same_subject_leakage"] != 0 or stats["self_refs"] != 0:
                    raise RuntimeError(f"Hard negative table leakage check FAILED: {stats}")
                row_to_dataset = torch.as_tensor(train_idx, dtype=torch.long)
                neg_dataset_idx = row_to_dataset[hard_neg_table_in_train]
                hard_neg_table = torch.zeros((len(dataset), hard_neg_pool_size), dtype=torch.long)
                hard_neg_table[row_to_dataset] = neg_dataset_idx
                print(f"  [hard-neg] built in {time.time() - t_hn:.1f}s | leakage=0 ✓")

            current_loader = rebuild_hard_neg_loader()
            epoch_grad_accum = 1
        else:
            current_loader = train_loader

        epoch_loss = 0.0
        n_optim_steps = 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)
        z_text_buf: list[torch.Tensor] = []
        z_sig_buf: list[torch.Tensor] = []

        for step, batch in enumerate(current_loader, 1):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            item_ids = batch["item_ids"].to(device, non_blocking=True)
            values = batch["values"].to(device, non_blocking=True)
            signal_mask = batch["signal_mask"].to(device, non_blocking=True)
            hours = batch["hours"].to(device, non_blocking=True)
            deltas = batch["deltas"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                z_text = text_tower(input_ids, attention_mask)
                z_signal = signal_tower(item_ids, values, signal_mask, hours, deltas)

            z_text_buf.append(z_text)
            z_sig_buf.append(z_signal)

            if (step % epoch_grad_accum != 0) and (step != len(current_loader)):
                continue

            with torch.amp.autocast("cuda", enabled=amp_enabled, dtype=torch.float16):
                loss = info_nce_loss(
                    torch.cat(z_text_buf, dim=0), torch.cat(z_sig_buf, dim=0), temperature
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(text_tower.parameters()) + list(signal_tower.parameters()), GRAD_CLIP
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item()
            n_optim_steps += 1
            z_text_buf.clear()
            z_sig_buf.clear()

            if n_optim_steps % 25 == 0 or step == len(current_loader):
                print(f"  Ep {epoch}/{epochs} | step {n_optim_steps:4d} | loss {loss.item():.4f}")

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
        val_n = float(val_stats.get("val_infonce_n_used", val_stats.get("val_infonce_n_mean", 0.0)))
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} | train {train_loss:.4f} | val {val_loss:.4f} "
            f"| val_N≈{val_n:.1f} | lr {current_lr:.2e} | {elapsed:.1f}s\n"
        )

        with log_path.open("a") as f:
            f.write(
                f"{epoch},{train_loss:.6f},{val_loss:.6f},{val_n:.2f},"
                f"{current_lr:.6e},{elapsed:.2f}\n"
            )

        take_snapshot(
            run_dir,
            epoch,
            text_tower,
            signal_tower,
            dataset,
            val_idx,
            sim_indices,
            batch_size,
            device,
        )

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            torch.save(text_tower.state_dict(), run_dir / "best_text_tower.pt")
            torch.save(signal_tower.state_dict(), run_dir / "best_signal_tower.pt")
        else:
            bad_epochs += 1

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
            print(f"Early stopping at epoch {epoch}.")
            break

        if time.time() - session_start >= max_seconds:
            stopped_by_max_time = True
            elapsed_h = (time.time() - session_start) / 3600.0
            print(f"\n[max-time] Stopped after {elapsed_h:.2f}h. Resume with:")
            print(
                f"  uv run python -m src.training.train_contrastive "
                f"--resume {run_dir} --epochs {epochs}"
                + (f" --max-time-hours {max_time_hours}" if max_time_hours else "")
            )
            break

    if stopped_by_max_time:
        print("[max-time] Resume to finish — final artifacts exported then.")
        return run_dir

    torch.save(text_tower.state_dict(), run_dir / "final_text_tower.pt")
    torch.save(signal_tower.state_dict(), run_dir / "final_signal_tower.pt")

    print("Exporting full node_embeddings.pt for all notes…")
    all_idx = list(range(len(dataset)))
    full_embeds = compute_all_embeddings(text_tower, dataset, all_idx, batch_size, device)
    torch.save(full_embeds, run_dir / "node_embeddings.pt")

    canonical = csv_path.parent.parent / "embeddings"
    canonical.mkdir(parents=True, exist_ok=True)
    torch.save(full_embeds, canonical / "node_embeddings.pt")
    torch.save(text_tower.state_dict(), canonical / "text_tower.pt")
    torch.save(signal_tower.state_dict(), canonical / "signal_tower.pt")
    print(f"Final artifacts → {run_dir} | canonical copies → {canonical}")

    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def apply_training_preset(ns: argparse.Namespace) -> None:
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
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--early-stop-patience", type=int, default=EARLY_STOP_PATIENCE)
    p.add_argument("--freeze-bottom-layers", type=int, default=FREEZE_BOTTOM_LAYERS)
    p.add_argument("--freeze-bert", action="store_true")
    p.add_argument("--proj-dropout", type=float, default=PROJ_DROPOUT)
    p.add_argument("--grad-accum-steps", type=int, default=GRAD_ACCUM_STEPS)
    p.add_argument("--val-loss-mode", choices=("macro", "full"), default=VAL_LOSS_MODE)
    p.add_argument("--preset", choices=("none", "low-data-bert"), default="none")
    p.add_argument("--csv-path", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max-time-hours", type=float, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--hard-negatives", action="store_true")
    p.add_argument("--hard-neg-refresh-every", type=int, default=HARD_NEG_REFRESH_EVERY)
    p.add_argument("--hard-neg-pool-size", type=int, default=HARD_NEG_POOL_SIZE)
    p.add_argument("--hard-neg-anchors-per-batch", type=int, default=HARD_NEG_ANCHORS_PER_BATCH)
    p.add_argument("--init-from-weights", type=str, default=None)
    p.add_argument("--max-text-len", type=int, default=MAX_TEXT_LEN)
    p.add_argument("--seq-len", type=int, default=SEQ_LEN)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.preset == "low-data-bert":
        apply_training_preset(args)
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    csv_path = (
        Path(args.csv_path).resolve()
        if args.csv_path
        else PROJECT_ROOT / "data" / "processed" / "pairs_all-icus_note_level.csv"
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
        hard_negatives=args.hard_negatives,
        hard_neg_refresh_every=args.hard_neg_refresh_every,
        hard_neg_pool_size=args.hard_neg_pool_size,
        hard_neg_anchors_per_batch=args.hard_neg_anchors_per_batch,
        init_from_weights=args.init_from_weights,
        max_text_len=args.max_text_len,
        seq_len=args.seq_len,
    )
