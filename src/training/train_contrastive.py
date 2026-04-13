"""
Phase 1: Multimodal Contrastive Pre-training (Self-Supervised).

Pipeline:
  cardio_pairs.csv  →  CardiacPairsDataset  →  Two-Tower (Text + Signal)
                    →  InfoNCE (NT-Xent) loss  →  trained encoders
                    →  export all note embeddings to data/embeddings/node_embeddings.pt

Usage:
    uv run python -m src.training.train_contrastive
    uv run python -m src.training.train_contrastive --epochs 10 --batch-size 16
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from src.models.towers import BERT_MODEL, EMBED_DIM, SignalTower, TextTower

# ---------------------------------------------------------------------------
# Hyper-parameters (overridable via CLI)
# ---------------------------------------------------------------------------
EPOCHS = 5
BATCH_SIZE = 8       # keep small – BERT + gradients are heavy on 8 GB VRAM
MAX_TEXT_LEN = 256   # BERT max is 512; 256 captures most radiology reports
SEQ_LEN = 32         # max vitals per note (pad/truncate)
LR_BERT = 2e-5       # lower LR for pre-trained BERT weights
LR_HEAD = 2e-4       # higher LR for projection heads & signal tower
TEMPERATURE = 0.07   # InfoNCE temperature (SimCLR default)
GRAD_CLIP = 1.0      # gradient clipping norm

# Vital normalisation ranges (physiologically plausible extremes)
HR_ITEM_ID = 220045
BP_ITEM_ID = 220181
ITEM_ID_TO_IDX = {HR_ITEM_ID: 0, BP_ITEM_ID: 1}
NORM_RANGE = {
    HR_ITEM_ID: (30.0, 250.0),
    BP_ITEM_ID: (20.0, 200.0),
}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CardiacPairsDataset(Dataset):
    """
    Reads cardio_pairs.csv (one row per note–vital pair) and collapses it into
    one sample per unique note_id.

    Each sample contains:
        input_ids       (MAX_TEXT_LEN,)  – Bio_ClinicalBERT token ids
        attention_mask  (MAX_TEXT_LEN,)
        signal          (SEQ_LEN, 2)     – [(item_type_float, normalised_value), ...]
        note_id         str
        mortality       int  (0 or 1)
    """

    def __init__(
        self,
        csv_path: Path,
        tokenizer: AutoTokenizer,
        max_text_len: int = MAX_TEXT_LEN,
        seq_len: int = SEQ_LEN,
    ) -> None:
        df = pl.read_csv(csv_path)

        # --- Aggregate vitals per note (sorted by vital_time for temporal order) ---
        vitals_by_note: dict[str, list[list[float]]] = {}
        for row in df.sort("vital_time").iter_rows(named=True):
            nid = row["note_id"]
            item_id = int(row["itemid"])
            raw_val = float(row["valuenum"])

            lo, hi = NORM_RANGE[item_id]
            norm_val = float(np.clip((raw_val - lo) / (hi - lo), 0.0, 1.0))
            item_float = float(ITEM_ID_TO_IDX[item_id])

            vitals_by_note.setdefault(nid, []).append([item_float, norm_val])

        # --- Unique notes with text and label ---
        notes_df = df.unique("note_id").select(["note_id", "text", "mortality"])

        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.seq_len = seq_len

        self.note_ids: list[str] = []
        self.encodings: list[dict] = []
        self.signals: list[torch.Tensor] = []
        self.mortality: list[int] = []

        print(f"  Tokenising {len(notes_df)} unique notes (this may take a moment)…")
        for row in notes_df.iter_rows(named=True):
            nid = row["note_id"]

            # Tokenise once at construction time to avoid repeated work in __getitem__
            enc = tokenizer(
                row["text"],
                max_length=max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            vitals = vitals_by_note.get(nid, [[0.0, 0.5]])
            # Pad or truncate to fixed seq_len
            if len(vitals) >= seq_len:
                vitals = vitals[:seq_len]
            else:
                vitals += [[0.0, 0.0]] * (seq_len - len(vitals))

            self.note_ids.append(nid)
            self.encodings.append({
                "input_ids": enc["input_ids"].squeeze(0),        # (L,)
                "attention_mask": enc["attention_mask"].squeeze(0),  # (L,)
            })
            self.signals.append(torch.tensor(vitals, dtype=torch.float32))  # (SEQ_LEN, 2)
            self.mortality.append(int(row["mortality"]))

    def __len__(self) -> int:
        return len(self.note_ids)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.encodings[idx]["input_ids"],
            "attention_mask": self.encodings[idx]["attention_mask"],
            "signal": self.signals[idx],
            "note_id": self.note_ids[idx],
            "mortality": self.mortality[idx],
        }


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def info_nce_loss(z_text: torch.Tensor, z_signal: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """
    Symmetric InfoNCE (NT-Xent) loss.

    Args:
        z_text:   (N, D) L2-normalised text embeddings
        z_signal: (N, D) L2-normalised signal embeddings
        temperature: softmax temperature τ

    Positive pairs: (z_text[i], z_signal[i])  — same index in the batch.
    Negatives:      all other j ≠ i within the batch (in-batch negatives).

    Returns scalar mean loss.
    """
    N = z_text.size(0)
    # Similarity matrix: entry [i, j] = cos_sim(text_i, signal_j) / τ
    logits = torch.matmul(z_text, z_signal.T) / temperature  # (N, N)
    labels = torch.arange(N, device=z_text.device)

    # Cross-entropy in both directions
    loss_t2s = F.cross_entropy(logits, labels)
    loss_s2t = F.cross_entropy(logits.T, labels)

    return (loss_t2s + loss_s2t) / 2.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(
    csv_path: Path,
    embeddings_dir: Path,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr_bert: float = LR_BERT,
    lr_head: float = LR_HEAD,
    embed_dim: int = EMBED_DIM,
    temperature: float = TEMPERATURE,
) -> tuple[TextTower, SignalTower]:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # --- Data ---
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
    dataset = CardiacPairsDataset(csv_path, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # --- Models ---
    text_tower = TextTower(embed_dim=embed_dim).to(device)
    signal_tower = SignalTower(embed_dim=embed_dim).to(device)

    # Two learning-rate groups: slow for pre-trained BERT, fast for heads
    optimizer = torch.optim.AdamW(
        [
            {"params": text_tower.bert.parameters(), "lr": lr_bert},
            {"params": text_tower.proj.parameters(), "lr": lr_head},
            {"params": signal_tower.parameters(), "lr": lr_head},
        ],
        weight_decay=1e-4,
    )

    # Cosine annealing over the full training run
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(loader)
    )

    # --- Training ---
    print(f"\nTraining for {epochs} epochs | {len(dataset)} samples | batch={batch_size}\n")
    for epoch in range(1, epochs + 1):
        text_tower.train()
        signal_tower.train()

        epoch_loss = 0.0
        for step, batch in enumerate(loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            signal = batch["signal"].to(device)

            z_text = text_tower(input_ids, attention_mask)    # (B, D)
            z_signal = signal_tower(signal)                   # (B, D)

            loss = info_nce_loss(z_text, z_signal, temperature)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(text_tower.parameters()) + list(signal_tower.parameters()),
                GRAD_CLIP,
            )
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % 10 == 0 or step == len(loader):
                print(f"  Epoch {epoch}/{epochs} | step {step:3d}/{len(loader)} | loss {loss.item():.4f}")

        print(f"Epoch {epoch}/{epochs} done | avg loss: {epoch_loss / len(loader):.4f}\n")

    # --- Save model weights ---
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    torch.save(text_tower.state_dict(), embeddings_dir / "text_tower.pt")
    torch.save(signal_tower.state_dict(), embeddings_dir / "signal_tower.pt")
    print("Saved model weights → data/embeddings/{text,signal}_tower.pt")

    # --- Export node embeddings for ALL unique notes → GNN Phase 2 ---
    _export_node_embeddings(text_tower, dataset, embeddings_dir, batch_size, device)

    return text_tower, signal_tower


def _export_node_embeddings(
    text_tower: TextTower,
    dataset: CardiacPairsDataset,
    embeddings_dir: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    """
    Runs inference on all unique notes and saves a dict:
        { note_id: torch.Tensor(embed_dim,) }
    to data/embeddings/node_embeddings.pt
    """
    print("Generating node embeddings for all unique notes…")
    text_tower.eval()

    note_embeddings: dict[str, torch.Tensor] = {}

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            items = [dataset[i] for i in range(start, end)]

            input_ids = torch.stack([it["input_ids"] for it in items]).to(device)
            attention_mask = torch.stack([it["attention_mask"] for it in items]).to(device)
            note_ids_batch = [it["note_id"] for it in items]

            z = text_tower(input_ids, attention_mask).cpu()  # (B, D)

            for nid, emb in zip(note_ids_batch, z):
                note_embeddings[nid] = emb

    out_path = embeddings_dir / "node_embeddings.pt"
    torch.save(note_embeddings, out_path)
    print(f"Saved {len(note_embeddings)} embeddings → {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1: Contrastive pre-training")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr-bert", type=float, default=LR_BERT)
    p.add_argument("--lr-head", type=float, default=LR_HEAD)
    p.add_argument("--embed-dim", type=int, default=EMBED_DIM)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    train(
        csv_path=PROJECT_ROOT / "data" / "processed" / "cardio_pairs.csv",
        embeddings_dir=PROJECT_ROOT / "data" / "embeddings",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_bert=args.lr_bert,
        lr_head=args.lr_head,
        embed_dim=args.embed_dim,
        temperature=args.temperature,
    )
