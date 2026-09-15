"""
Trainer: Group-aware Training Pipeline for Ancient Greek Mythology Classification.
Trains on high-resolution BAPD photographic records using CLIP vision embeddings on CUDA.
"""

import os
import re
import json
import glob
import random
import datetime
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image

# Ensure UTF-8 output
import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


TARGET_FIGURES = ["Athena", "Dionysos", "Herakles", "Apollon"]


class MythologyMLP(nn.Module):
    """
    Multi-Layer Perceptron classifier head on top of normalized CLIP visual embeddings.
    """
    def __init__(self, input_dim: int = 512, hidden_dim: int = 256, num_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CorpusTrainer:
    """
    Manages data loading, group-aware splitting, feature extraction, training and evaluation.
    """

    def __init__(
        self,
        corpus_dir: str = "data/bapd_corpus",
        model_id: str = "openai/clip-vit-base-patch32",
        output_model_path: str = "models/mythology_classifier.pt"
    ):
        self.corpus_dir = corpus_dir
        self.model_id = model_id
        self.output_model_path = output_model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.classes = TARGET_FIGURES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def load_samples(self) -> List[Dict[str, Any]]:
        """
        Loads all image and metadata pairs and parses their target deity and vase ID.
        """
        samples = []
        txt_files = sorted(glob.glob(os.path.join(self.corpus_dir, "bapd_*.txt")))

        for txt_path in txt_files:
            stem = os.path.splitext(os.path.basename(txt_path))[0]
            # Infer matching image
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(self.corpus_dir, f"{stem}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if not img_path:
                continue

            # Parse metadata
            meta = {}
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip()] = v.strip()

            label = meta.get("Hauptmotiv", "")
            if not label:
                # Infer from SKOS-URI or Name
                skos = meta.get("SKOS-URI", "")
                if "athena" in skos.lower(): label = "Athena"
                elif "dionys" in skos.lower(): label = "Dionysos"
                elif "heracl" in skos.lower(): label = "Herakles"
                elif "apollo" in skos.lower(): label = "Apollon"
                else:
                    name = meta.get("Name/Bezeichnung", "")
                    for target in TARGET_FIGURES:
                        if target.lower() in name.lower():
                            label = target
                            break

            if label not in self.class_to_idx:
                continue

            # Group ID: base vase number (e.g. bapd_596 from bapd_596_2)
            parts = stem.split("_")
            base_vase_id = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else stem

            samples.append({
                "stem": stem,
                "img_path": img_path,
                "label": label,
                "label_idx": self.class_to_idx[label],
                "vase_id": base_vase_id,
                "shape": meta.get("Gefäßform", ""),
                "ware": meta.get("Ware", meta.get("Ware/Stil", ""))
            })

        return samples

    def grouped_train_val_split(
        self,
        samples: List[Dict[str, Any]],
        val_fraction: float = 0.2,
        seed: int = 42
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Splits dataset by physical vase_id to completely avoid multi-view data leakage.
        Stratified by dominant figure on the vase.
        """
        random.seed(seed)

        # Group samples by vase_id
        vases = defaultdict(list)
        vase_label = {}
        for s in samples:
            vid = s["vase_id"]
            vases[vid].append(s)
            vase_label[vid] = s["label"]

        # Stratified bucket by label
        by_label = defaultdict(list)
        for vid, lbl in vase_label.items():
            by_label[lbl].append(vid)

        train_vases = set()
        val_vases = set()

        for lbl, vids in by_label.items():
            random.shuffle(vids)
            n_val = max(1, int(len(vids) * val_fraction))
            val_vids = vids[:n_val]
            train_vids = vids[n_val:]
            val_vases.update(val_vids)
            train_vases.update(train_vids)

        train_samples = [s for s in samples if s["vase_id"] in train_vases]
        val_samples = [s for s in samples if s["vase_id"] in val_vases]

        return train_samples, val_samples

    def extract_features(self, samples: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts and caches normalized visual embeddings using CLIP ViT on CUDA.
        """
        from transformers import CLIPProcessor, CLIPModel

        print(f"Loading CLIP backbone ({self.model_id}) on {self.device}...")
        processor = CLIPProcessor.from_pretrained(self.model_id)
        model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        model.eval()

        features_list = []
        labels_list = []

        batch_size = 32
        total = len(samples)

        print(f"Extracting visual features for {total} images (batch size: {batch_size})...")
        with torch.no_grad():
            for i in range(0, total, batch_size):
                batch_samples = samples[i : i + batch_size]
                images = []
                for s in batch_samples:
                    try:
                        im = Image.open(s["img_path"]).convert("RGB")
                        images.append(im)
                    except Exception as e:
                        print(f"Warning: could not open {s['img_path']}: {e}")
                        im = Image.new("RGB", (224, 224), color=(0, 0, 0))
                        images.append(im)

                inputs = processor(images=images, return_tensors="pt").to(self.device)
                img_feats = model.get_image_features(**inputs)
                img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

                features_list.append(img_feats.cpu())
                labels_list.extend([s["label_idx"] for s in batch_samples])

        all_feats = torch.cat(features_list, dim=0)
        all_labels = torch.tensor(labels_list, dtype=torch.long)
        return all_feats, all_labels

    def train(
        self,
        epochs: int = 40,
        batch_size: int = 32,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_fraction: float = 0.2
    ) -> Dict[str, Any]:
        """
        Full training workflow: Grouped Split -> Feature Extraction -> MLP Training -> Evaluation.
        """
        print(f"\n{'='*65}")
        print(f"  Ancient Ceramics Mythology AI Classifier Training")
        print(f"  Target Deities: {', '.join(self.classes)}")
        print(f"  Device: {self.device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
        print(f"{'='*65}\n")

        # 1. Load Samples
        samples = self.load_samples()
        if not samples:
            raise ValueError(f"No valid samples found in {self.corpus_dir}")

        total_vases = len(set(s["vase_id"] for s in samples))
        print(f"Loaded {len(samples)} images across {total_vases} unique ancient vases.")

        class_counts = Counter(s["label"] for s in samples)
        for c in self.classes:
            print(f"  * {c:10}: {class_counts[c]} views")

        # 2. Grouped Split
        train_samples, val_samples = self.grouped_train_val_split(samples, val_fraction=val_fraction)
        train_vases_cnt = len(set(s["vase_id"] for s in train_samples))
        val_vases_cnt = len(set(s["vase_id"] for s in val_samples))
        print(f"\nDataset Split (Grouped by Vase ID):")
        print(f"  Training   : {len(train_samples)} images ({train_vases_cnt} unique vases)")
        print(f"  Validation : {len(val_samples)} images ({val_vases_cnt} unique vases)")

        # 3. Feature Extraction
        train_x, train_y = self.extract_features(train_samples)
        val_x, val_y = self.extract_features(val_samples)

        train_loader = DataLoader(
            TensorDataset(train_x, train_y),
            batch_size=batch_size,
            shuffle=True
        )

        # 4. Model Setup
        input_dim = train_x.shape[1]
        model = MythologyMLP(input_dim=input_dim, hidden_dim=256, num_classes=len(self.classes), dropout=0.3).to(self.device)

        # Class weights for balanced loss
        class_weights = torch.ones(len(self.classes))
        for c, idx in self.class_to_idx.items():
            cnt = class_counts[c]
            class_weights[idx] = len(samples) / (len(self.classes) * cnt) if cnt > 0 else 1.0
        class_weights = class_weights.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        # 5. Training Loop
        best_val_acc = 0.0
        best_state = None
        history = []

        print(f"\nStarting optimization for {epochs} epochs...")
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            total_train = 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * bx.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == by).sum().item()
                total_train += bx.size(0)

            scheduler.step()
            train_loss = total_loss / total_train
            train_acc = correct / total_train

            # Validation
            model.eval()
            with torch.no_grad():
                v_logits = model(val_x.to(self.device))
                v_loss = criterion(v_logits, val_y.to(self.device)).item()
                v_preds = v_logits.argmax(dim=-1)
                val_acc = (v_preds == val_y.to(self.device)).float().mean().item()

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": v_loss,
                "val_acc": val_acc
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % 5 == 0 or epoch == epochs:
                print(f"Epoch [{epoch:2d}/{epochs:2d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% | Val Loss: {v_loss:.4f} | Val Acc: {val_acc*100:.1f}% {'★ (Best)' if val_acc == best_val_acc else ''}")

        # 6. Evaluation with Best Model
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(self.device)

        model.eval()
        with torch.no_grad():
            v_logits = model(val_x.to(self.device))
            v_preds = v_logits.argmax(dim=-1).cpu().numpy()
            y_true = val_y.numpy()

        # Compute Metrics & Confusion Matrix
        cm = [[0 for _ in range(len(self.classes))] for _ in range(len(self.classes))]
        for t, p in zip(y_true, v_preds):
            cm[t][p] += 1

        metrics = {}
        for i, c in enumerate(self.classes):
            tp = cm[i][i]
            fp = sum(cm[row][i] for row in range(len(self.classes)) if row != i)
            fn = sum(cm[i][col] for col in range(len(self.classes)) if col != i)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            support = sum(cm[i])
            metrics[c] = {
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "support": support
            }

        overall_acc = sum(cm[i][i] for i in range(len(self.classes))) / len(y_true)

        # 7. Print Evaluation Report
        print(f"\n{'='*65}")
        print(f"  Final Evaluation Report (Held-out Test Vases)")
        print(f"  Overall Accuracy: {overall_acc*100:.2f}%")
        print(f"{'='*65}")
        print(f"{'Deity':14} | {'Precision':10} | {'Recall':10} | {'F1-Score':10} | {'Vases':8}")
        print(f"{'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
        for c in self.classes:
            m = metrics[c]
            print(f"{c:14} | {m['precision']*100:9.1f}% | {m['recall']*100:9.1f}% | {m['f1_score']*100:9.1f}% | {m['support']:8d}")
        print(f"{'-'*65}")

        print(f"\nConfusion Matrix (Rows: Actual Deity, Columns: Predicted Deity):")
        header = f"{'':14} | " + " | ".join(f"{c[:8]:8}" for c in self.classes)
        print(header)
        print("-" * len(header))
        for i, c in enumerate(self.classes):
            row_str = f"{c:14} | " + " | ".join(f"{cm[i][j]:8d}" for j in range(len(self.classes)))
            print(row_str)
        print(f"{'='*65}\n")

        # 8. Save Checkpoint & Metadata
        os.makedirs(os.path.dirname(self.output_model_path), exist_ok=True)
        checkpoint = {
            "state_dict": best_state,
            "classes": self.classes,
            "class_to_idx": self.class_to_idx,
            "input_dim": input_dim,
            "hidden_dim": 256,
            "backbone": self.model_id,
            "val_accuracy": float(overall_acc),
            "date": datetime.datetime.now().isoformat()
        }
        torch.save(checkpoint, self.output_model_path)

        meta_path = self.output_model_path.replace(".pt", "_meta.json")
        meta_data = {
            "classes": self.classes,
            "val_accuracy": round(overall_acc, 4),
            "per_class_metrics": metrics,
            "confusion_matrix": cm,
            "total_images": len(samples),
            "total_vases": total_vases,
            "train_images": len(train_samples),
            "val_images": len(val_samples),
            "date": datetime.datetime.now().isoformat(),
            "history": history
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)

        print(f"✨ Model successfully saved to: {self.output_model_path}")
        print(f"📊 Training metadata saved to: {meta_path}")

        return {
            "model_path": self.output_model_path,
            "val_accuracy": overall_acc,
            "metrics": metrics,
            "confusion_matrix": cm
        }
