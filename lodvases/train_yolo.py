"""
YOLOv8 Training Pipeline for Ancient Greek Vase Iconography.
Fine-tunes YOLOv8 detection on GPU (NVIDIA RTX 4080) using the BAPD deities dataset.
"""

import os
import shutil
import json
import torch
from datetime import datetime
from typing import Dict, Any, Optional
from ultralytics import YOLO

DEITY_CLASSES = ["Athena", "Dionysos", "Herakles", "Apollon"]
DEITY_SKOS = {
    "Athena": "https://hector.bcdh.uni-bonn.de/athena",
    "Dionysos": "https://hector.bcdh.uni-bonn.de/dionysos",
    "Herakles": "https://hector.bcdh.uni-bonn.de/herakles",
    "Apollon": "https://hector.bcdh.uni-bonn.de/apollo"
}


def train_yolo(
    data_yaml: str = "data/yolo_deities/data.yaml",
    epochs: int = 30,
    batch_size: int = 16,
    imgsz: int = 640,
    base_model: str = "yolov8n.pt",
    output_weights: str = "models/yolo_deities_best.pt",
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Fine-tunes YOLOv8 on ancient vase iconography and saves the best model.
    """
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"YAML dataset config not found at {data_yaml}. Run build-yolo-dataset first.")

    # Determine compute device
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else "CPU"
    print(f"\n{'='*60}")
    print(f"  YOLOv8 Deities Object Detection Training")
    print(f"  Device        : {device} ({device_name})")
    print(f"  Base Model    : {base_model}")
    print(f"  Dataset Config: {data_yaml}")
    print(f"  Epochs        : {epochs}")
    print(f"  Batch Size    : {batch_size}")
    print(f"  Image Size    : {imgsz}")
    print(f"  Target Classes: {DEITY_CLASSES}")
    print(f"{'='*60}\n")

    # Initialize model
    model = YOLO(base_model)

    # Train model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        project="runs/detect",
        name="train_deities",
        exist_ok=True,
        plots=True,
        verbose=True,
        save=True,
        workers=2,
        optimizer="auto"
    )

    # Locate best weights from results save directory
    save_dir = str(getattr(results, "save_dir", "runs/detect/train_deities"))
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if not os.path.exists(best_pt):
        best_pt = os.path.join(save_dir, "weights", "last.pt")

    os.makedirs(os.path.dirname(output_weights), exist_ok=True)
    if os.path.exists(best_pt):
        shutil.copy2(best_pt, output_weights)
        print(f"\n[OK] Best model saved to: {output_weights}")
    else:
        print(f"\n[WARN] Could not find weights at {best_pt}")

    # Extract metrics
    metrics_summary = {}
    try:
        val_results = model.val(data=data_yaml, device=device)
        metrics_summary = {
            "mAP50": float(val_results.box.map50),
            "mAP50_95": float(val_results.box.map),
            "precision": float(val_results.box.mp),
            "recall": float(val_results.box.mr)
        }
        print(f"\nValidation Results:")
        print(f"  mAP@50     : {metrics_summary['mAP50']:.4f}")
        print(f"  mAP@50-95  : {metrics_summary['mAP50_95']:.4f}")
        print(f"  Precision  : {metrics_summary['precision']:.4f}")
        print(f"  Recall     : {metrics_summary['recall']:.4f}")
    except Exception as e:
        print(f"Metrics extraction note: {e}")

    # Save metadata JSON
    meta_path = output_weights.rsplit(".", 1)[0] + "_meta.json"
    metadata = {
        "model_type": "YOLOv8-ObjectDetection",
        "base_model": base_model,
        "classes": DEITY_CLASSES,
        "skos_uris": DEITY_SKOS,
        "epochs": epochs,
        "batch_size": batch_size,
        "imgsz": imgsz,
        "device": device_name,
        "trained_at": datetime.now().isoformat(),
        "metrics": metrics_summary,
        "weights_path": output_weights
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[OK] Model metadata saved to: {meta_path}\n")

    return {
        "weights_path": output_weights,
        "metadata_path": meta_path,
        "metrics": metrics_summary
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train YOLOv8 Deities Object Detection Model")
    parser.add_argument("--data", default="data/yolo_deities/data.yaml", help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--output", default="models/yolo_deities_best.pt", help="Output path for best weights")
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        output_weights=args.output
    )
