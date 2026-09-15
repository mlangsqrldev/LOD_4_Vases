"""
YOLO Dataset Builder for Ancient Greek Pottery Iconography.
Extracts weakly-supervised bounding box annotations from the BAPD corpus,
validates them with CLIP feature classification, and produces YOLOv8 dataset format
with strict group-aware splitting to prevent data leakage between views of the same vessel.
"""

import os
import re
import glob
import shutil
import random
import cv2
import numpy as np
import yaml
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional

CLASS_NAMES = ["Athena", "Dionysos", "Herakles", "Apollon"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

DEITY_KEYWORDS = {
    "Athena": ["ATHENA", "ATHENE", "MINERVA"],
    "Dionysos": ["DIONYS", "BACCH", "MAENAD", "SATYR"],
    "Herakles": ["HERAKL", "HERCUL"],
    "Apollon": ["APOLLO"]
}

SKOS_URIS = {
    "Athena": "https://hector.bcdh.uni-bonn.de/athena",
    "Dionysos": "https://hector.bcdh.uni-bonn.de/dionysos",
    "Herakles": "https://hector.bcdh.uni-bonn.de/herakles",
    "Apollon": "https://hector.bcdh.uni-bonn.de/apollo"
}


class YOLODatasetBuilder:
    """
    Builds a complete YOLOv8 object detection dataset from the BAPD corpus.
    """

    def __init__(self, classifier=None, sam_segmenter=None):
        if classifier is None:
            from .classifier import classifier as clf
            self.classifier = clf
        else:
            self.classifier = classifier

        if sam_segmenter is None:
            from .sam_segmenter import sam_segmenter as sam
            self.sam_segmenter = sam
        else:
            self.sam_segmenter = sam_segmenter

    @staticmethod
    def extract_base_id(filepath: str) -> str:
        """
        Extracts base vase ID to group multiple views of the same vase.
        E.g. 'data/bapd_corpus/bapd_596_2.jpg' -> 'bapd_596'.
        """
        fname = os.path.basename(filepath)
        stem = os.path.splitext(fname)[0]
        # Match pattern like bapd_596 or bapd_596_1
        parts = stem.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
        return stem

    @staticmethod
    def extract_documented_deities(txt_path: str) -> List[str]:
        """
        Reads BAPD description and returns documented deities from target classes.
        """
        if not os.path.exists(txt_path):
            return []
        try:
            with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().upper()
        except Exception:
            return []

        found = []
        for deity, kws in DEITY_KEYWORDS.items():
            if any(kw in content for kw in kws):
                found.append(deity)
        return found

    def propose_and_label_boxes(
        self,
        image: Image.Image,
        documented_deities: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Proposes bounding boxes for figures on the vase and labels them
        conditioned on documented deities from BAPD metadata.
        Returns list of dicts: {"class_name": ..., "class_idx": ..., "bbox_xyxy": ..., "bbox_yolo": ..., "conf": ...}
        """
        w, h = image.size
        if w < 10 or h < 10:
            return []

        # Default if no deities documented
        if not documented_deities:
            # Predict whole image
            pred = self.classifier.predict_mythology(image)
            top = pred["top_mythologies"][0]
            documented_deities = [top["label"]]

        struct_mask = self.sam_segmenter.extract_structure_mask(image)
        # Exclude rim (top 8%) and base (bottom 8%) where figures rarely sit
        mid_mask = struct_mask[int(h * 0.08):int(h * 0.92), :]
        col_sum = np.sum(mid_mask > 0, axis=0)

        boxes = []
        is_multi = len(documented_deities) >= 2

        if not is_multi:
            # Single deity scene
            target_deity = documented_deities[0]
            target_idx = CLASS_TO_IDX[target_deity]
            
            # Find active horizontal range of the figure frieze
            active_cols = np.where(col_sum > (np.max(col_sum) * 0.15 + 1))[0]
            if len(active_cols) > 20:
                x1 = max(0, int(active_cols[0]))
                x2 = min(w, int(active_cols[-1]))
            else:
                x1 = int(w * 0.15)
                x2 = int(w * 0.85)

            # Find active vertical range
            row_sum = np.sum(struct_mask[:, x1:x2] > 0, axis=1)
            active_rows = np.where(row_sum > (np.max(row_sum) * 0.15 + 1))[0]
            if len(active_rows) > 20:
                y1 = max(0, int(active_rows[0]))
                y2 = min(h, int(active_rows[-1]))
            else:
                y1 = int(h * 0.12)
                y2 = int(h * 0.88)

            # Verify aspect ratio and minimum size
            bw = x2 - x1
            bh = y2 - y1
            if bw >= 20 and bh >= 20:
                boxes.append({
                    "class_name": target_deity,
                    "class_idx": target_idx,
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "conf": 1.0
                })
        else:
            # Multi-figure scene: 2 or more documented deities
            # Generate candidate figure regions along the frieze
            active_cols = np.where(col_sum > (np.max(col_sum) * 0.12 + 1))[0]
            if len(active_cols) > 40:
                start_x = int(active_cols[0])
                end_x = int(active_cols[-1])
            else:
                start_x = int(w * 0.08)
                end_x = int(w * 0.92)

            frieze_w = max(50, end_x - start_x)
            y1 = int(h * 0.10)
            y2 = int(h * 0.90)

            # Slicing options: Left, Center, Right segments
            num_targets = min(3, len(documented_deities))
            segment_candidates = []

            if num_targets == 2:
                # Find valley in central 40% of frieze
                mid_start = start_x + int(frieze_w * 0.30)
                mid_end = start_x + int(frieze_w * 0.70)
                if mid_end > mid_start:
                    sub_profile = col_sum[mid_start:mid_end]
                    split_x = mid_start + int(np.argmin(sub_profile))
                else:
                    split_x = start_x + frieze_w // 2

                # Left figure window and Right figure window (with slight overlap)
                pad = int(frieze_w * 0.05)
                box_left = [start_x, y1, min(w, split_x + pad), y2]
                box_right = [max(0, split_x - pad), y1, end_x, y2]
                segment_candidates = [box_left, box_right]
            else:
                # 3 segments: Left, Center, Right
                step = frieze_w // 3
                pad = int(step * 0.15)
                box1 = [start_x, y1, min(w, start_x + step + pad), y2]
                box2 = [max(0, start_x + step - pad), y1, min(w, start_x + 2 * step + pad), y2]
                box3 = [max(0, start_x + 2 * step - pad), y1, end_x, y2]
                segment_candidates = [box1, box2, box3]

            # Classify each candidate segment and assign to documented deities
            assigned_deities = set()
            for cand_box in segment_candidates:
                cx1, cy1, cx2, cy2 = cand_box
                if (cx2 - cx1) < 20 or (cy2 - cy1) < 20:
                    continue
                crop = image.crop((cx1, cy1, cx2, cy2))
                res = self.classifier.predict_mythology(crop)
                top_preds = res.get("top_mythologies", [])

                best_match = None
                best_conf = 0.0
                for pred_item in top_preds:
                    deity_name = pred_item["label"]
                    if deity_name in documented_deities and deity_name not in assigned_deities:
                        best_match = deity_name
                        best_conf = pred_item["confidence"]
                        break

                if best_match is None:
                    # Fallback to top documented deity with highest score
                    for pred_item in top_preds:
                        if pred_item["label"] in documented_deities:
                            best_match = pred_item["label"]
                            best_conf = pred_item["confidence"]
                            break

                if best_match is None and top_preds:
                    best_match = documented_deities[len(assigned_deities) % len(documented_deities)]
                    best_conf = 0.50

                if best_match:
                    assigned_deities.add(best_match)
                    boxes.append({
                        "class_name": best_match,
                        "class_idx": CLASS_TO_IDX[best_match],
                        "bbox_xyxy": cand_box,
                        "conf": float(best_conf)
                    })

        # Calculate normalized YOLO format: <class_idx> <x_center> <y_center> <width> <height>
        results = []
        for b in boxes:
            x1, y1, x2, y2 = b["bbox_xyxy"]
            x_c = ((x1 + x2) / 2.0) / w
            y_c = ((y1 + y2) / 2.0) / h
            bw_norm = (x2 - x1) / w
            bh_norm = (y2 - y1) / h

            # Clamp coordinates to [0.0, 1.0]
            x_c = max(0.001, min(0.999, x_c))
            y_c = max(0.001, min(0.999, y_c))
            bw_norm = max(0.005, min(1.0, bw_norm))
            bh_norm = max(0.005, min(1.0, bh_norm))

            b["bbox_yolo"] = [x_c, y_c, bw_norm, bh_norm]
            results.append(b)

        return results

    def build_dataset(
        self,
        corpus_dir: str = "data/bapd_corpus",
        output_dir: str = "data/yolo_deities",
        train_ratio: float = 0.80,
        random_seed: int = 42,
        max_images: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Builds the entire YOLOv8 dataset structure and YAML configuration.
        """
        img_files = sorted(glob.glob(os.path.join(corpus_dir, "*.jpg")))
        if max_images:
            img_files = img_files[:max_images]

        print(f"Discovered {len(img_files)} images in {corpus_dir}")

        # 1. Group images by base vessel ID
        vessel_groups: Dict[str, List[str]] = {}
        for img_path in img_files:
            base_id = self.extract_base_id(img_path)
            vessel_groups.setdefault(base_id, []).append(img_path)

        base_ids = sorted(list(vessel_groups.keys()))
        random.seed(random_seed)
        random.shuffle(base_ids)

        num_train_vessels = int(len(base_ids) * train_ratio)
        train_vessels = set(base_ids[:num_train_vessels])
        val_vessels = set(base_ids[num_train_vessels:])

        # 2. Setup directory tree
        splits = ["train", "val"]
        dirs = {}
        for s in splits:
            img_dir = os.path.join(output_dir, "images", s)
            lbl_dir = os.path.join(output_dir, "labels", s)
            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(lbl_dir, exist_ok=True)
            dirs[f"img_{s}"] = img_dir
            dirs[f"lbl_{s}"] = lbl_dir

        stats = {
            "total_images": len(img_files),
            "train_images": 0,
            "val_images": 0,
            "train_boxes": 0,
            "val_boxes": 0,
            "class_counts": {c: 0 for c in CLASS_NAMES},
            "multi_figure_images": 0
        }

        print(f"Generating YOLO annotations (Group Split: {len(train_vessels)} Train, {len(val_vessels)} Val vessels)...")

        for idx, img_path in enumerate(img_files):
            base_id = self.extract_base_id(img_path)
            split = "train" if base_id in train_vessels else "val"
            img_name = os.path.basename(img_path)
            stem = os.path.splitext(img_name)[0]

            # Locate corresponding text description
            txt_path = os.path.join(corpus_dir, f"{stem}.txt")
            if not os.path.exists(txt_path):
                txt_path = os.path.join(corpus_dir, f"{base_id}.txt")

            doc_deities = self.extract_documented_deities(txt_path)
            if len(doc_deities) >= 2:
                stats["multi_figure_images"] += 1

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Skipping unreadable image {img_path}: {e}")
                continue

            # Propose and label boxes
            boxes = self.propose_and_label_boxes(img, doc_deities)
            if not boxes:
                continue

            # Copy image to destination split
            dest_img = os.path.join(dirs[f"img_{split}"], img_name)
            if not os.path.exists(dest_img):
                shutil.copy2(img_path, dest_img)

            # Write YOLO label file
            dest_lbl = os.path.join(dirs[f"lbl_{split}"], f"{stem}.txt")
            lbl_lines = []
            for b in boxes:
                cls_idx = b["class_idx"]
                xc, yc, bw, bh = b["bbox_yolo"]
                lbl_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
                stats["class_counts"][CLASS_NAMES[cls_idx]] += 1
                if split == "train":
                    stats["train_boxes"] += 1
                else:
                    stats["val_boxes"] += 1

            with open(dest_lbl, "w", encoding="utf-8") as f:
                f.write("\n".join(lbl_lines) + "\n")

            if split == "train":
                stats["train_images"] += 1
            else:
                stats["val_images"] += 1

            if (idx + 1) % 100 == 0 or (idx + 1) == len(img_files):
                print(f"  Processed {idx + 1}/{len(img_files)} images...")

        # 3. Generate data.yaml
        abs_output_dir = os.path.abspath(output_dir).replace("\\", "/")
        yaml_content = {
            "path": abs_output_dir,
            "train": "images/train",
            "val": "images/val",
            "names": {i: name for i, name in enumerate(CLASS_NAMES)}
        }
        yaml_path = os.path.join(output_dir, "data.yaml")
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)

        print("\n=== YOLO Dataset Generation Summary ===")
        print(f"Total images processed : {stats['total_images']}")
        print(f"Train images / boxes   : {stats['train_images']} / {stats['train_boxes']}")
        print(f"Val images / boxes     : {stats['val_images']} / {stats['val_boxes']}")
        print(f"Multi-figure images    : {stats['multi_figure_images']}")
        print(f"Per-Class box counts   : {stats['class_counts']}")
        print(f"YAML config written to : {yaml_path}")
        print("========================================\n")

        return stats


yolo_dataset_builder = YOLODatasetBuilder()
