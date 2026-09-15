"""
Detector: YOLOv8 Object Detection & Spatial Figure Localization on Ancient Greek Vases.
Detects individual deities (Athena, Dionysos, Herakles, Apollon) with bounding boxes,
relative spatial positioning ('links', 'Mitte', 'rechts'), and BCDH SKOS URI linking.
"""

import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Any, Union, Optional
from ultralytics import YOLO

DEITY_CLASSES = ["Athena", "Dionysos", "Herakles", "Apollon"]

DEITY_SKOS = {
    "Athena": "https://hector.bcdh.uni-bonn.de/athena",
    "Dionysos": "https://hector.bcdh.uni-bonn.de/dionysos",
    "Herakles": "https://hector.bcdh.uni-bonn.de/herakles",
    "Apollon": "https://hector.bcdh.uni-bonn.de/apollo"
}

# Distinct archaeologist-tailored palette for pottery iconography
DEITY_COLORS = {
    "Athena": (255, 179, 0),      # Athenian Amber Gold
    "Dionysos": (171, 71, 188),    # Dionysian Vine Purple
    "Herakles": (244, 81, 30),     # Heraklean Terracotta Crimson
    "Apollon": (0, 188, 212)       # Apollonian Laurel Cyan
}


class VaseFigureDetector:
    """
    Object detection engine for recognizing and localizing ancient deities
    directly on Greek pottery images.
    """

    def __init__(self, model_path: str = "models/yolo_deities_best.pt"):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            try:
                self._model = YOLO(self.model_path)
            except Exception as e:
                print(f"[VaseFigureDetector] Could not load YOLO weights from {self.model_path}: {e}")
                self._model = None
        else:
            print(f"[VaseFigureDetector] Model weights not found at {self.model_path}")

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @staticmethod
    def calculate_spatial_position(x1: int, x2: int, img_w: int) -> str:
        """
        Determines relative spatial placement along the horizontal frieze:
        - links: center in left 38%
        - Mitte: center in middle 24% (38% to 62%)
        - rechts: center in right 38%
        """
        if img_w <= 0:
            return "Mitte"
        xc_norm = ((x1 + x2) / 2.0) / img_w
        if xc_norm < 0.38:
            return "links"
        elif xc_norm > 0.62:
            return "rechts"
        else:
            return "Mitte"

    def detect(
        self,
        image: Union[str, Image.Image, np.ndarray],
        conf_threshold: float = 0.18,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Runs object detection on the input vase image.
        Returns a list of detected figures sorted horizontally from left to right.
        """
        if self._model is None:
            self._load_model()
            if self._model is None:
                return []

        # Convert input to PIL image to obtain dimensions
        if isinstance(image, str):
            pil_img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        w, h = pil_img.size

        # Run inference
        results = self._model(
            pil_img,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device,
            verbose=False
        )

        detections = []
        if not results or not results[0].boxes:
            return detections

        boxes = results[0].boxes
        for b in boxes:
            cls_id = int(b.cls[0])
            cls_name = self._model.names.get(cls_id, DEITY_CLASSES[cls_id % len(DEITY_CLASSES)])
            conf = float(b.conf[0])
            xyxy = [int(round(coord)) for coord in b.xyxy[0].tolist()]
            x1, y1, x2, y2 = xyxy

            # Clamp coordinates
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))

            spatial_pos = self.calculate_spatial_position(x1, x2, w)
            skos_uri = DEITY_SKOS.get(cls_name, "")

            detections.append({
                "label": cls_name,
                "class_id": cls_id,
                "confidence": conf,
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_norm": [
                    ((x1 + x2) / 2.0) / w,
                    ((y1 + y2) / 2.0) / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h
                ],
                "spatial_position": spatial_pos,
                "spatial_description": f"{cls_name} ({spatial_pos})",
                "skos_uri": skos_uri,
                "color": DEITY_COLORS.get(cls_name, (200, 200, 200))
            })

        # Sort detections from left to right by x1
        detections.sort(key=lambda d: d["bbox_xyxy"][0])
        return detections

    def draw_detections(
        self,
        image: Union[str, Image.Image],
        detections: List[Dict[str, Any]],
        line_width: int = 3,
        show_skos: bool = True
    ) -> Image.Image:
        """
        Renders sleek archaeological bounding box overlays and labels directly onto the vase image.
        """
        if isinstance(image, str):
            canvas = Image.open(image).convert("RGBA")
        else:
            canvas = image.convert("RGBA")

        w, h = canvas.size
        draw = ImageDraw.Draw(canvas)

        # Load font
        try:
            font_size = max(13, int(min(w, h) * 0.028))
            font = ImageFont.truetype("arial.ttf", font_size)
            small_font = ImageFont.truetype("arial.ttf", max(10, font_size - 3))
        except Exception:
            font = ImageFont.load_default()
            small_font = font

        for d in detections:
            x1, y1, x2, y2 = d["bbox_xyxy"]
            color = d.get("color", (255, 179, 0))
            label = d["label"]
            conf = d["confidence"]
            pos = d["spatial_position"]

            # 1. Draw outer glowing bounding box
            # Main border
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            # Corner accents for a modern archaeological HUD feel
            corner_len = min(20, (x2 - x1) // 4, (y2 - y1) // 4)
            corner_w = line_width + 2
            # Top-left
            draw.line([(x1, y1), (x1 + corner_len, y1)], fill=color, width=corner_w)
            draw.line([(x1, y1), (x1, y1 + corner_len)], fill=color, width=corner_w)
            # Top-right
            draw.line([(x2, y1), (x2 - corner_len, y1)], fill=color, width=corner_w)
            draw.line([(x2, y1), (x2, y1 + corner_len)], fill=color, width=corner_w)
            # Bottom-left
            draw.line([(x1, y2), (x1 + corner_len, y2)], fill=color, width=corner_w)
            draw.line([(x1, y2), (x1, y2 - corner_len)], fill=color, width=corner_w)
            # Bottom-right
            draw.line([(x2, y2), (x2 - corner_len, y2)], fill=color, width=corner_w)
            draw.line([(x2, y2), (x2, y2 - corner_len)], fill=color, width=corner_w)

            # 2. Label badge
            badge_text = f"{label} [{pos.upper()}] {conf * 100:.0f}%"
            try:
                bbox_text = font.getbbox(badge_text)
                tw = bbox_text[2] - bbox_text[0]
                th = bbox_text[3] - bbox_text[1]
            except Exception:
                tw, th = len(badge_text) * 8, 14

            badge_h = th + 8
            badge_w = tw + 12
            badge_y1 = max(0, y1 - badge_h - 2) if y1 >= badge_h + 4 else y1 + 4
            badge_y2 = badge_y1 + badge_h
            badge_x1 = max(0, x1)
            badge_x2 = min(w, badge_x1 + badge_w)

            # Dark translucent badge background
            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], fill=(20, 20, 25, 230))
            draw.rectangle([badge_x1, badge_y1, badge_x2, badge_y2], outline=color, width=1)
            draw.text((badge_x1 + 6, badge_y1 + 4), badge_text, fill=(255, 255, 255), font=font)

        return canvas.convert("RGB")


# Global singleton instance
figure_detector = VaseFigureDetector()
