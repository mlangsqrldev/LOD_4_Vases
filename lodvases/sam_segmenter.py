"""
SAM (Segment Anything Model) Segmenter for Ancient Vase Motifs and Iconography.
Supports multiple high-precision backends:
1. MobileSAM (Vision Transformer ViT, ~40MB, ultra-fast & high precision)
2. Meta SAM 1 (sam_b.pt, ViT-Base, ~350MB, highest detail)
3. Archaeological Silhouette Engine (adaptive GrabCut & edge-preserving contouring)
4. FastSAM (Real-time CNN backend)
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Dict, Any, Tuple, Optional

class SAMSegmenter:
    """
    Handles promptable and automatic figure/motif segmentation with switchable backends.
    """
    def __init__(self, backend: str = "meta_sam"):
        self.backend = backend.lower()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._models = {}

    def set_backend(self, backend: str):
        """Switches the active segmentation engine."""
        self.backend = backend.lower()
        print(f"SAM Segmenter backend switched to: {self.backend}")

    def _get_model(self, model_name: str):
        if model_name not in self._models:
            from ultralytics import SAM, FastSAM
            print(f"Loading model '{model_name}' on {self.device}...")
            if "fastsam" in model_name.lower():
                self._models[model_name] = FastSAM(model_name)
            else:
                self._models[model_name] = SAM(model_name)
        return self._models[model_name]

    def detect_ceramic_technique(self, image: Image.Image) -> str:
        """
        Determines whether a vase painting is Schwarzfigurig or Rotfigurig
        by analyzing the color and luminance contrast of the frieze scene.
        """
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape
        r = img_np[:, :, 0]
        g = img_np[:, :, 1]
        b = img_np[:, :, 2]
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)

        # Sample central picture frieze
        f_hsv = hsv[int(h * 0.25):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
        f_r = r[int(h * 0.25):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
        f_g = g[int(h * 0.25):int(h * 0.75), int(w * 0.20):int(w * 0.80)]
        f_b = b[int(h * 0.25):int(h * 0.75), int(w * 0.20):int(w * 0.80)]

        # Clay ground: warm hue, high chroma, R > B + 15, R > 75, V > 70
        is_clay = (f_hsv[:, :, 1] > 45) & ((f_hsv[:, :, 0] <= 35) | (f_hsv[:, :, 0] >= 165)) & (f_r > f_b + 15) & (f_r > 70) & (f_hsv[:, :, 2] > 65)
        # Black glaze: dark, low V
        is_dark = (f_r < 65) & (f_g < 65) & (f_b < 65)

        clay_pct = float(np.mean(is_clay))
        dark_pct = float(np.mean(is_dark))

        # In black-figure, the carrier of the frieze is the terracotta clay ground (> 20% of frieze)
        if clay_pct > 0.20 and clay_pct >= dark_pct * 0.50:
            return "Attisch Schwarzfigurig"
        else:
            return "Attisch Rotfigurig"

    def invert_mask(self, mask: np.ndarray) -> np.ndarray:
        """Inverts a binary mask (foreground <-> background)."""
        return cv2.bitwise_not(mask)

    def extract_structure_mask(self, image: Image.Image, style: str = "auto", invert: bool = False) -> np.ndarray:
        """
        Separates the carrier background (clay ground or black glaze) from preserved
        figure & ornament structures according to the ceramic painting technique:
        - Rotfigurig: Figures = Terracotta Clay, Background = Black Glaze (Firnis) & scan background.
        - Schwarzfigurig / Korinthisch: Background = Clay ground (attic orange or corinthian pale cream),
          Figures = Dark glaze silhouettes + purple/white added paint + incised details.
        - Geometrisch: Figures/Ornaments = Dark paint bands/meanders, Background = Light clay ground.
        - Weißgrundig: Figures = Polychrome & glaze outlines, Background = White slip.
        """
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape
        total_area = w * h
        
        detected_style = style
        if style.lower() == "auto":
            try:
                # 1. First test physical color contrast in frieze
                detected_style = self.detect_ceramic_technique(image)
            except Exception:
                try:
                    from lodvases.classifier import classifier
                    pred = classifier.predict(image)
                    detected_style = pred.get("predicted_ware") or pred.get("top_ware") or "Attisch Rotfigurig"
                except Exception:
                    detected_style = "Attisch Rotfigurig"
                
        style_lower = detected_style.lower()
        is_red_figure = "rotfigurig" in style_lower
        is_corinthian = "korinthisch" in style_lower
        is_geometric = "geometrisch" in style_lower
        is_black_figure = "schwarzfigurig" in style_lower or is_corinthian
        
        r = img_np[:, :, 0]
        g = img_np[:, :, 1]
        b = img_np[:, :, 2]
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        struct_mask = np.zeros((h, w), dtype=np.uint8)
        
        is_grayscale = np.mean(hsv[:, :, 1]) < 12.0
        # Identify photo background outside vase (neutral scan background or paper backdrop)
        neutral_paper = (hsv[:, :, 1] < 40) & (gray > 80)
        border_mask = np.zeros((h, w), dtype=np.uint8)
        border_mask[0, :] = 1
        border_mask[-1, :] = 1
        border_mask[:, 0] = 1
        border_mask[:, -1] = 1
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(neutral_paper.astype(np.uint8))
        is_scan_bg = np.zeros((h, w), dtype=bool)
        for lbl in range(1, num_labels):
            if np.any(labels[border_mask == 1] == lbl):
                is_scan_bg[labels == lbl] = True

        # Also detect handle holes (neutral paper inside handle openings)
        handle_hole_zone = np.zeros((h, w), dtype=bool)
        handle_hole_zone[: int(h * 0.45), : int(w * 0.35)] = True
        handle_hole_zone[: int(h * 0.45), int(w * 0.65) :] = True
        is_scan_bg = is_scan_bg | (neutral_paper & handle_hole_zone)

        # Dilate slightly to seal borders against vignette gradients
        k_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        is_scan_bg = cv2.dilate(is_scan_bg.astype(np.uint8), k_bg).astype(bool)

        # Identify vase silhouette and vertical landmarks for 2D photographs
        vase_body = ~is_scan_bg
        vy, vx = np.where(vase_body)
        if len(vy) > 100:
            vy_min, vy_max = int(vy.min()), int(vy.max())
            vx_min, vx_max = int(vx.min()), int(vx.max())
            vh = vy_max - vy_min
            vw = vx_max - vx_min
            
            # Frieze zone (belly panel): excludes neck palmettes and lower meander/foot
            # On 3D panoramic rollouts (w > 1.6 * h), frieze spans full height
            if w > 1.6 * h:
                frieze_zone = np.ones((h, w), dtype=bool)
            else:
                fy1 = vy_min + int(vh * 0.24)
                fy2 = vy_min + int(vh * 0.77)
                fx1 = vx_min + int(vw * 0.12)
                fx2 = vx_min + int(vw * 0.88)
                frieze_zone = np.zeros((h, w), dtype=bool)
                frieze_zone[fy1:fy2, fx1:fx2] = True
        else:
            frieze_zone = np.ones((h, w), dtype=bool)

        if is_black_figure:
            # -------------------------------------------------------------
            # SCHWARZFIGURIG (Black-Figure):
            # Carrier Background = Terracotta Clay ground + Scan Paper
            # Figures = Black glaze silhouettes + added white paint (Athena, etc.) + purple
            # -------------------------------------------------------------
            is_clay_ground = (
                (hsv[:, :, 1] > 60)
                & ((hsv[:, :, 0] <= 35) | (hsv[:, :, 0] >= 165))
                & (r > 75)
                & (r > b + 20)
                & (gray < 225)
                & ~is_scan_bg
            )

            # For wide 3D panoramic rollouts:
            if w > 1.6 * h:
                num_p, p_lbls, p_stats, _ = cv2.connectedComponentsWithStats(is_clay_ground.astype(np.uint8))
                panel_mask = np.zeros((h, w), dtype=bool)
                for p in range(1, num_p):
                    if p_stats[p, cv2.CC_STAT_AREA] > (total_area * 0.015):
                        panel_mask[p_lbls == p] = True
                k_p = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
                panel_mask = cv2.dilate(panel_mask.astype(np.uint8), k_p).astype(bool)
                figures_raw = panel_mask & (~is_clay_ground) & ~is_scan_bg
            else:
                # For 2D vase photographs: figures are non-clay inside the central frieze panel
                figures_raw = frieze_zone & (~is_clay_ground) & vase_body

            # Bridge incisions and fine Ritzlinien
            k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            closed = cv2.morphologyEx(figures_raw.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_close)

            # Connected components: keep discrete figure entities (exclude speckles < 0.15%)
            num_comp, comp_labels, comp_stats, _ = cv2.connectedComponentsWithStats(closed)
            clean_struct = np.zeros((h, w), dtype=np.uint8)
            for i in range(1, num_comp):
                c_area = comp_stats[i, cv2.CC_STAT_AREA]
                if (total_area * 0.0015) <= c_area <= (total_area * 0.40):
                    clean_struct[comp_labels == i] = 255

            # Fill small internal incision holes inside figures (incisions, shield devices, white paint)
            inv_closed = cv2.bitwise_not(clean_struct)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv_closed)
            max_hole_area = total_area * 0.005
            for i in range(1, num_labels):
                hole_area = stats[i, cv2.CC_STAT_AREA]
                if hole_area < max_hole_area:
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    bw = stats[i, cv2.CC_STAT_WIDTH]
                    bh = stats[i, cv2.CC_STAT_HEIGHT]
                    if x > 0 and y > 0 and (x + bw) < w and (y + bh) < h:
                        clean_struct[labels == i] = 255

            struct_mask = clean_struct

        elif is_red_figure:
            # -------------------------------------------------------------
            # ROTFIGURIG (Red-Figure):
            # Carrier Background = Black Glaze (Firnis) & scan background
            # Figures = Terracotta Clay
            # -------------------------------------------------------------
            if is_grayscale:
                struct_mask = (gray >= 35).astype(np.uint8) * 255
            else:
                smoothed = cv2.bilateralFilter(img_np, d=7, sigmaColor=50, sigmaSpace=50)
                s_hsv = cv2.cvtColor(smoothed, cv2.COLOR_RGB2HSV)
                s_r = smoothed[:, :, 0]
                s_b = smoothed[:, :, 2]
                
                hue = s_hsv[:, :, 0]
                warm_hue = ((hue <= 38) | (hue >= 162))
                
                clay_candidate = warm_hue & (s_hsv[:, :, 1] >= 45) & (s_r > s_b + 15) & (s_hsv[:, :, 2] >= 38) & ~is_scan_bg
                if w <= 1.6 * h:
                    clay_candidate &= frieze_zone
                raw_mask = (clay_candidate.astype(np.uint8)) * 255
                
                k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                cleaned = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k_open)
                
                k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                closed = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)
                struct_mask = closed
            
            # Fast morphological closing to bridge internal relief lines & incisions
            k_seal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            struct_mask = cv2.morphologyEx(struct_mask, cv2.MORPH_CLOSE, k_seal)

            num_fg, fg_labels, fg_stats, _ = cv2.connectedComponentsWithStats(struct_mask)
            final_struct = np.zeros((h, w), dtype=np.uint8)
            for i in range(1, num_fg):
                area = fg_stats[i, cv2.CC_STAT_AREA]
                if area >= total_area * 0.0008:
                    final_struct[fg_labels == i] = 255
            struct_mask = final_struct
        else:
            # Geometrisch / Weißgrundig
            otsu_val, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            dark_thresh = min(140, int(otsu_val * 0.95))
            dark_mask = (gray < dark_thresh).astype(np.uint8) * 255
            struct_mask = dark_mask & ~is_scan_bg.astype(np.uint8) * 255

        if invert:
            struct_mask = cv2.bitwise_not(struct_mask)

        return struct_mask

    def segment_all(
        self,
        image: Image.Image,
        style: str = "auto",
        use_style_constraint: bool = True,
        min_area_pct: float = 0.08,
        max_area_pct: float = 40.0,
        invert: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Segments all figures and iconography objects in ancient Greek vase paintings.
        Extracts both composite parent entities (e.g. Reiter auf Delphin) and hierarchical
        sub-parts (Delphin, Reiter, Helm, Schild, Schildzeichen, Lanze).
        """
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape
        total_area = w * h
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        
        # 1. Compute style-informed structure mask
        if use_style_constraint:
            struct_mask = self.extract_structure_mask(image, style=style, invert=invert)
        else:
            struct_mask = np.full((h, w), 255, dtype=np.uint8)

        # 2. Discover prompt seeds from distance transform peaks and component grids
        dist = cv2.distanceTransform(struct_mask, cv2.DIST_L2, 5)
        max_d = dist.max() if dist.size > 0 else 0
        peaks = cv2.threshold(dist, 0.20 * max_d, 255, cv2.THRESH_BINARY)[1].astype(np.uint8) if max_d > 0 else struct_mask
        
        num_peaks, peak_labels, peak_stats, peak_centroids = cv2.connectedComponentsWithStats(peaks)
        seeds = []
        for i in range(1, num_peaks):
            p_area = peak_stats[i, cv2.CC_STAT_AREA]
            pcx, pcy = int(peak_centroids[i][0]), int(peak_centroids[i][1])
            if struct_mask[pcy, pcx] > 0:
                seeds.append((pcx, pcy, p_area))

        # Sample grid seeds across connected figure components
        num_c, _, c_stats, _ = cv2.connectedComponentsWithStats(struct_mask)
        for i in range(1, num_c):
            bx, by = c_stats[i, cv2.CC_STAT_LEFT], c_stats[i, cv2.CC_STAT_TOP]
            bw, bh = c_stats[i, cv2.CC_STAT_WIDTH], c_stats[i, cv2.CC_STAT_HEIGHT]
            n_x = max(3, int(bw / (w * 0.12)))
            n_y = max(2, int(bh / (h * 0.18)))
            for gx in np.linspace(bx + bw * 0.10, bx + bw * 0.90, n_x):
                for gy in np.linspace(by + bh * 0.20, by + bh * 0.80, n_y):
                    ix, iy = int(gx), int(gy)
                    if struct_mask[iy, ix] > 0:
                        seeds.append((ix, iy, 100))

        # For wide panoramic 3D rollouts, also sample grid seeds within detected picture panels
        if w > 1.6 * h:
            r = img_np[:, :, 0]
            b = img_np[:, :, 2]
            is_clay = (hsv[:, :, 1] > 40) & (r > b + 15) & (r > 70)
            num_panels, _, p_stats, _ = cv2.connectedComponentsWithStats(is_clay.astype(np.uint8))
            for p in range(1, num_panels):
                if p_stats[p, cv2.CC_STAT_AREA] > (w * h * 0.015):
                    px = p_stats[p, cv2.CC_STAT_LEFT]
                    py = p_stats[p, cv2.CC_STAT_TOP]
                    pw = p_stats[p, cv2.CC_STAT_WIDTH]
                    ph = p_stats[p, cv2.CC_STAT_HEIGHT]
                    for gx in np.linspace(px + pw * 0.10, px + pw * 0.90, 6):
                        for gy in np.linspace(py + ph * 0.22, py + ph * 0.82, 5):
                            seeds.append((int(gx), int(gy), 100))

        seeds.sort(key=lambda s: s[2], reverse=True)

        raw_components = []

        # 1. Primary Human Figure Discovery using YOLOv8 (high-confidence person columns)
        try:
            from ultralytics import YOLO
            yolo_model = YOLO("yolov8n.pt")
            y_res = yolo_model(image, conf=0.15, verbose=False)
            if y_res and len(y_res[0].boxes) > 0:
                p_boxes = []
                for b in y_res[0].boxes:
                    if int(b.cls[0]) == 0:  # person class
                        p_boxes.append([int(v) for v in b.xyxy[0].tolist()])
                
                # Merge boxes belonging to the same vertical person column
                merged_p = []
                p_boxes.sort(key=lambda b: b[0])
                for b in p_boxes:
                    bx1, by1, bx2, by2 = b
                    bw_b = bx2 - bx1
                    merged = False
                    for mb in merged_p:
                        mx1, my1, mx2, my2 = mb
                        mw = mx2 - mx1
                        x_overlap = max(0, min(bx2, mx2) - max(bx1, mx1))
                        if x_overlap / (min(bw_b, mw) + 1e-6) > 0.45:
                            mb[0] = min(mx1, bx1); mb[1] = min(my1, by1)
                            mb[2] = max(mx2, bx2); mb[3] = max(my2, by2)
                            merged = True
                            break
                    if not merged:
                        merged_p.append([bx1, by1, bx2, by2])
                        
                for pb in merged_p:
                    bx1, by1, bx2, by2 = pb
                    bw_b = bx2 - bx1
                    bh_b = by2 - by1
                    # Expand horizontally by 28% for outstretched arms, spears, sceptres, wings
                    ebx1 = max(0, int(bx1 - bw_b * 0.28))
                    ebx2 = min(w, int(bx2 + bw_b * 0.28))
                    eby1 = max(int(h * 0.14), int(by1 - max(bh_b * 0.35, h * 0.20)))
                    eby2 = min(int(h * 0.90), int(by2 + bh_b * 0.10))
                    p_motif = self.segment_at_point_or_box(image, box=(ebx1, eby1, ebx2, eby2), style=style, use_style_constraint=False)
                    if p_motif and p_motif["area_pct"] >= 0.8:
                        p_motif["is_primary_figure"] = True
                        raw_components.append(p_motif)
        except Exception as e:
            pass

        # 2. Complementary seed-prompted SAM for figures, limbs, and attributes not covered by YOLO
        for cx, cy, p_area in seeds:
            if any(e["mask"][cy, cx] for e in raw_components):
                continue
            motif = self.segment_at_point_or_box(image, point=(cx, cy), style=style, use_style_constraint=False)
            if motif and 0.35 <= motif["area_pct"] <= 28.0:
                bw_m = motif["bbox"][2]
                caspect = bw_m / (motif["bbox"][3] + 1e-6)
                if bw_m > w * 0.40 and caspect > 2.5:
                    continue
                dup = False
                for e in raw_components:
                    inter = np.sum(motif["mask"] & e["mask"])
                    iou = inter / (np.sum(motif["mask"]) + np.sum(e["mask"]) - inter + 1e-6)
                    if iou > 0.40:
                        dup = True
                        break
                if not dup:
                    raw_components.append(motif)

        # 3. Hierarchical Scene & Figure Resolution
        # A. Separate horizontal ornament bands (neck, rim, shoulder patterns)
        ornament_bands = []
        scene_elements = []
        for c in raw_components:
            bw, bh = c["bbox"][2], c["bbox"][3]
            caspect = bw / (bh + 1e-6)
            if (caspect > 3.2 and bh < h * 0.22) or (bw > w * 0.40 and bh < h * 0.22) or (bw > w * 0.60):
                c["label"] = f"Ornamentband #{len(ornament_bands)+1}"
                c["is_ornament"] = True
                ornament_bands.append(c)
            else:
                scene_elements.append(c)

        # B. Hierarchical Human Figure Assembly ("nach menschlichen Figuren schauen und alle Teile mitnehmen")
        # Step 1: Identify primary human figure bodies (cores)
        figure_cores = []
        parts_to_absorb = []
        for c in scene_elements:
            bh = c["bbox"][3]
            if bh >= h * 0.20 and c["area_pct"] >= 1.0 and c["bbox"][2] < w * 0.35:
                figure_cores.append(c)
            else:
                parts_to_absorb.append(c)

        # Sort figure cores from left to right along the horizontal frieze
        figure_cores.sort(key=lambda c: c["bbox"][0])
        
        # Deduplicate figure cores that belong to the same person standing (within 5% frieze width horizontally)
        distinct_figures = []
        for fc in figure_cores:
            fc_x = fc["bbox"][0] + fc["bbox"][2] / 2
            merged_into_existing = False
            for df in distinct_figures:
                df_x = df["core"]["bbox"][0] + df["core"]["bbox"][2] / 2
                if abs(fc_x - df_x) < (w * 0.05):
                    df["mask"] |= fc["mask"]
                    df["members"].append(fc)
                    merged_into_existing = True
                    break
            if not merged_into_existing:
                distinct_figures.append({
                    "core": fc,
                    "mask": fc["mask"].copy(),
                    "members": [fc]
                })

        # If no tall cores found (e.g. smaller scene), fallback to scene_elements
        if not distinct_figures:
            for el in scene_elements:
                distinct_figures.append({
                    "core": el,
                    "mask": el["mask"].copy(),
                    "members": [el]
                })
        else:
            # Step 2: Greedily absorb all adjacent limbs, outstretched arms, hands, spears, sceptres, wings, and lion skins!
            k_touch = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
            for part in parts_to_absorb:
                p_mask = part["mask"]
                pbx, pby, pbw, pbh = part["bbox"]
                pcx = pbx + pbw / 2
                
                best_fig = None
                best_score = float("inf")
                
                for fig in distinct_figures:
                    f_mask = fig["mask"]
                    fbx, fby, fbw, fbh = fig["core"]["bbox"]
                    fcx = fbx + fbw / 2
                    
                    # 1. Morphological contour connection (arm touches shoulder, hand touches spear/sceptre)
                    dilated = cv2.dilate(f_mask.astype(np.uint8), k_touch) > 0
                    touches = np.any(dilated & p_mask)
                    
                    # 2. Outstretched arm / horizontal proximity
                    x_dist = max(0, max(fbx, pbx) - min(fbx + fbw, pbx + pbw))
                    y_overlap = max(0, min(fby + fbh, pby + pbh) - max(fby, pby))
                    is_arm_or_attr = (x_dist < 45) and (y_overlap > 20 or touches)
                    
                    # 3. Vertical stacking (head, helmet, feet)
                    x_overlap = max(0, min(fbx + fbw, pbx + pbw) - max(fbx, pbx))
                    min_w = min(fbw, pbw)
                    is_vert = (x_overlap / (min_w + 1e-6) > 0.20) and (touches or y_overlap > 0)
                    
                    if touches or is_arm_or_attr or is_vert:
                        score = abs(pcx - fcx)
                        if score < best_score:
                            best_score = score
                            best_fig = fig
                            
                if best_fig is not None:
                    best_fig["mask"] |= p_mask
                    best_fig["members"].append(part)

        # Step 3: Recalculate complete human figure motifs with all parts included
        distinct_figures.sort(key=lambda g: g["core"]["bbox"][0])
        final_entities = []
        for idx, fig in enumerate(distinct_figures):
            parent_motif = self.recalculate_motif(image, fig["mask"], motif_id=len(final_entities))
            if parent_motif:
                parent_motif["id"] = len(final_entities)
                parent_motif["label"] = f"👤 Menschliche Figur #{idx+1}"
                parent_motif["sub_components"] = fig["members"]
                final_entities.append(parent_motif)
                
                # If composite with multiple parts, also include the discrete sub-parts
                if len(fig["members"]) > 1:
                    sub_sorted = sorted(fig["members"], key=lambda p: p["bbox"][1])
                    for s_idx, part in enumerate(sub_sorted):
                        p_motif = part.copy()
                        p_motif["id"] = len(final_entities)
                        part_desc = "Oberkörper / Kopf" if s_idx == 0 else ("Unterkörper / Stand" if s_idx == len(sub_sorted) - 1 else f"Teil #{s_idx+1}")
                        p_motif["label"] = f"  ↳ 🧩 {part_desc}"
                        p_motif["is_child"] = True
                        final_entities.append(p_motif)

        # Add ornament bands to final entities
        for ob in ornament_bands:
            ob["id"] = len(final_entities)
            final_entities.append(ob)

        # If still empty but raw_components exist, include raw components
        if not final_entities and raw_components:
            for rc in raw_components:
                rc["id"] = len(final_entities)
                final_entities.append(rc)

        for idx, m in enumerate(final_entities):
            m["id"] = idx
            if "label" not in m:
                m["label"] = f"Motiv #{idx+1}"
        return final_entities

    def segment_at_point_or_box(
        self,
        image: Image.Image,
        point: Optional[Tuple[int, int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None,
        points: Optional[List[Tuple[int, int]]] = None,
        labels: Optional[List[int]] = None,
        style: str = "auto",
        use_style_constraint: bool = True,
        invert: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Segment target motif using active backend with style structure constraint.
        """
        img_w, img_h = image.size
        struct_mask = None
        if use_style_constraint:
            struct_mask = self.extract_structure_mask(image, style=style, invert=invert)
        b_mode = getattr(self, "backend", "mobilesam").lower()

        # Handle ViT SAM backends (MobileSAM and Meta SAM)
        if "sam" in b_mode and "archaeological" not in b_mode:
            model_file = "sam_b.pt" if "meta" in b_mode else "mobile_sam.pt"
            if "fastsam" in b_mode:
                model_file = "FastSAM-s.pt"

            try:
                model = self._get_model(model_file)
                prompt_pts = []
                prompt_lbls = []

                if points is not None and labels is not None:
                    prompt_pts = points
                    prompt_lbls = labels
                elif point is not None:
                    prompt_pts = [[point[0], point[1]]]
                    prompt_lbls = [1]

                kwargs = {"retina_masks": True, "verbose": False}
                if prompt_pts:
                    kwargs["points"] = prompt_pts
                    kwargs["labels"] = prompt_lbls
                elif box is not None:
                    bx1, by1, bx2, by2 = box
                    min_x, max_x = min(bx1, bx2), max(bx1, bx2)
                    min_y, max_y = min(by1, by2), max(by1, by2)
                    kwargs["bboxes"] = [[min_x, min_y, max_x, max_y]]

                results = model(image, **kwargs)
                if results and results[0].masks is not None:
                    raw_masks = results[0].masks.data.cpu().numpy()
                    if len(raw_masks) > 0:
                        img_w, img_h = image.size
                        best_m = raw_masks[0]
                        if best_m.shape != (img_h, img_w):
                            best_m = cv2.resize(best_m.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0.5
                        else:
                            best_m = best_m > 0.5

                        # Post-process: Preserve complete organic figure mask including outstretched arms, spears, sceptres & headbands
                        num_c, c_labels, c_stats, _ = cv2.connectedComponentsWithStats(best_m.astype(np.uint8))
                        if num_c > 2:
                            target_label = 1
                            if point is not None and 0 <= point[1] < img_h and 0 <= point[0] < img_w:
                                p_lbl = c_labels[point[1], point[0]]
                                if p_lbl > 0:
                                    target_label = p_lbl
                                else:
                                    target_label = 1 + int(np.argmax(c_stats[1:, cv2.CC_STAT_AREA]))
                            else:
                                target_label = 1 + int(np.argmax(c_stats[1:, cv2.CC_STAT_AREA]))
                            
                            target_comp = (c_labels == target_label)
                            k_limb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
                            dil_target = cv2.dilate(target_comp.astype(np.uint8), k_limb) > 0
                            
                            # Keep target core plus all adjacent limbs, outstretched arms, and held attributes
                            kept_m = target_comp.copy()
                            for c_i in range(1, num_c):
                                if c_i != target_label:
                                    c_mask = (c_labels == c_i)
                                    # Include if touching / adjacent to figure within 35px
                                    if np.any(dil_target & c_mask):
                                        kept_m |= c_mask
                            best_m = kept_m
                            
                        # 2. Gentle closing to bridge fine internal relief lines, arm gaps & hair locks
                        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                        closed = cv2.morphologyEx(best_m.astype(np.uint8), cv2.MORPH_CLOSE, k_close) > 0
                        
                        # 3. Exclude white scan paper if any (never on grayscale drawings or white-painted skin)
                        img_np = np.array(image.convert("RGB"))
                        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
                        is_gray = np.mean(hsv[:, :, 1]) < 12.0
                        if not is_gray:
                            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                            white_page = (gray > 240) & (hsv[:, :, 1] < 20)
                            # Only if connected to outer image border
                            b_mask = np.zeros_like(white_page, dtype=np.uint8)
                            b_mask[0, :] = 1; b_mask[-1, :] = 1; b_mask[:, 0] = 1; b_mask[:, -1] = 1
                            num_w, w_labels, _, _ = cv2.connectedComponentsWithStats(white_page.astype(np.uint8))
                            for w_lbl in range(1, num_w):
                                if np.any(w_labels[b_mask == 1] == w_lbl):
                                    closed[w_labels == w_lbl] = False

                        return self.recalculate_motif(image, closed, motif_id=0)
            except Exception as e:
                print(f"ViT SAM failed ({e}), falling back to archaeological engine...")

        # Fallback / Direct Archaeological Engine
        if point is not None:
            res = self.extract_vase_figure_at_point(image, point[0], point[1])
            if res and struct_mask is not None:
                res = self.recalculate_motif(image, res["mask"] & (struct_mask > 0), motif_id=0)
            return res

        if box is not None:
            bx1, by1, bx2, by2 = box
            min_x, max_x = max(0, min(bx1, bx2)), min(image.width, max(bx1, bx2))
            min_y, max_y = max(0, min(by1, by2)), min(image.height, max(by1, by2))
            bw, bh = max_x - min_x, max_y - min_y

            if bw <= 4 or bh <= 4:
                return None

            img_np = np.array(image.convert("RGB"))
            roi_rgb = img_np[min_y:max_y, min_x:max_x]
            roi_gray = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2GRAY)

            dark_half = roi_gray[roi_gray < 128]
            glaze_thresh = max(20, min(50, int(np.mean(dark_half) + np.std(dark_half) * 0.5))) if len(dark_half) > 100 else 35

            is_glaze = roi_gray < glaze_thresh
            gc_mask = np.full((bh, bw), cv2.GC_PR_FGD, dtype=np.uint8)
            gc_mask[is_glaze] = cv2.GC_BGD
            gc_mask[0:2, :] = cv2.GC_BGD; gc_mask[-2:, :] = cv2.GC_BGD; gc_mask[:, 0:2] = cv2.GC_BGD; gc_mask[:, -2:] = cv2.GC_BGD

            bgd = np.zeros((1, 65), np.float64)
            fgd = np.zeros((1, 65), np.float64)

            try:
                cv2.grabCut(roi_rgb, gc_mask, (2, 2, bw - 4, bh - 4), bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
                fg = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
            except Exception:
                fg = ~is_glaze

            k_size = max(5, int(min(bh, bw) * 0.02))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            fg_closed = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, k)

            contours, _ = cv2.findContours(fg_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                main_cnt = max(contours, key=cv2.contourArea)
                filled_roi = np.zeros((bh, bw), dtype=np.uint8)
                cv2.drawContours(filled_roi, [main_cnt], -1, 1, thickness=-1)

                full_mask = np.zeros((image.height, image.width), dtype=bool)
                full_mask[min_y:max_y, min_x:max_x] = (filled_roi > 0)
                if struct_mask is not None:
                    full_mask = full_mask & (struct_mask > 0)
                return self.recalculate_motif(image, full_mask, motif_id=0)

        return None

    def extract_vase_figure_at_point(self, image: Image.Image, px: int, py: int) -> Optional[Dict[str, Any]]:
        """
        Extracts a clean, cohesive vase figure or ornament at (px, py) using adaptive GrabCut and color/edge analysis.
        Seamlessly separates red-figure/black-figure drawings from black glaze and book paper backgrounds,
        capturing everything that directly connects as ONE single closed polygon without background leakage.
        """
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape
        if not (0 <= px < w and 0 <= py < h):
            return None

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

        roi_rad_x = int(w * 0.45)
        roi_rad_y = int(h * 0.50)

        x0 = max(0, px - roi_rad_x)
        y0 = max(0, py - roi_rad_y)
        x1 = min(w, px + roi_rad_x)
        y1 = min(h, py + roi_rad_y)

        roi_rgb = img_np[y0:y1, x0:x1]
        roi_gray = gray[y0:y1, x0:x1]
        rh, rw = roi_gray.shape
        rcx, rcy = px - x0, py - y0

        max_dim = 600
        scale = min(1.0, max_dim / max(rh, rw))

        if scale < 1.0:
            small_rgb = cv2.resize(roi_rgb, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            small_gray = cv2.resize(roi_gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            s_rcx, s_rcy = int(rcx * scale), int(rcy * scale)
        else:
            small_rgb = roi_rgb
            small_gray = roi_gray
            s_rcx, s_rcy = rcx, rcy

        s_h, s_w = small_gray.shape

        dark_half = small_gray[small_gray < 128]
        if len(dark_half) > 100:
            glaze_thresh = max(20, min(50, int(np.mean(dark_half) + np.std(dark_half) * 0.5)))
        else:
            glaze_thresh = 35

        is_glaze = small_gray < glaze_thresh
        is_white_paper = (small_gray > 245) & (y0 == 0 or y1 == h or x0 == 0 or x1 == w)

        gc_mask = np.full((s_h, s_w), cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[is_glaze] = cv2.GC_BGD
        gc_mask[is_white_paper] = cv2.GC_BGD

        seed_r = max(4, int(min(s_h, s_w) * 0.03))
        cv2.circle(gc_mask, (s_rcx, s_rcy), seed_r, cv2.GC_FGD, -1)

        seed_color = small_rgb[s_rcy, s_rcx].astype(np.float32)
        col_diff = np.linalg.norm(small_rgb.astype(np.float32) - seed_color, axis=2)
        gc_mask[(col_diff < 60) & (~is_glaze)] = cv2.GC_PR_FGD

        gc_mask[0:3, :] = cv2.GC_BGD
        gc_mask[-3:, :] = cv2.GC_BGD
        gc_mask[:, 0:3] = cv2.GC_BGD
        gc_mask[:, -3:] = cv2.GC_BGD

        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(small_rgb, gc_mask, (2, 2, s_w - 4, s_h - 4), bgd, fgd, 2, cv2.GC_INIT_WITH_MASK)
            fg_mask_small = (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD)
        except Exception:
            fg_mask_small = (~is_glaze) & (~is_white_paper)

        if scale < 1.0:
            fg_mask = cv2.resize(fg_mask_small.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST) > 0
        else:
            fg_mask = fg_mask_small

        fg_mask &= (roi_gray >= max(15, glaze_thresh - 5))

        k_size = max(7, int(min(rh, rw) * 0.02))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        fg_closed = cv2.morphologyEx(fg_mask.astype(np.uint8), cv2.MORPH_CLOSE, k)

        num_l, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_closed, connectivity=8)
        target_lbl = labels[rcy, rcx]

        if target_lbl == 0:
            for dist in range(1, 30):
                patch = labels[max(0, rcy - dist):min(rh, rcy + dist), max(0, rcx - dist):min(rw, rcx + dist)]
                nz = patch[patch > 0]
                if len(nz) > 0:
                    target_lbl = nz[0]
                    break

        final_roi_mask = (labels == target_lbl).astype(np.uint8) if target_lbl > 0 else np.zeros((rh, rw), np.uint8)

        contours, _ = cv2.findContours(final_roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        main_cnt = max(contours, key=cv2.contourArea)
        filled_roi = np.zeros((rh, rw), dtype=np.uint8)
        cv2.drawContours(filled_roi, [main_cnt], -1, 1, thickness=-1)

        full_mask = np.zeros((h, w), dtype=bool)
        full_mask[y0:y1, x0:x1] = (filled_roi > 0)

        return self.recalculate_motif(image, full_mask, motif_id=0)

    def draw_stroke_on_mask(
        self,
        mask: np.ndarray,
        points: List[Tuple[int, int]],
        value: int = 1,
        radius: int = 12
    ) -> np.ndarray:
        """
        Draws or erases a brush/pen stroke along a sequence of (x, y) coordinates on a binary mask.
        value=1 for addition (pen), value=0 for subtraction (eraser).
        """
        if not points:
            return mask

        new_mask = mask.astype(np.uint8).copy()
        thickness = max(2, radius * 2)

        # Single dot click
        if len(points) == 1:
            cv2.circle(new_mask, points[0], radius, value, -1)
            return new_mask > 0

        # Continuous smooth stroke
        for i in range(len(points) - 1):
            cv2.line(new_mask, points[i], points[i + 1], value, thickness=thickness)
            cv2.circle(new_mask, points[i + 1], radius, value, -1)

        return new_mask > 0

    def draw_segmentation_overlay(
        self,
        image: Image.Image,
        motifs: List[Dict[str, Any]],
        selected_id: Optional[int] = None,
        alpha: float = 0.45
    ) -> Image.Image:
        """
        Draws semi-transparent colored masks and bounding labels on the original image.
        """
        img_np = np.array(image.convert("RGB"))
        overlay = img_np.copy()
        h, w, _ = img_np.shape

        palette = [
            (59, 130, 246),   # Blue
            (245, 158, 11),   # Amber
            (16, 185, 129),   # Emerald
            (236, 72, 153),   # Pink
            (168, 85, 247),   # Purple
            (6, 182, 212),    # Cyan
            (239, 68, 68),    # Red
            (132, 204, 22),   # Lime
        ]

        for idx, m in enumerate(motifs):
            is_selected = (selected_id is not None and m.get("id") == selected_id)
            color = (255, 215, 0) if is_selected else palette[idx % len(palette)]
            
            raw_m = np.squeeze(m["mask"])
            if raw_m.ndim != 2 or raw_m.shape != (h, w):
                mask = cv2.resize(raw_m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            else:
                mask = raw_m > 0

            if not np.any(mask):
                continue

            overlay[mask] = (overlay[mask].astype(np.float32) * (1.0 - alpha) + np.array(color, dtype=np.float32) * alpha).astype(np.uint8)

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            border_thickness = 3 if is_selected else 2
            cv2.drawContours(overlay, contours, -1, color, border_thickness)

            x1, y1, bw, bh = m["bbox"]
            label_text = f"#{idx+1}"
            if m.get("label"):
                label_text += f": {m['label']}"
            cv2.rectangle(overlay, (x1, max(0, y1 - 22)), (x1 + len(label_text)*10 + 10, y1), color, -1)
            cv2.putText(overlay, label_text, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

        return Image.fromarray(overlay)

    def dilate_mask(self, mask: np.ndarray, pixels: int = 6) -> np.ndarray:
        k = max(3, pixels * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.dilate(mask.astype(np.uint8), kernel) > 0

    def erode_mask(self, mask: np.ndarray, pixels: int = 6) -> np.ndarray:
        k = max(3, pixels * 2 + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        return cv2.erode(mask.astype(np.uint8), kernel) > 0

    def merge_masks(self, mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
        return (mask_a > 0) | (mask_b > 0)

    def recalculate_motif(
        self,
        image: Image.Image,
        mask: np.ndarray,
        motif_id: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Recomputes bounding box, single continuous polygon, and RGBA cutout for an updated mask.
        """
        img_np = np.array(image.convert("RGB"))
        h, w, _ = img_np.shape
        raw_m = np.squeeze(mask)
        if raw_m.shape != (h, w):
            raw_m = cv2.resize(raw_m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0

        y_indices, x_indices = np.where(raw_m > 0)
        if len(y_indices) == 0:
            return None

        x1, x2 = int(np.min(x_indices)), int(np.max(x_indices))
        y1, y2 = int(np.min(y_indices)), int(np.max(y_indices))
        bw = max(1, x2 - x1 + 1)
        bh = max(1, y2 - y1 + 1)

        rgba_crop = np.zeros((bh, bw, 4), dtype=np.uint8)
        crop_rgb = img_np[y1:y2+1, x1:x2+1]
        crop_mask = raw_m[y1:y2+1, x1:x2+1]

        rgba_crop[:, :, :3] = crop_rgb
        rgba_crop[:, :, 3] = (crop_mask * 255).astype(np.uint8)
        cutout_pil = Image.fromarray(rgba_crop, "RGBA")

        # Guarantee 1 clean continuous closed polygon
        contours, _ = cv2.findContours(raw_m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygon = []
        if contours:
            largest_cnt = max(contours, key=cv2.contourArea)
            epsilon = 0.0015 * cv2.arcLength(largest_cnt, True)
            approx_cnt = cv2.approxPolyDP(largest_cnt, epsilon, True)
            polygon = approx_cnt.reshape(-1, 2).tolist()

        area = int(np.sum(raw_m > 0))
        area_pct = round((area / float(w * h)) * 100.0, 2)

        return {
            "id": motif_id,
            "bbox": [x1, y1, bw, bh],
            "area": area,
            "area_pct": area_pct,
            "cutout": cutout_pil,
            "mask": raw_m > 0,
            "polygon": polygon,
            "center": (int(x1 + bw / 2), int(y1 + bh / 2))
        }

sam_segmenter = SAMSegmenter()
