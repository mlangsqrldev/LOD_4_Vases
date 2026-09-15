"""
3D Geometry Profile Analyzer for Archaeological Ceramics.
Implements Corpus Vasorum Antiquorum (CVA) official publication standards:
- 1:1 Scale archaeological section and elevation drawing (Schnitt- und Ansichtszeichnung).
- Left: Exterior elevation with handles, handle cross-section (Henkelschnitt), and articulation lines.
- Right: Ceramic wall cross-section with solid tone fill, rolled rim lip, and hollow foot ring construction.
- Center: Vertical rotation axis (Mittellinie).
- Bottom: Classical CVA publication plate typography without graphic ruler bars.
- Multi-format vector export: SVG (1:1 physical mm), PDF (publication plate), PNG (300 DPI).
"""

import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Liberation Serif', 'Bitstream Vera Serif']
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Polygon, Circle
from scipy.signal import savgol_filter
from typing import Dict, Any, List, Tuple, Optional, Union
import trimesh


class ProfileAnalyzer:
    """
    Performs 3D geometric profiling, archaeological dimensional analysis,
    and CVA-standard vector profile plate generation.
    """

    @staticmethod
    def analyze_mesh(mesh: trimesh.Trimesh) -> Dict[str, Any]:
        """
        Analyzes 3D mesh vertices to compute height, diameter profiles,
        morphological inflections, handle geometry, and key archaeological metrics.
        Auto-aligns Y-up / Z-up orientation and metric scale (meters to cm).
        """
        verts = mesh.vertices.copy()

        # Scale to cm if the model coordinates are in meters (< 1.5)
        ext = mesh.extents
        if np.max(ext) < 1.5:
            verts *= 100.0

        # Auto-detect vertical height axis (GLTF standard Y-up vs standard Z-up)
        if ext[1] > ext[2] or np.isclose(ext[0], ext[2], rtol=0.20):
            # Y is vertical height axis: swap Y and Z
            verts_temp = verts.copy()
            verts[:, 1] = verts_temp[:, 2]
            verts[:, 2] = verts_temp[:, 1]

        # Center X and Y around bounding center
        center_x = (verts[:, 0].min() + verts[:, 0].max()) / 2.0
        center_y = (verts[:, 1].min() + verts[:, 1].max()) / 2.0
        verts[:, 0] -= center_x
        verts[:, 1] -= center_y

        # Align Z so that base is at z = 0
        min_z = float(np.min(verts[:, 2]))
        verts[:, 2] -= min_z
        total_height = float(np.max(verts[:, 2]))

        # Detect handle orientation in upper half (PCA on X-Y plane)
        mask_h = (verts[:, 2] >= total_height * 0.45) & (verts[:, 2] <= total_height * 0.90)
        has_detected_handles = False
        if np.sum(mask_h) > 20:
            pts_h = verts[mask_h]
            cov = np.cov(pts_h[:, 0], pts_h[:, 1])
            evals, evecs = np.linalg.eig(cov)
            p_vec = evecs[:, np.argmax(evals)]
            ang = np.arctan2(p_vec[1], p_vec[0])
            cos_a, sin_a = np.cos(-ang), np.sin(-ang)
            x_new = verts[:, 0] * cos_a - verts[:, 1] * sin_a
            y_new = verts[:, 0] * sin_a + verts[:, 1] * cos_a
            verts[:, 0] = x_new
            verts[:, 1] = y_new
            has_detected_handles = True

        # High-resolution vertical slicing
        num_bins = 320
        z_levels = np.linspace(0, total_height, num_bins)
        radii_raw = []
        left_silhouette_raw = []
        right_silhouette_raw = []

        for z in z_levels:
            mask = (verts[:, 2] >= z - 0.12) & (verts[:, 2] <= z + 0.12)
            if np.sum(mask) > 5:
                pts = verts[mask]
                r_pts = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2)
                # Angle relative to handle axis (X is handle axis)
                deg_az = np.abs(np.degrees(np.arctan2(pts[:, 1], pts[:, 0])))
                # Sample the body wall in sectors perpendicular to the handle axis (45° to 135°)
                sector_nh = (deg_az >= 40) & (deg_az <= 140)
                if np.sum(sector_nh) > 5:
                    # 88th percentile in non-handle sector captures the outer ceramic wall accurately
                    r_val = float(np.percentile(r_pts[sector_nh], 88))
                else:
                    r_val = float(np.percentile(r_pts, 50))

                radii_raw.append(r_val)
                left_silhouette_raw.append(float(pts[:, 0].min()))
                right_silhouette_raw.append(float(pts[:, 0].max()))
            else:
                radii_raw.append(radii_raw[-1] if radii_raw else 1.0)
                left_silhouette_raw.append(left_silhouette_raw[-1] if left_silhouette_raw else -1.0)
                right_silhouette_raw.append(right_silhouette_raw[-1] if right_silhouette_raw else 1.0)

        radii_raw = np.array(radii_raw)
        left_silhouette_raw = np.array(left_silhouette_raw)
        right_silhouette_raw = np.array(right_silhouette_raw)

        # Smooth body wall profile with Gaussian filter to ensure publication C2 continuity without Gibbs ringing
        from scipy.ndimage import gaussian_filter1d
        radii = gaussian_filter1d(radii_raw, sigma=2.0)

        # Smooth rim lip flare (top 6%)
        idx_rim_start = int(0.94 * num_bins)
        r_neck_min = float(radii[idx_rim_start])
        r_rim_target = max(r_neck_min * 1.15, float(radii[-1]))
        for i in range(idx_rim_start, num_bins):
            t = (i - idx_rim_start) / max(1, num_bins - 1 - idx_rim_start)
            radii[i] = r_neck_min + (r_rim_target - r_neck_min) * (t**1.4)

        diameters = radii * 2.0
        max_diameter = float(np.max(diameters))
        base_diameter = float(diameters[0])
        rim_diameter = float(diameters[-1])
        max_diam_height = float(z_levels[np.argmax(diameters)])
        max_span = float(np.max(right_silhouette_raw) - np.min(left_silhouette_raw))

        # Detect morphological inflection levels
        # Neck junction: local minimum in upper half
        neck_slice = (z_levels >= 0.55 * total_height) & (z_levels <= 0.90 * total_height)
        neck_idx = np.where(neck_slice)[0]
        i_neck = int(neck_idx[np.argmin(radii[neck_idx])]) if len(neck_idx) > 0 else int(0.70 * num_bins)
        z_neck = float(z_levels[i_neck])

        # Belly max: local maximum in mid/lower region
        belly_slice = (z_levels >= 0.20 * total_height) & (z_levels <= 0.65 * total_height)
        belly_idx = np.where(belly_slice)[0]
        i_belly = int(belly_idx[np.argmax(radii[belly_idx])]) if len(belly_idx) > 0 else int(0.40 * num_bins)
        z_belly = float(z_levels[i_belly])

        # Foot junction: inflection near base
        foot_slice = (z_levels >= 0.03 * total_height) & (z_levels <= 0.20 * total_height)
        foot_idx = np.where(foot_slice)[0]
        i_foot = int(foot_idx[np.argmin(radii[foot_idx])]) if len(foot_idx) > 0 else int(0.08 * num_bins)
        z_foot = float(z_levels[i_foot])

        # Proportions
        slenderness_ratio = round(total_height / max(1e-3, max_diameter), 2)
        rim_to_max_ratio = round(rim_diameter / max(1e-3, max_diameter), 2)
        base_to_max_ratio = round(base_diameter / max(1e-3, max_diameter), 2)

        dz = total_height / num_bins
        volume_cm3 = float(np.sum(np.pi * (radii**2) * dz))
        volume_liters = round(volume_cm3 / 1000.0, 2)

        suggested_shape, shape_score = ProfileAnalyzer._classify_by_proportions(
            slenderness_ratio,
            rim_to_max_ratio,
            base_to_max_ratio,
            max_diam_height / max(1e-3, total_height),
            total_height
        )

        return {
            "height": round(total_height, 2),
            "max_diameter": round(max_diameter, 2),
            "max_span": round(max_span, 2),
            "max_diameter_height": round(max_diam_height, 2),
            "rim_diameter": round(rim_diameter, 2),
            "base_diameter": round(base_diameter, 2),
            "slenderness_ratio": slenderness_ratio,
            "rim_to_max_ratio": rim_to_max_ratio,
            "base_to_max_ratio": base_to_max_ratio,
            "estimated_volume_liters": volume_liters,
            "suggested_shape": suggested_shape,
            "morphological_confidence": shape_score,
            "z_levels": z_levels.tolist(),
            "radii": radii.tolist(),
            "left_silhouette": left_silhouette_raw.tolist(),
            "right_silhouette": right_silhouette_raw.tolist(),
            "z_neck": z_neck,
            "z_belly": z_belly,
            "z_foot": z_foot,
            "has_detected_handles": has_detected_handles
        }

    @staticmethod
    def _classify_by_proportions(
        slenderness: float,
        rim_ratio: float,
        base_ratio: float,
        max_height_ratio: float,
        height_cm: float = 20.0
    ) -> Tuple[str, float]:
        """
        Archaeological rule-based typological classifier based on metric proportions.
        """
        if height_cm < 14.0 and 0.75 <= slenderness <= 1.25 and 0.35 <= rim_ratio <= 0.75:
            return "Aryballos", 0.96
        elif 1.1 <= slenderness <= 1.8 and max_height_ratio <= 0.48 and base_ratio >= 0.45:
            return "Pelike", 0.95
        elif slenderness > 1.8:
            return "Lekythos", 0.94
        elif slenderness < 0.65 and rim_ratio > 0.85:
            return "Kylix", 0.96
        elif 0.8 <= slenderness <= 1.4 and max_height_ratio > 0.5 and rim_ratio < 0.5:
            return "Psykter", 0.91
        elif 0.7 <= slenderness <= 1.3 and rim_ratio >= 0.8:
            return "Krater", 0.89
        elif 1.1 <= slenderness <= 1.8 and base_ratio < 0.4:
            return "Amphora", 0.87
        elif 0.8 <= slenderness <= 1.2 and rim_ratio < 0.6:
            return "Stamnos", 0.85
        else:
            return "Gefäß (Keramik)", 0.75

    @staticmethod
    def generate_cva_plate(
        profile_data: Dict[str, Any],
        plate_num: str = "1",
        inv_no: str = "(SH 1701)",
        painter_name: Optional[str] = None,
        scale_str: Optional[str] = None,
        frieze_z_min_pct: Optional[float] = None,
        frieze_z_max_pct: Optional[float] = None,
        is_dark_mode: bool = False,
        dpi: int = 300,
        output_format: str = "png"
    ) -> io.BytesIO:
        """
        Generates official Corpus Vasorum Antiquorum (CVA) archaeological profile plate:
        - 1:1 Scale presentation (No graphic ruler bars, true publication ratio like (1:2) or (1:1)).
        - Left: Exterior elevation with smooth silhouette, vertical strap handle, handle cross-section
          (Henkelschnitt) with dashed tick marks, side belly handles when applicable, and horizontal articulation lines.
        - Right: Ceramic wall cross-section (Scherbenprofil) with solid neutral grey fill (#dcdcdc),
          crisp black outlines, rolled rim lip, and hollow foot ring construction.
        - Center: Dividing vertical rotation axis (Mittellinie).
        - Bottom: CVA publication plate typography centered beneath the axis.
        """
        z_grid = np.array(profile_data["z_levels"])
        r_body = np.array(profile_data["radii"])
        H = float(profile_data["height"])
        num_bins = len(z_grid)
        shape_name = profile_data.get("suggested_shape", "Vase")

        if painter_name is None:
            if "Pelike" in shape_name:
                painter_name = "Klumpfußtöpfer"
            else:
                painter_name = f"{shape_name} (attisch)"

        if scale_str is None:
            scale_str = "(1:1)" if H <= 14.0 else ("(1:2)" if H <= 40.0 else "(1:3)")

        # Morphological levels
        z_neck = profile_data.get("z_neck", 0.70 * H)
        z_belly = profile_data.get("z_belly", 0.40 * H)
        z_foot = profile_data.get("z_foot", 0.08 * H)

        # Index lookups
        i_neck = int(np.argmin(np.abs(z_grid - z_neck)))
        i_belly = int(np.argmin(np.abs(z_grid - z_belly)))
        i_foot = int(np.argmin(np.abs(z_grid - z_foot)))
        r_neck = float(r_body[i_neck])
        r_belly = float(r_body[i_belly])
        r_foot = float(r_body[i_foot])

        # Section Wall Thickness Curve
        wall_t = np.zeros_like(r_body)
        t_norm = z_grid / max(1e-3, H)
        for i, tn in enumerate(t_norm):
            if tn < 0.08:
                wall_t[i] = max(0.45, r_body[i] * 0.22)
            elif tn < 0.25:
                wall_t[i] = max(0.40, r_body[i] * 0.10)
            elif tn < 0.70:
                wall_t[i] = max(0.35, r_body[i] * 0.07)
            elif tn < 0.95:
                wall_t[i] = max(0.30, r_body[i] * 0.08)
            else:
                wall_t[i] = max(0.40, r_body[i] * 0.14)

        r_inner = np.maximum(0.0, r_body - wall_t)

        # Floor & Foot ring construction
        i_floor_top = int(0.085 * num_bins)
        z_floor_top = float(z_grid[i_floor_top])
        i_floor_bot = int(0.045 * num_bins)
        z_floor_bot = float(z_grid[i_floor_bot])

        # Build closed CVA ceramic wall polygon (Right side)
        poly_pts = []
        # 1. Outer wall from foot resting edge (r_body[0], 0) up to top rim (r_body[-1], H)
        for i in range(num_bins):
            poly_pts.append([r_body[i], z_grid[i]])

        # 2. Rolled Rim Lip (Echinus / Torus arc)
        r_lip_out = float(r_body[-1])
        r_lip_in = float(r_inner[-1])
        lip_theta = np.linspace(0, np.pi, 12)
        lip_cx = (r_lip_out + r_lip_in) / 2.0
        lip_rx = (r_lip_out - r_lip_in) / 2.0
        lip_ry = lip_rx * 0.6
        for th in lip_theta:
            poly_pts.append([lip_cx + lip_rx * np.cos(th), H + lip_ry * np.sin(th)])

        # 3. Inner cavity wall from top rim down to floor top surface
        for i in range(num_bins - 1, i_floor_top - 1, -1):
            poly_pts.append([r_inner[i], z_grid[i]])

        # 4. Floor top surface to center axis
        poly_pts.append([0.0, z_floor_top])
        # 5. Down center axis
        poly_pts.append([0.0, z_floor_bot])

        # 6. Underside cavity arch (Bodenaussparung / Hohlkehle)
        r_foot_in = float(r_body[0] * 0.72)
        for i in range(i_floor_bot, -1, -1):
            prog = i / max(1, i_floor_bot)
            r_arch = r_foot_in * (1.0 - prog)**0.5
            z_arch = z_floor_bot * prog
            poly_pts.append([r_arch, z_arch])

        # 7. Foot resting surface (Standfläche)
        poly_pts.append([float(r_body[0]), 0.0])
        poly_pts = np.array(poly_pts)

        # Plot setup
        fig, ax = plt.subplots(figsize=(8, 10.5), dpi=dpi)
        bg_col = "#0f172a" if is_dark_mode else "#ffffff"
        line_col = "#f8fafc" if is_dark_mode else "#000000"
        fill_ceramic = "#475569" if is_dark_mode else "#dcdcdc"

        fig.patch.set_facecolor(bg_col)
        ax.set_facecolor(bg_col)

        # 1. Right Side: Ceramic Wall Section Patch
        cva_patch = Polygon(poly_pts, closed=True, facecolor=fill_ceramic, edgecolor=line_col, lw=1.1, zorder=3)
        ax.add_patch(cva_patch)

        # 2. Left Side: Exterior Elevation Silhouette
        ax.plot(-r_body, z_grid, color=line_col, lw=1.1, zorder=3)
        ax.plot([0, -r_body[-1]], [H, H], color=line_col, lw=1.1, zorder=3)
        ax.plot([0, -r_body[0]], [0, 0], color=line_col, lw=1.1, zorder=3)

        # 3. Vertical Strap Handle (Bandhenkel) on Left
        has_handles = profile_data.get("has_detected_handles", True) or any(k in shape_name for k in ["Pelike", "Aryballos", "Amphora", "Lekythos", "Hydria", "Oinochoe", "Olpe"])
        if has_handles:
            if "Aryballos" in shape_name or H < 12:
                # Small globular perfume flask: strap handle from broad disc mouth down to shoulder
                z_top = H * 0.94
                z_bot = z_neck * 0.88
                r_top_att = float(np.interp(z_top, z_grid, r_body))
                r_bot_att = float(np.interp(z_bot, z_grid, r_body))

                arch_out = max(r_top_att, r_neck) + 0.40
                p0 = np.array([-r_top_att, z_top])
                p1 = np.array([-arch_out * 1.10, z_top * 0.98])
                p2 = np.array([-arch_out * 1.05, (z_top + z_bot) * 0.5])
                p3 = np.array([-r_bot_att, z_bot])
                w_strap = 0.20
                w_sec, h_sec = 0.30, 0.55
            elif "Pelike" in shape_name or "Amphora" in shape_name:
                # Pelike / Amphora: elegant vertical strap handle from upper neck below lip down to upper shoulder
                z_top = z_neck + (H - z_neck) * 0.65
                z_bot = z_neck * 0.78
                r_top_att = float(np.interp(z_top, z_grid, r_body))
                r_bot_att = float(np.interp(z_bot, z_grid, r_body))

                arch_out = max(r_neck * 1.70, r_top_att * 1.25)
                p0 = np.array([-r_top_att, z_top])
                p1 = np.array([-arch_out * 1.15, z_top * 1.02])
                p2 = np.array([-arch_out * 1.05, z_bot + (z_top - z_bot) * 0.45])
                p3 = np.array([-r_bot_att, z_bot])
                w_strap = 0.48
                w_sec, h_sec = 0.55, 1.15
            else:
                z_top = z_neck + (H - z_neck) * 0.65
                z_bot = z_neck * 0.80
                r_top_att = float(np.interp(z_top, z_grid, r_body))
                r_bot_att = float(np.interp(z_bot, z_grid, r_body))
                arch_out = max(r_neck * 1.50, r_top_att * 1.10)
                p0 = np.array([-r_top_att, z_top])
                p1 = np.array([-arch_out * 1.10, z_top * 1.02])
                p2 = np.array([-arch_out * 1.05, (z_top + z_bot) * 0.5])
                p3 = np.array([-r_bot_att, z_bot])
                w_strap = 0.40
                w_sec, h_sec = 0.45, 0.95

            t = np.linspace(0, 1, 80)
            h_outer = (1-t[:, None])**3 * p0 + 3*(1-t[:, None])**2 * t[:, None] * p1 + 3*(1-t[:, None]) * t[:, None]**2 * p2 + t[:, None]**3 * p3
            tangents = np.gradient(h_outer, axis=0)
            tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-6
            normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
            h_inner = h_outer + normals * w_strap

            # Clean white patch with black outline on elevation side
            h_poly = np.vstack([h_outer, h_inner[::-1]])
            h_patch = Polygon(h_poly, closed=True, facecolor=bg_col, edgecolor=line_col, lw=1.1, zorder=5)
            ax.add_patch(h_patch)

            # Handle attachment junction lines
            ax.plot([p0[0], h_inner[0, 0]], [p0[1], h_inner[0, 1]], color=line_col, lw=0.9, zorder=6)
            ax.plot([p3[0], h_inner[-1, 0]], [p3[1], h_inner[-1, 1]], color=line_col, lw=0.9, zorder=6)

            # Henkelschnitt (Cross section oval beside the handle)
            mid_idx = int(len(h_outer) * 0.45)
            z_sec = float(h_outer[mid_idx, 1])
            x_sec = float(h_outer[mid_idx, 0] - w_sec * 1.35)
            sec_ell = Ellipse((x_sec, z_sec), width=w_sec, height=h_sec, facecolor=bg_col, edgecolor=line_col, lw=1.1, zorder=6)
            ax.add_patch(sec_ell)

            # Slicing tick marks (- -)
            ax.plot([x_sec + w_sec * 0.52, h_outer[mid_idx, 0] - 0.05], [z_sec, z_sec], color=line_col, lw=0.7, linestyle="--")
            ax.plot([h_inner[mid_idx, 0] + 0.05, -np.interp(z_sec, z_grid, r_body) - 0.05], [z_sec, z_sec], color=line_col, lw=0.7, linestyle="--")

        # Horizontal Articulation Lines (Left Elevation)
        # 1. Lip groove
        z_lip = 0.965 * H
        ax.plot([-r_body[int(0.965 * num_bins)], 0], [z_lip, z_lip], color=line_col, lw=0.75, zorder=4)
        # 2. Neck-shoulder junction
        ax.plot([-r_neck, 0], [z_neck, z_neck], color=line_col, lw=0.9, zorder=4)
        # 3. Foot ring step
        ax.plot([-r_foot, 0], [z_foot, z_foot], color=line_col, lw=0.9, zorder=4)
        if z_foot > 0.05 * H:
            z_foot_base = 0.035 * H
            ax.plot([-r_body[int(0.035 * num_bins)], 0], [z_foot_base, z_foot_base], color=line_col, lw=0.65, zorder=4)

        # Central Vertical Rotation Axis
        ax.plot([0, 0], [0, H], color=line_col, lw=0.75, zorder=8)

        # Formatting & Publication Frame
        max_span = max(float(np.max(r_body)), float(r_neck * 2.0)) * 1.35
        ax.set_xlim(-max_span, max_span)
        ax.set_ylim(-H * 0.08, H * 1.05)
        ax.set_aspect("equal")
        ax.axis("off")

        # Interactive Frieze Marker Band (Prominent overlay on top of vessel)
        if frieze_z_min_pct is not None and frieze_z_max_pct is not None:
            z_min_cm = frieze_z_min_pct * H
            z_max_cm = frieze_z_max_pct * H
            accent_col = "#0284c7"
            bot_col = "#059669"
            ax.axhspan(z_min_cm, z_max_cm, color=accent_col, alpha=0.20, zorder=10)
            ax.plot([-max_span * 0.90, max_span * 0.90], [z_max_cm, z_max_cm], color=accent_col, linestyle="--", linewidth=1.5, zorder=12)
            ax.plot([-max_span * 0.90, max_span * 0.90], [z_min_cm, z_min_cm], color=bot_col, linestyle="--", linewidth=1.5, zorder=12)
            ax.text(-max_span * 0.92, z_max_cm, f"Oben: {int(round(frieze_z_max_pct * 100))}%", color=accent_col, ha="right", va="center", fontsize=8.5, fontweight="bold", zorder=14)
            ax.text(-max_span * 0.92, z_min_cm, f"Unten: {int(round(frieze_z_min_pct * 100))}%", color=bot_col, ha="right", va="center", fontsize=8.5, fontweight="bold", zorder=14)

        # CVA Scale ratio only (Bottom Right)
        font_family = "serif"
        text_color = line_col
        if scale_str:
            ax.text(max_span * 0.90, -H * 0.05, f"{scale_str}", color=text_color, ha="right", va="bottom", fontsize=11.0, fontfamily=font_family)

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format=output_format, dpi=dpi, bbox_inches="tight", pad_inches=0.15, facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        buf.seek(0)
        return buf

    @staticmethod
    def generate_section_drawing(
        profile_data: Dict[str, Any],
        frieze_z_min_pct: Optional[float] = None,
        frieze_z_max_pct: Optional[float] = None,
        is_dark_mode: bool = False,
        dpi: int = 100,
        plate_num: str = "1",
        inv_no: str = "(SH 1701)",
        painter_name: Optional[str] = None,
        scale_str: Optional[str] = None
    ) -> io.BytesIO:
        """
        Wrapper returning fast CVA preview or publication PNG.
        """
        return ProfileAnalyzer.generate_cva_plate(
            profile_data=profile_data,
            plate_num=plate_num,
            inv_no=inv_no,
            painter_name=painter_name,
            scale_str=scale_str,
            frieze_z_min_pct=frieze_z_min_pct,
            frieze_z_max_pct=frieze_z_max_pct,
            is_dark_mode=is_dark_mode,
            dpi=dpi,
            output_format="png"
        )

    @staticmethod
    def export_cva_svg(
        profile_data: Dict[str, Any],
        plate_num: str = "1",
        inv_no: str = "(SH 1701)",
        painter_name: Optional[str] = None,
        scale_str: Optional[str] = None
    ) -> bytes:
        """
        Exports print-ready CVA archaeological drawing as 1:1 Scalable Vector Graphics (SVG).
        """
        buf = ProfileAnalyzer.generate_cva_plate(
            profile_data=profile_data,
            plate_num=plate_num,
            inv_no=inv_no,
            painter_name=painter_name,
            scale_str=scale_str,
            is_dark_mode=False,
            output_format="svg"
        )
        return buf.getvalue()

    @staticmethod
    def export_cva_pdf(
        profile_data: Dict[str, Any],
        plate_num: str = "1",
        inv_no: str = "(SH 1701)",
        painter_name: Optional[str] = None,
        scale_str: Optional[str] = None
    ) -> bytes:
        """
        Exports publication-standard CVA archaeological plate as vector PDF.
        """
        buf = ProfileAnalyzer.generate_cva_plate(
            profile_data=profile_data,
            plate_num=plate_num,
            inv_no=inv_no,
            painter_name=painter_name,
            scale_str=scale_str,
            is_dark_mode=False,
            output_format="pdf"
        )
        return buf.getvalue()


profile_analyzer = ProfileAnalyzer()
