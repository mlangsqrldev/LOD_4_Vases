"""
LODVases - CustomTkinter Desktop Application
Ancient Ceramics Vision AI Classifier, SAM Motif Segmenter, 3D GLB Studio & SKOS LOD Indexer
BCDH Universität Bonn | HECTOR Heritage Assets Vocabulary
"""

import os
import io
import json
import base64
import threading
from typing import List, Dict, Any, Tuple, Optional
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import cv2

from lodvases.classifier import classifier, SHAPE_CLASSES, WARE_CLASSES, SCENE_CLASSES
from lodvases.skos_client import skos_client
from lodvases.lod_exporter import LODExporter
from lodvases.metadata_parser import load_directory_metadata, VaseMetadata
from lodvases.sam_segmenter import sam_segmenter
from lodvases.dataset_exporter import DatasetExporter
from lodvases.detector import figure_detector
from lodvases.glb_processor import glb_processor
from lodvases.profile_analyzer import profile_analyzer
from lodvases.babylon_viewer import babylon_viewer

ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("blue")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PICS_DIR = os.path.join(BASE_DIR, "pics")

# -----------------------------------------------------------------------------
# Modern Semantic Design Token System (WCAG AA Compliant Light & Dark)
# -----------------------------------------------------------------------------
THEME = {
    # Surfaces & Canvas
    "bg_app": ("#f8fafc", "#0b0f19"),
    "bg_sidebar": ("#ffffff", "#0f172a"),
    "bg_card": ("#ffffff", "#1e293b"),
    "bg_card_inner": ("#f1f5f9", "#0f172a"),
    "bg_canvas": ("#ffffff", "#0f172a"),
    "border_subtle": ("#e2e8f0", "#334155"),
    "border_focus": ("#0284c7", "#38bdf8"),

    # Typography / Text
    "text_primary": ("#0f172a", "#f8fafc"),
    "text_secondary": ("#475569", "#94a3b8"),
    "text_muted": ("#64748b", "#64748b"),
    "text_accent": ("#0284c7", "#38bdf8"),
    "text_on_accent": "#ffffff",

    # Brand Colors & Interactive Roles
    "primary": "#0284c7",
    "primary_hover": "#0369a1",
    "ai_purple": "#7c3aed",
    "ai_purple_hover": "#6d28d9",
    "secondary_btn": ("#e2e8f0", "#334155"),
    "secondary_btn_hover": ("#cbd5e1", "#475569"),
    "nav_hover": ("#f1f5f9", "#1e293b"),
    "success": ("#10b981", "#059669"),
    "warning": ("#f59e0b", "#d97706"),
    "danger": "#ef4444",
    "danger_hover": "#dc2626",
}


class ZoomableMotifPreviewDialog(ctk.CTkToplevel):
    def __init__(self, master, motif_data: Dict[str, Any]):
        super().__init__(master)
        self.title(f"🔍 Detailansicht — {motif_data.get('label', 'Figur')}")
        self.geometry("850x700")
        self.minsize(500, 400)
        self.motif_data = motif_data
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._pan_start = None

        header = ctk.CTkFrame(self, fg_color=THEME["bg_card"], height=48, corner_radius=0)
        header.pack(fill="x", padx=0, pady=(0, 4))
        lbl_title = ctk.CTkLabel(
            header,
            text=f"✨ {motif_data.get('label', 'Motiv')} | Bounding Box: {motif_data.get('bbox', [])}",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=THEME["text_accent"]
        )
        lbl_title.pack(side="left", padx=14, pady=8)

        btn_zoom_in = ctk.CTkButton(header, text="➕ Zoom +", width=80, fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"], command=lambda: self._step_zoom(1.25))
        btn_zoom_in.pack(side="right", padx=4, pady=8)
        btn_zoom_out = ctk.CTkButton(header, text="➖ Zoom -", width=80, fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"], command=lambda: self._step_zoom(0.80))
        btn_zoom_out.pack(side="right", padx=4, pady=8)
        btn_zoom_reset = ctk.CTkButton(header, text="🔄 100%", width=70, fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"], command=self._reset_zoom)
        btn_zoom_reset.pack(side="right", padx=4, pady=8)

        self.canvas = tk.Canvas(self, bg="#0f172a", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=10, pady=6)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)
        self.canvas.bind("<Button-5>", self._on_mousewheel)
        self.canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<B1-Motion>", self._on_pan_drag)
        self.canvas.bind("<Configure>", lambda e: self._redraw())
        self.after(60, self._redraw)

    def _step_zoom(self, factor):
        self.zoom = max(0.2, min(8.0, self.zoom * factor))
        self._redraw()

    def _reset_zoom(self):
        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self._redraw()

    def _on_mousewheel(self, event):
        delta = getattr(event, "delta", 0)
        factor = 1.15 if delta > 0 or getattr(event, "num", 0) == 4 else 0.85
        self._step_zoom(factor)

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._pan_start:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self.pan_x += dx
            self.pan_y += dy
            self._pan_start = (event.x, event.y)
            self._redraw()

    def _redraw(self):
        cutout = self.motif_data.get("cutout")
        if cutout is None:
            return
        cw = max(100, self.canvas.winfo_width())
        ch = max(100, self.canvas.winfo_height())
        iw, ih = cutout.size

        base_scale = min(cw / max(1, iw), ch / max(1, ih)) * 0.90
        scale = base_scale * self.zoom
        disp_w = max(1, int(iw * scale))
        disp_h = max(1, int(ih * scale))

        resized = cutout.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        cx = cw // 2 + self.pan_x
        cy = ch // 2 + self.pan_y
        self.canvas.create_image(cx, cy, image=self._tk_img, anchor="center")


class MotifCard(ctk.CTkFrame):
    """
    Vertically stacked, zoomable individual motif card for the human-in-the-loop verification pipeline.
    """
    def __init__(self, master, motif_data: Dict[str, Any], index: int, app: Any):
        super().__init__(master, fg_color=THEME["bg_card_inner"], corner_radius=10, border_width=1, border_color=THEME["border_subtle"])
        self.motif_data = motif_data
        self.index = index
        self.app = app
        self.zoom = 1.0
        self._displayed_tk_img = None
        self._setup_ui()

    def _setup_ui(self):
        # Header Row
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=10, pady=(8, 4))

        m_id = self.motif_data.get("id", self.index)
        m_label = self.motif_data.get("label", f"Motiv #{m_id + 1}")
        area_pct = self.motif_data.get("area_pct", 0.0)
        bw, bh = self.motif_data.get("bbox", (0, 0, 0, 0))[2], self.motif_data.get("bbox", (0, 0, 0, 0))[3]

        lbl_id = ctk.CTkLabel(
            header,
            text=f"🏷️ #{m_id+1}: {m_label}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_accent"]
        )
        lbl_id.pack(side="left")

        lbl_dims = ctk.CTkLabel(
            header,
            text=f"📐 {bw}×{bh} px ({area_pct:.1f}% Fläche)",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_secondary"]
        )
        lbl_dims.pack(side="left", padx=10)

        btn_del = ctk.CTkButton(
            header, text="🗑️", width=28, height=24,
            fg_color="#ef4444", hover_color="#dc2626", text_color="#ffffff",
            command=self._on_delete
        )
        btn_del.pack(side="right")

        # Content Split Frame
        content_frame = ctk.CTkFrame(self, fg_color="transparent")
        content_frame.pack(fill="x", padx=10, pady=(0, 8))
        content_frame.grid_columnconfigure(0, weight=0)
        content_frame.grid_columnconfigure(1, weight=1)

        # Left Column: Image Canvas & Zoom Controls
        img_col = ctk.CTkFrame(content_frame, fg_color=THEME["bg_card"], corner_radius=8, border_width=1, border_color=THEME["border_subtle"])
        img_col.grid(row=0, column=0, padx=(0, 8), pady=4, sticky="ns")

        self.img_label = ctk.CTkLabel(img_col, text="", width=150, height=150, corner_radius=6)
        self.img_label.pack(padx=6, pady=6)
        self.img_label.bind("<Button-1>", lambda e: self._open_modal_preview())

        # Zoom bar
        zoom_bar = ctk.CTkFrame(img_col, fg_color="transparent")
        zoom_bar.pack(fill="x", padx=4, pady=(0, 6))

        ctk.CTkButton(zoom_bar, text="➕", width=28, height=22, font=ctk.CTkFont(size=10), fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"], command=lambda: self._step_zoom(1.25)).pack(side="left", padx=1)
        ctk.CTkButton(zoom_bar, text="➖", width=28, height=22, font=ctk.CTkFont(size=10), fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"], command=lambda: self._step_zoom(0.80)).pack(side="left", padx=1)
        ctk.CTkButton(zoom_bar, text="100%", width=40, height=22, font=ctk.CTkFont(size=10), fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"], command=self._reset_zoom).pack(side="left", padx=1)
        ctk.CTkButton(zoom_bar, text="🔍", width=28, height=22, font=ctk.CTkFont(size=10), fg_color=THEME["primary"], hover_color=THEME["primary_hover"], text_color="#ffffff", command=self._open_modal_preview).pack(side="right", padx=1)

        # Right Column: AI Suggestion, SKOS Verification & Actions
        right_col = ctk.CTkFrame(content_frame, fg_color="transparent")
        right_col.grid(row=0, column=1, padx=(4, 0), pady=2, sticky="nsew")
        right_col.grid_columnconfigure(0, weight=1)

        ai_label, ai_conf, ai_skos = self._predict_ai_motif()

        # AI Proposal Badge
        ai_card = ctk.CTkFrame(right_col, fg_color=THEME["bg_card"], corner_radius=6, border_width=1, border_color=THEME["border_subtle"])
        ai_card.pack(fill="x", pady=(0, 6))

        skos_local = ai_skos.local_id if ai_skos else "N/A"
        lbl_ai = ctk.CTkLabel(
            ai_card,
            text=f"✨ KI-Vorschlag: {ai_label} ({ai_conf*100:.1f}% Konfidenz)\n🏛️ SKOS: {ai_skos.pref_label if ai_skos else ai_label} | ID: {skos_local}",
            font=ctk.CTkFont(size=11),
            justify="left",
            text_color=THEME["text_primary"]
        )
        lbl_ai.pack(side="left", padx=8, pady=6)

        btn_accept_ai = ctk.CTkButton(
            ai_card, text="✅ Übernehmen", width=95, height=26,
            fg_color="#7c3aed", hover_color="#6d28d9", text_color="#ffffff",
            font=ctk.CTkFont(size=10, weight="bold"),
            command=lambda: self._set_label(ai_label)
        )
        btn_accept_ai.pack(side="right", padx=6, pady=6)

        # SKOS Concept Selector with Filter Entry
        sel_frame = ctk.CTkFrame(right_col, fg_color="transparent")
        sel_frame.pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(sel_frame, text="Verifiziertes SKOS-Label:", font=ctk.CTkFont(size=11, weight="bold"), text_color=THEME["text_primary"]).pack(anchor="w")

        # Fast filter input
        self.entry_filter = ctk.CTkEntry(sel_frame, placeholder_text="🔍 Filter im SKOS-Thesaurus (z.B. Delphin, Dionysos)...", height=28, font=ctk.CTkFont(size=11))
        self.entry_filter.pack(fill="x", pady=(2, 3))
        self.entry_filter.bind("<KeyRelease>", self._on_filter_labels)

        # Dropdown
        self.all_skos_labels = skos_client.get_all_labels_for_selection()
        initial_val = ai_label if ai_label in self.all_skos_labels else (self.all_skos_labels[0] if self.all_skos_labels else "Unbekannt")
        self.combo_label = ctk.CTkComboBox(sel_frame, values=self.all_skos_labels[:150], height=28, font=ctk.CTkFont(size=11))
        self.combo_label.set(initial_val)
        self.combo_label.pack(fill="x")

        # Action Buttons Row
        btn_row = ctk.CTkFrame(right_col, fg_color="transparent")
        btn_row.pack(fill="x", pady=(4, 0))

        btn_save = ctk.CTkButton(
            btn_row, text="💾 Verifizieren & Speichern",
            fg_color="#10b981", hover_color="#059669", text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"), height=28,
            command=self._on_save
        )
        btn_save.pack(side="left", fill="x", expand=True, padx=(0, 4))

        btn_skos_edit = ctk.CTkButton(
            btn_row, text="🌳 SKOS-Editor",
            fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"],
            font=ctk.CTkFont(size=11), height=28, width=120,
            command=self._open_in_skos_editor
        )
        btn_skos_edit.pack(side="right", padx=(4, 0))

        self._update_image_display()

    def _predict_ai_motif(self) -> Tuple[str, float, Optional[Any]]:
        cutout = self.motif_data.get("cutout")
        if cutout is None:
            return "Unbekannt", 0.0, None
        try:
            bg_img = Image.new("RGB", cutout.size, (0, 0, 0))
            bg_img.paste(cutout, mask=cutout.split()[3])
            pred = classifier.predict_scene(bg_img)
            label = pred["predicted_scene"]
            conf = pred["scene_confidence"]
            skos = skos_client.get_concept_by_label(label)
            return label, conf, skos
        except Exception:
            return "Motiv", 0.5, None

    def _step_zoom(self, factor):
        self.zoom = max(0.5, min(4.0, self.zoom * factor))
        self._update_image_display()

    def _reset_zoom(self):
        self.zoom = 1.0
        self._update_image_display()

    def _update_image_display(self):
        cutout = self.motif_data.get("cutout")
        if cutout is None:
            self.img_label.configure(image=None, text="Kein Bild")
            return
        try:
            iw, ih = cutout.size
            base_scale = min(140 / max(1, iw), 140 / max(1, ih))
            scale = base_scale * self.zoom
            disp_w = max(1, int(iw * scale))
            disp_h = max(1, int(ih * scale))

            resized = cutout.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
            ctk_img = ctk.CTkImage(light_image=resized, dark_image=resized, size=(disp_w, disp_h))
            self.img_label.configure(image=ctk_img, text="")
            self.img_label._displayed_image_ref = ctk_img
        except Exception as e:
            print(f"Error displaying motif card image: {e}")

    def _open_modal_preview(self):
        ZoomableMotifPreviewDialog(self.winfo_toplevel(), self.motif_data)

    def _set_label(self, label: str):
        self.combo_label.set(label)

    def _on_filter_labels(self, event):
        q = self.entry_filter.get().strip().lower()
        if not q:
            filtered = self.all_skos_labels[:150]
        else:
            filtered = [l for l in self.all_skos_labels if q in l.lower()][:150]
        if filtered:
            self.combo_label.configure(values=filtered)
            self.combo_label.set(filtered[0])

    def refresh_skos_dropdown(self):
        self.all_skos_labels = skos_client.get_all_labels_for_selection()
        cur = self.combo_label.get()
        self.combo_label.configure(values=self.all_skos_labels[:150])
        if cur in self.all_skos_labels:
            self.combo_label.set(cur)

    def _on_save(self):
        v_label = self.combo_label.get().strip()
        skos = skos_client.get_concept_by_label(v_label)
        entry = {
            "image_path": self.app.sam_current_path,
            "img_w": self.app.sam_current_pil.width if self.app.sam_current_pil else 500,
            "img_h": self.app.sam_current_pil.height if self.app.sam_current_pil else 500,
            "bbox": self.motif_data.get("bbox", (0, 0, 0, 0)),
            "polygon": self.motif_data.get("polygon", []),
            "area": self.motif_data.get("area", 0),
            "cutout": self.motif_data.get("cutout"),
            "label": v_label,
            "skos_uri": skos.uri if skos else "",
            "confidence": 1.0,
            "category": skos.category if skos else "Ikonographie"
        }
        self.app.training_dataset.append(entry)
        self.app.show_toast(f"Figur #{self.index+1} als '{v_label}' gespeichert (Pool: {len(self.app.training_dataset)})", "success")

    def _open_in_skos_editor(self):
        v_label = self.combo_label.get().strip()
        self.app.open_skos_editor_with_term(v_label)

    def _on_delete(self):
        if 0 <= self.index < len(self.app.sam_detected_motifs):
            self.app.sam_detected_motifs.pop(self.index)
            self.app._render_all_motif_cards()
            if self.app.sam_current_pil is not None:
                overlay = sam_segmenter.draw_segmentation_overlay(self.app.sam_current_pil, self.app.sam_detected_motifs)
                self.app.sam_overlay_pil = overlay
                self.app._redraw_sam_canvas()
            self.app.show_toast("Motiv aus der Liste entfernt.", "info")


class SKOSEditorFrame(ctk.CTkFrame):
    """
    Complete SKOS Vocabulary & Thesaurus Editor with RDFLib Graph binding.
    """
    def __init__(self, master, app: Any, is_popup: bool = False):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self.is_popup = is_popup
        self.current_concept_uri = None
        self._setup_ui()
        self._refresh_concepts_table()

    def _setup_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Top Control Toolbar
        top_bar = ctk.CTkFrame(self, height=50, fg_color=THEME["bg_card"], corner_radius=8, border_width=1, border_color=THEME["border_subtle"])
        top_bar.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 4))

        ttl_name = os.path.basename(skos_client.ttl_file_path or "heritage_assets.ttl")
        self.lbl_vocab_info = ctk.CTkLabel(
            top_bar,
            text=f"🌳 SKOS-Thesaurus: {ttl_name} ({len(skos_client.concepts_by_uri)} Begriffe)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=THEME["text_accent"]
        )
        self.lbl_vocab_info.pack(side="left", padx=12, pady=8)

        if not self.is_popup:
            btn_popout = ctk.CTkButton(
                top_bar, text="🪟 In separatem Fenster (Multi-Screen)",
                font=ctk.CTkFont(size=11, weight="bold"),
                fg_color=THEME["primary"], hover_color=THEME["primary_hover"], text_color="#ffffff",
                command=self.app._open_multi_screen_skos_window
            )
            btn_popout.pack(side="right", padx=6, pady=8)

        btn_save_ttl = ctk.CTkButton(
            top_bar, text="💾 In TTL sichern", width=120,
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#10b981", hover_color="#059669", text_color="#ffffff",
            command=self._save_to_ttl_file
        )
        btn_save_ttl.pack(side="right", padx=4, pady=8)

        btn_load_ttl = ctk.CTkButton(
            top_bar, text="📂 TTL laden", width=105,
            font=ctk.CTkFont(size=11),
            fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"],
            command=self._load_ttl_file
        )
        btn_load_ttl.pack(side="right", padx=4, pady=8)

        btn_fetch_gh = ctk.CTkButton(
            top_bar, text="🌐 GitHub Sync", width=105,
            font=ctk.CTkFont(size=11),
            fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"],
            command=self._sync_from_github
        )
        btn_fetch_gh.pack(side="right", padx=4, pady=8)

        # Main Split Body
        main_split = ctk.CTkFrame(self, fg_color="transparent")
        main_split.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)
        main_split.grid_rowconfigure(0, weight=1)
        main_split.grid_columnconfigure(0, weight=5)
        main_split.grid_columnconfigure(1, weight=6)

        # Left Column: Search & Table
        left_box = ctk.CTkFrame(main_split, fg_color=THEME["bg_card"], corner_radius=8, border_width=1, border_color=THEME["border_subtle"])
        left_box.grid(row=0, column=0, sticky="nsew", padx=(2, 4), pady=2)
        left_box.grid_rowconfigure(2, weight=1)
        left_box.grid_columnconfigure(0, weight=1)

        search_bar = ctk.CTkFrame(left_box, fg_color="transparent")
        search_bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        search_bar.grid_columnconfigure(0, weight=1)

        self.entry_search = ctk.CTkEntry(search_bar, placeholder_text="🔍 Suche nach Begriff, Maler, Gottheit oder URI...", font=ctk.CTkFont(size=11))
        self.entry_search.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.entry_search.bind("<KeyRelease>", self._on_search_change)

        self.combo_cat_filter = ctk.CTkComboBox(
            search_bar,
            values=["Alle Kategorien", "Mythologie / Figur", "Gefäßform", "Maler / Werkstatt", "Ornament", "Technik / Ware", "Kulturgut"],
            width=140, font=ctk.CTkFont(size=11), command=lambda v: self._refresh_concepts_table()
        )
        self.combo_cat_filter.set("Alle Kategorien")
        self.combo_cat_filter.grid(row=0, column=1)

        tbl_container = ctk.CTkFrame(left_box, fg_color="transparent")
        tbl_container.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        tbl_container.grid_rowconfigure(0, weight=1)
        tbl_container.grid_columnconfigure(0, weight=1)

        scroll_y = ttk.Scrollbar(tbl_container, orient="vertical")
        scroll_y.grid(row=0, column=1, sticky="ns")

        self.tree_concepts = ttk.Treeview(
            tbl_container,
            columns=("DE", "EN", "Kategorie", "URI"),
            show="headings",
            yscrollcommand=scroll_y.set
        )
        scroll_y.config(command=self.tree_concepts.yview)

        self.tree_concepts.heading("DE", text="Deutsches Label (prefLabel)")
        self.tree_concepts.heading("EN", text="English Label")
        self.tree_concepts.heading("Kategorie", text="Kategorie")
        self.tree_concepts.heading("URI", text="SKOS URI")

        self.tree_concepts.column("DE", width=160)
        self.tree_concepts.column("EN", width=130)
        self.tree_concepts.column("Kategorie", width=110)
        self.tree_concepts.column("URI", width=180)

        self.tree_concepts.grid(row=0, column=0, sticky="nsew")
        self.tree_concepts.bind("<<TreeviewSelect>>", self._on_select_concept_from_table)

        # Right Column: Concept Edit Mask
        right_box = ctk.CTkFrame(main_split, fg_color=THEME["bg_card"], corner_radius=8, border_width=1, border_color=THEME["border_subtle"])
        right_box.grid(row=0, column=1, sticky="nsew", padx=(4, 2), pady=2)
        right_box.grid_rowconfigure(6, weight=1)
        right_box.grid_columnconfigure(0, weight=1)

        lbl_mask_title = ctk.CTkLabel(
            right_box, text="✏️ Begriff bearbeiten & einpflegen",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=THEME["text_accent"]
        )
        lbl_mask_title.grid(row=0, column=0, sticky="w", padx=12, pady=(10, 6))

        form = ctk.CTkFrame(right_box, fg_color="transparent")
        form.grid(row=1, column=0, sticky="nsew", padx=12, pady=4)
        form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(form, text="Label (DE) *:", font=ctk.CTkFont(size=11, weight="bold")).grid(row=0, column=0, sticky="w", pady=3)
        self.ent_label_de = ctk.CTkEntry(form, placeholder_text="z. B. Delphinreiter mit Lanze", font=ctk.CTkFont(size=11))
        self.ent_label_de.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=3)

        ctk.CTkLabel(form, text="Label (EN):", font=ctk.CTkFont(size=11)).grid(row=1, column=0, sticky="w", pady=3)
        self.ent_label_en = ctk.CTkEntry(form, placeholder_text="z. B. Dolphin Rider with Spear", font=ctk.CTkFont(size=11))
        self.ent_label_en.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=3)

        ctk.CTkLabel(form, text="Synonyme (altLabel):", font=ctk.CTkFont(size=11)).grid(row=2, column=0, sticky="w", pady=3)
        self.ent_alt_labels = ctk.CTkEntry(form, placeholder_text="Kommagetrennt, z. B. Krieger auf Delphin, Delphinkrieger", font=ctk.CTkFont(size=11))
        self.ent_alt_labels.grid(row=2, column=1, sticky="ew", padx=(6, 0), pady=3)

        ctk.CTkLabel(form, text="Kategorie:", font=ctk.CTkFont(size=11)).grid(row=3, column=0, sticky="w", pady=3)
        self.combo_category = ctk.CTkComboBox(
            form, values=["Mythologie / Figur", "Gefäßform", "Maler / Werkstatt", "Ornament", "Technik / Ware", "Kulturgut"],
            font=ctk.CTkFont(size=11)
        )
        self.combo_category.set("Mythologie / Figur")
        self.combo_category.grid(row=3, column=1, sticky="ew", padx=(6, 0), pady=3)

        ctk.CTkLabel(form, text="Oberbegriff (broader):", font=ctk.CTkFont(size=11)).grid(row=4, column=0, sticky="w", pady=3)
        self.ent_broader = ctk.CTkEntry(form, placeholder_text="URI oder Label des Oberbegriffs", font=ctk.CTkFont(size=11))
        self.ent_broader.grid(row=4, column=1, sticky="ew", padx=(6, 0), pady=3)

        ctk.CTkLabel(form, text="Definition / Notiz:", font=ctk.CTkFont(size=11)).grid(row=5, column=0, sticky="nw", pady=3)
        self.txt_definition = ctk.CTkTextbox(form, height=90, font=ctk.CTkFont(size=11))
        self.txt_definition.grid(row=5, column=1, sticky="ew", padx=(6, 0), pady=3)

        ctk.CTkLabel(form, text="Kanonische URI:", font=ctk.CTkFont(size=11)).grid(row=6, column=0, sticky="w", pady=3)
        self.ent_uri = ctk.CTkEntry(form, placeholder_text="Wird automatisch generiert (oder manuell anpassen)", font=ctk.CTkFont(size=11))
        self.ent_uri.grid(row=6, column=1, sticky="ew", padx=(6, 0), pady=3)

        mask_btn_bar = ctk.CTkFrame(right_box, fg_color="transparent")
        mask_btn_bar.grid(row=2, column=0, sticky="ew", padx=12, pady=(10, 12))

        btn_new = ctk.CTkButton(
            mask_btn_bar, text="➕ Neuer Begriff", width=110, height=32,
            font=ctk.CTkFont(size=11), fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"],
            command=self._clear_form
        )
        btn_new.pack(side="left", padx=(0, 6))

        btn_save_concept = ctk.CTkButton(
            mask_btn_bar, text="💾 Begriff speichern & einpflegen", height=32,
            font=ctk.CTkFont(size=11, weight="bold"), fg_color="#10b981", hover_color="#059669", text_color="#ffffff",
            command=self._save_concept_from_form
        )
        btn_save_concept.pack(side="left", fill="x", expand=True, padx=4)

        btn_del_concept = ctk.CTkButton(
            mask_btn_bar, text="🗑️ Löschen", width=85, height=32,
            font=ctk.CTkFont(size=11), fg_color="#ef4444", hover_color="#dc2626", text_color="#ffffff",
            command=self._delete_concept_from_form
        )
        btn_del_concept.pack(side="right", padx=(6, 0))

    def _on_search_change(self, event=None):
        self._refresh_concepts_table()

    def _refresh_concepts_table(self):
        query = self.entry_search.get().strip()
        cat = self.combo_cat_filter.get()
        results = skos_client.search_concepts(query=query, category=cat, limit=300)

        for item in self.tree_concepts.get_children():
            self.tree_concepts.delete(item)

        for c in results:
            self.tree_concepts.insert("", "end", values=(
                c.pref_label,
                "",
                c.category or "Kulturgut",
                c.uri
            ))

        ttl_name = os.path.basename(skos_client.ttl_file_path or "heritage_assets.ttl")
        self.lbl_vocab_info.configure(text=f"🌳 SKOS-Thesaurus: {ttl_name} ({len(skos_client.concepts_by_uri)} Begriffe)")

    def _on_select_concept_from_table(self, event):
        selected = self.tree_concepts.selection()
        if not selected:
            return
        vals = self.tree_concepts.item(selected[0], "values")
        if not vals:
            return
        uri = vals[3]
        concept = skos_client.get_concept_by_label(uri) or skos_client.get_concept_by_label(vals[0])
        if concept:
            self.load_concept_into_form(concept)

    def load_concept_into_form(self, concept: Any):
        self.current_concept_uri = concept.uri
        self.ent_label_de.delete(0, "end")
        self.ent_label_de.insert(0, concept.pref_label or "")

        self.ent_label_en.delete(0, "end")
        self.ent_alt_labels.delete(0, "end")
        self.ent_alt_labels.insert(0, ", ".join(concept.alt_labels))

        self.combo_category.set(concept.category or "Mythologie / Figur")

        self.ent_broader.delete(0, "end")
        self.ent_broader.insert(0, concept.broader_uri or concept.broader_label or "")

        self.txt_definition.delete("1.0", "end")
        self.txt_definition.insert("1.0", concept.definition or "")

        self.ent_uri.delete(0, "end")
        self.ent_uri.insert(0, concept.uri or "")

    def _clear_form(self):
        self.current_concept_uri = None
        self.ent_label_de.delete(0, "end")
        self.ent_label_en.delete(0, "end")
        self.ent_alt_labels.delete(0, "end")
        self.combo_category.set("Mythologie / Figur")
        self.ent_broader.delete(0, "end")
        self.txt_definition.delete("1.0", "end")
        self.ent_uri.delete(0, "end")

    def _save_concept_from_form(self):
        de_label = self.ent_label_de.get().strip()
        if not de_label:
            self.app.show_toast("Bitte gib mindestens ein deutsches Label ein.", "warning")
            return
        en_label = self.ent_label_en.get().strip()
        alt_str = self.ent_alt_labels.get().strip()
        alts = [a.strip() for a in alt_str.split(",") if a.strip()]
        cat = self.combo_category.get()
        broader = self.ent_broader.get().strip()
        definition = self.txt_definition.get("1.0", "end").strip()
        uri = self.ent_uri.get().strip()

        concept = skos_client.add_or_update_concept(
            pref_label_de=de_label,
            pref_label_en=en_label,
            alt_labels=alts,
            broader_uri=broader,
            definition=definition,
            category=cat,
            custom_uri=uri
        )
        self.current_concept_uri = concept.uri
        self.ent_uri.delete(0, "end")
        self.ent_uri.insert(0, concept.uri)

        self._refresh_concepts_table()
        self.app.show_toast(f"Begriff '{de_label}' erfolgreich im SKOS-Thesaurus gespeichert!", "success")

    def _delete_concept_from_form(self):
        uri = self.ent_uri.get().strip() or self.current_concept_uri
        if not uri:
            self.app.show_toast("Kein Begriff zum Löschen ausgewählt.", "warning")
            return
        skos_client.delete_concept(uri)
        self._clear_form()
        self._refresh_concepts_table()
        self.app.show_toast("Begriff aus dem SKOS-Thesaurus gelöscht.", "info")

    def _save_to_ttl_file(self):
        save_path = filedialog.asksaveasfilename(
            title="SKOS-Thesaurus in Turtle (.ttl) Datei speichern",
            defaultextension=".ttl",
            filetypes=[("RDF Turtle (.ttl)", "*.ttl")],
            initialfile=os.path.basename(skos_client.ttl_file_path or "heritage_assets.ttl")
        )
        if save_path:
            saved = skos_client.save_to_ttl(save_path)
            self._refresh_concepts_table()
            self.app.show_toast(f"Vokabular in '{os.path.basename(saved)}' gesichert!", "success")

    def _load_ttl_file(self):
        open_path = filedialog.askopenfilename(
            title="Lokale SKOS Turtle (.ttl) Datei laden",
            filetypes=[("RDF Turtle (.ttl)", "*.ttl"), ("All files", "*.*")]
        )
        if open_path:
            count = skos_client.load_local_ttl(open_path)
            self._refresh_concepts_table()
            self.app.show_toast(f"Erfolgreich {count} Begriffe aus '{os.path.basename(open_path)}' geladen!", "success")

    def _sync_from_github(self):
        def _task():
            import urllib.request
            url = "https://raw.githubusercontent.com/bcdhbonn/hector-editor-skos/main/vocabularies/heritage_assets/heritage_assets.ttl"
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dest = os.path.join(base_dir, "data", "vocabularies", "heritage_assets.ttl")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                urllib.request.urlretrieve(url, dest)
                count = skos_client.load_local_ttl(dest)
                def _done():
                    self._refresh_concepts_table()
                    self.app.show_toast(f"GitHub Sync: {count} Begriffe aktualisiert!", "success")
                self.after(0, _done)
            except Exception as e:
                def _err():
                    self.app.show_toast(f"Fehler beim GitHub Sync: {e}", "error")
                self.after(0, _err)
        threading.Thread(target=_task, daemon=True).start()


class SKOSEditorWindow(ctk.CTkToplevel):
    """Floating multi-screen window for the SKOS Editor."""
    def __init__(self, master, app: Any):
        super().__init__(master)
        self.title("🌳 HECTOR SKOS-Editor (BCDH Bonn) — Multi-Screen")
        self.geometry("1180x780")
        self.minsize(800, 550)
        self.editor = SKOSEditorFrame(self, app=app, is_popup=True)
        self.editor.pack(fill="both", expand=True, padx=10, pady=10)


class MotifVerificationWindow(ctk.CTkToplevel):
    """Floating multi-screen window for the full Motif Verification Studio."""
    def __init__(self, master, app: Any):
        super().__init__(master)
        self.title("🏷️ Motiv-Prüfungs-Studio (BCDH Bonn) — Multi-Screen")
        self.geometry("1180x880")
        self.minsize(700, 500)
        self.app = app
        
        top_bar = ctk.CTkFrame(self, height=48, fg_color=THEME["bg_card"], corner_radius=8, border_width=1, border_color=THEME["border_subtle"])
        top_bar.pack(fill="x", padx=10, pady=(10, 4))
        self.lbl_count = ctk.CTkLabel(
            top_bar,
            text=f"✨ Motiv-Prüfung ({len(app.sam_detected_motifs)} Figuren erkannt)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=THEME["text_accent"]
        )
        self.lbl_count.pack(side="left", padx=12, pady=8)

        btn_skos = ctk.CTkButton(
            top_bar, text="🌳 SKOS-Editor öffnen",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"],
            command=app._open_multi_screen_skos_window
        )
        btn_skos.pack(side="right", padx=6, pady=8)

        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self._cards: List[MotifCard] = []
        self._populate()

    def _populate(self):
        for w in self.scroll_frame.winfo_children():
            w.destroy()
        self._cards.clear()
        for idx, m in enumerate(self.app.sam_detected_motifs):
            card = MotifCard(self.scroll_frame, m, idx, self.app)
            card.pack(fill="x", pady=6, padx=4)
            self._cards.append(card)
        self.lbl_count.configure(text=f"✨ Motiv-Prüfung ({len(self.app.sam_detected_motifs)} Figuren erkannt)")

    def refresh_dropdowns(self):
        for c in self._cards:
            c.refresh_skos_dropdown()


class LODVasesApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🏺 LODVases — 3D-GLB Studio, SAM-Segmentierung & SKOS-Thesaurus (BCDH Bonn)")
        self.geometry("1560x940")
        self.minsize(1200, 780)
        try:
            self.after(100, lambda: self.state("zoomed"))
        except Exception:
            pass

        # App state
        self.current_glb_bytes = None
        self.current_glb_mesh = None
        self.current_glb_proxy = None
        self.current_glb_texture = None
        self.current_glb_uv = None
        self.current_glb_profile = None
        self.current_glb_filename = None
        self.current_glb_info = None
        self.current_texture_layer = "base_color"
        self.unroll_axis = "Y"
        self.unroll_flip = False
        self.unroll_azimuth_offset = 270.0
        self.unroll_cx_offset = 0.0
        self.unroll_cz_offset = 0.0
        self.unroll_z_min = 0.12
        self.unroll_z_max = 0.76
        self.unroll_projection_mode = "cylindrical"
        self.current_abrollung_pil = None
        self.current_abrollung_zoom = 1.0
        self._full_abrollung_backup = None
        self._abrollung_crop_start = None
        self._abrollung_crop_rect = None
        self._unroll_debounce_job = None
        self._3d_render_mode = "texture"
        self._is_dragging = False

        self.sam_current_pil = None
        self.sam_current_path = None
        self.sam_detected_motifs = []
        self.training_dataset = []

        self.single_pil = None
        self.single_crop = None
        self.catalog_items = []
        self._orbit_debounce_job = None
        self._3d_zoom_scale = 1.0
        self._mouse_drag_start_x = 0
        self._mouse_drag_start_y = 0
        self._is_rendering_frame = False
        self.sam_image_paths = []
        self.sam_current_idx = 0
        self.current_view_name = "3d"
        sam_segmenter.set_backend("meta_sam")
        self._motif_cards = []
        self._motif_window_toplevel = None
        self._skos_window_toplevel = None
        skos_client.register_on_change_listener(self._on_skos_vocab_changed)

        # Setup GUI Grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        self._create_sidebar()
        self._create_main_content()

        # Initialize with 3D Studio Tab
        self.select_view("3d")

    def _display_image_on_label(self, img_pil: Optional[Image.Image], label_widget: Optional[ctk.CTkLabel], max_size: Tuple[int, int]):
        if label_widget is None:
            return
        if img_pil is None:
            label_widget.configure(image=None, text="Kein Bild")
            return
        try:
            iw, ih = img_pil.size
            mw, mh = max_size
            scale = min(mw / max(1, iw), mh / max(1, ih))
            disp_w = max(1, int(iw * scale))
            disp_h = max(1, int(ih * scale))

            ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(disp_w, disp_h))
            label_widget.configure(image=ctk_img, text="")
            label_widget._displayed_image_ref = ctk_img
        except Exception as e:
            print(f"Failed to display image on label: {e}")

    def _create_sidebar(self):
        self.sidebar_frame = ctk.CTkFrame(self, width=255, corner_radius=0, fg_color=THEME["bg_sidebar"])
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(10, weight=1)

        # Brand Header with pill badge
        brand_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        brand_frame.grid(row=0, column=0, padx=16, pady=(20, 8), sticky="w")

        self.logo_label = ctk.CTkLabel(
            brand_frame,
            text="🏺 LODVases",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#0284c7"
        )
        self.logo_label.pack(anchor="w")

        self.sub_label = ctk.CTkLabel(
            brand_frame,
            text="BCDH Universität Bonn • Vision AI\nHECTOR SKOS Thesaurus",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_secondary"],
            justify="left"
        )
        self.sub_label.pack(anchor="w", pady=(2, 0))

        # Nav Section 1: 3D & Form
        lbl_sec1 = ctk.CTkLabel(
            self.sidebar_frame, text="3D & FORM-ANALYSE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_muted"]
        )
        lbl_sec1.grid(row=1, column=0, padx=18, pady=(10, 3), sticky="w")

        self.btn_3d = ctk.CTkButton(
            self.sidebar_frame, text="🏺 3D-Vasen-Studio (.glb)", height=36,
            anchor="w", font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
            command=lambda: self.select_view("3d")
        )
        self.btn_3d.grid(row=2, column=0, padx=12, pady=2, sticky="ew")

        # Nav Section 2: KI-Segmentierung
        lbl_sec2 = ctk.CTkLabel(
            self.sidebar_frame, text="KI-SEGMENTIERUNG & SZENEN",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_muted"]
        )
        lbl_sec2.grid(row=3, column=0, padx=18, pady=(10, 3), sticky="w")

        self.btn_sam = ctk.CTkButton(
            self.sidebar_frame, text="✨ SAM Figuren-Studio", height=36,
            anchor="w", font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
            command=lambda: self.select_view("sam")
        )
        self.btn_sam.grid(row=4, column=0, padx=12, pady=2, sticky="ew")

        self.btn_single = ctk.CTkButton(
            self.sidebar_frame, text="🔍 2D-Einzelbild & Gottheiten", height=36,
            anchor="w", font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
            command=lambda: self.select_view("single")
        )
        self.btn_single.grid(row=5, column=0, padx=12, pady=2, sticky="ew")

        # Nav Section 3: Linked Data & Export
        lbl_sec3 = ctk.CTkLabel(
            self.sidebar_frame, text="KORPUS & LINKED DATA",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_muted"]
        )
        lbl_sec3.grid(row=6, column=0, padx=18, pady=(10, 3), sticky="w")

        self.btn_batch = ctk.CTkButton(
            self.sidebar_frame, text="📂 Batch-Scanner & Katalog", height=36,
            anchor="w", font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
            command=lambda: self.select_view("batch")
        )
        self.btn_batch.grid(row=7, column=0, padx=12, pady=2, sticky="ew")

        self.btn_skos = ctk.CTkButton(
            self.sidebar_frame, text="🌳 HECTOR SKOS-Explorer", height=36,
            anchor="w", font=ctk.CTkFont(size=12, weight="bold"),
            corner_radius=8,
            command=lambda: self.select_view("skos")
        )
        self.btn_skos.grid(row=8, column=0, padx=12, pady=2, sticky="ew")

        # Theme Switch
        theme_frame = ctk.CTkFrame(self.sidebar_frame, fg_color="transparent")
        theme_frame.grid(row=9, column=0, padx=14, pady=(14, 0), sticky="ew")

        self.switch_dark = ctk.CTkSwitch(
            theme_frame, text="🌙 Dunkler Modus",
            command=self._on_toggle_theme,
            onvalue=1, offvalue=0,
            font=ctk.CTkFont(size=11)
        )
        self.switch_dark.pack(anchor="w", padx=4)

        # Hardware Profile Card (Elevated Card at Bottom)
        self.hw_frame = ctk.CTkFrame(self.sidebar_frame, corner_radius=10, fg_color=THEME["bg_card_inner"], border_width=1, border_color=THEME["border_subtle"])
        self.hw_frame.grid(row=11, column=0, padx=12, pady=16, sticky="sew")

        device_name = sam_segmenter.device.upper()
        gpu_str = "NVIDIA RTX 4080 (CUDA)" if "CUDA" in device_name else device_name
        self.hw_label = ctk.CTkLabel(
            self.hw_frame,
            text=f"🟢 {gpu_str}\n✨ SAM (ViT-B) & YOLOv8 aktiv\n🏛️ HECTOR SKOS bereit",
            font=ctk.CTkFont(size=11),
            text_color=THEME["success"],
            justify="left"
        )
        self.hw_label.pack(padx=12, pady=10, anchor="w")

    def _on_toggle_theme(self):
        is_dark = self.switch_dark.get() == 1
        mode_str = "Dark" if is_dark else "Light"
        ctk.set_appearance_mode(mode_str)
        canvas_bg = "#0f172a" if is_dark else "#ffffff"
        canvas_border = "#334155" if is_dark else "#e2e8f0"

        if hasattr(self, "canvas_sam"):
            self.canvas_sam.configure(bg=canvas_bg, highlightbackground=canvas_border)
        if hasattr(self, "canvas_abrollung"):
            self.canvas_abrollung.configure(bg=canvas_bg, highlightbackground=canvas_border)

        self.switch_dark.configure(text="🌙 Dunkler Modus" if is_dark else "☀️ Heller Modus")

        if hasattr(self, "_redraw_sam_canvas"):
            self._redraw_sam_canvas()
        if hasattr(self, "_on_abrollung_zoom_change"):
            self._on_abrollung_zoom_change()

    def _create_main_content(self):
        self.main_container = ctk.CTkFrame(self, corner_radius=0, fg_color=THEME["bg_app"])
        self.main_container.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_container.grid_rowconfigure(1, weight=1)
        self.main_container.grid_columnconfigure(0, weight=1)

        # Toast Banner in Row 0 (non-blocking notification bar)
        self.toast_banner = ctk.CTkFrame(self.main_container, fg_color="transparent", height=0)
        self.toast_banner.grid(row=0, column=0, sticky="ew", padx=10, pady=(0, 4))

        # Viewport frame in Row 1
        self.views_container = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.views_container.grid(row=1, column=0, sticky="nsew")
        self.views_container.grid_rowconfigure(0, weight=1)
        self.views_container.grid_columnconfigure(0, weight=1)

        # Build 5 Views inside views_container
        self.view_3d = self._build_3d_view(self.views_container)
        self.view_sam = self._build_sam_view(self.views_container)
        self.view_single = self._build_single_view(self.views_container)
        self.view_batch = self._build_batch_view(self.views_container)
        self.view_skos = self._build_skos_view(self.views_container)

    def show_toast(self, message: str, toast_type: str = "info", duration_ms: int = 3500):
        """Displays a sleek, non-blocking toast notification banner at the top of the workspace."""
        if self._toast_job:
            self.after_cancel(self._toast_job)
            self._toast_job = None

        for child in self.toast_banner.winfo_children():
            child.destroy()

        configs = {
            "success": {
                "icon": "✅",
                "fg": ("#ecfdf5", "#064e3b"),
                "border": ("#a7f3d0", "#059669"),
                "text": ("#065f46", "#6ee7b7"),
            },
            "info": {
                "icon": "ℹ️",
                "fg": ("#eff6ff", "#1e3a8a"),
                "border": ("#bfdbfe", "#2563eb"),
                "text": ("#1e40af", "#93c5fd"),
            },
            "warning": {
                "icon": "⚠️",
                "fg": ("#fffbeb", "#78350f"),
                "border": ("#fde68a", "#d97706"),
                "text": ("#92400e", "#fcd34d"),
            },
            "error": {
                "icon": "❌",
                "fg": ("#fef2f2", "#7f1d1d"),
                "border": ("#fecaca", "#dc2626"),
                "text": ("#991b1b", "#fca5a5"),
            }
        }
        cfg = configs.get(toast_type, configs["info"])

        pill = ctk.CTkFrame(
            self.toast_banner,
            fg_color=cfg["fg"],
            border_width=1,
            border_color=cfg["border"],
            corner_radius=18,
            height=32
        )
        pill.pack(side="top", pady=2)

        lbl = ctk.CTkLabel(
            pill,
            text=f"{cfg['icon']}  {message}",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=cfg["text"]
        )
        lbl.pack(side="left", padx=(14, 10), pady=4)

        btn_close = ctk.CTkButton(
            pill,
            text="✕",
            width=20,
            height=20,
            fg_color="transparent",
            hover_color=cfg["border"],
            text_color=cfg["text"],
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._dismiss_toast
        )
        btn_close.pack(side="right", padx=(0, 8), pady=4)

        self._toast_job = self.after(duration_ms, self._dismiss_toast)

    def _dismiss_toast(self):
        if self._toast_job:
            self.after_cancel(self._toast_job)
            self._toast_job = None
        for child in self.toast_banner.winfo_children():
            child.destroy()

    def select_view(self, name: str):
        self.current_view_name = name
        for v in [self.view_3d, self.view_sam, self.view_single, self.view_batch, self.view_skos]:
            v.grid_forget()

        # Inactive styling: clean transparent background, secondary text, subtle hover
        nav_buttons = [
            ("3d", self.btn_3d, THEME["primary"], THEME["primary_hover"]),
            ("sam", self.btn_sam, THEME["ai_purple"], THEME["ai_purple_hover"]),
            ("single", self.btn_single, THEME["primary"], THEME["primary_hover"]),
            ("batch", self.btn_batch, THEME["primary"], THEME["primary_hover"]),
            ("skos", self.btn_skos, THEME["primary"], THEME["primary_hover"]),
        ]

        for v_name, btn, active_col, hover_col in nav_buttons:
            if v_name == name:
                btn.configure(
                    fg_color=active_col,
                    text_color="#ffffff",
                    hover_color=hover_col
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=THEME["text_secondary"],
                    hover_color=THEME["nav_hover"]
                )

        if name == "3d":
            self.view_3d.grid(row=0, column=0, sticky="nsew")
        elif name == "sam":
            self.view_sam.grid(row=0, column=0, sticky="nsew")
            if not self.sam_image_paths and self.sam_current_pil is None:
                self._load_default_sam_image()
        elif name == "single":
            self.view_single.grid(row=0, column=0, sticky="nsew")
            if self.single_pil is None:
                self._load_default_single_image()
        elif name == "batch":
            self.view_batch.grid(row=0, column=0, sticky="nsew")
        elif name == "skos":
            self.view_skos.grid(row=0, column=0, sticky="nsew")

    # ---------------------------------------------------------
    # VIEW 1: 3D-Vasen-Studio (.glb)
    # ---------------------------------------------------------
    def _build_3d_view(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # Header Bar
        top_bar = ctk.CTkFrame(frame, height=50, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        lbl = ctk.CTkLabel(top_bar, text="🏺 3D-Vasen & Abrollungs-Studio: 360°-Bildfries (Abrollung) & Profilanalyse", font=ctk.CTkFont(size=14, weight="bold"), text_color=("#0f172a", "#f8fafc"))
        lbl.pack(side="left", padx=15, pady=8)

        btn_load = ctk.CTkButton(top_bar, text="📂 Eigene .glb Datei öffnen", command=self._open_glb_file, width=170, fg_color=("#0284c7", "#334155"), hover_color=("#0369a1", "#475569"))
        btn_load.pack(side="right", padx=10, pady=8)

        self.combo_preset_3d = ctk.CTkComboBox(
            top_bar, values=["Aryballos (8K Scan)", "Pelike (8K Scan)", "Lekythos", "Psykter", "Kylix"],
            command=self._load_sample_3d, width=175
        )
        self.combo_preset_3d.set("3D-Modell wählen...")
        self.combo_preset_3d.pack(side="right", padx=5, pady=8)

        # Left Column: 360° Abrollung / Bildfries Viewport
        col_left = ctk.CTkFrame(frame, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        col_left.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        col_left.grid_rowconfigure(4, weight=1)
        col_left.grid_columnconfigure(0, weight=1)

        self.lbl_abrollung_title = ctk.CTkLabel(col_left, text="📜 360° GigaMesh-Abrollung (Bildfries)", font=ctk.CTkFont(size=13, weight="bold"), text_color=("#0284c7", "#38bdf8"))
        self.lbl_abrollung_title.grid(row=0, column=0, padx=10, pady=(6, 2), sticky="w")

        # GigaMesh Calibration Control Panel
        calib_frame = ctk.CTkFrame(col_left, fg_color=("#f1f5f9", "#0f172a"), corner_radius=8)
        calib_frame.grid(row=1, column=0, padx=10, pady=4, sticky="ew")

        # Row 0: Decoration Type: Bildfeld vs Umlaufender Bildfries
        r_mode = ctk.CTkFrame(calib_frame, fg_color="transparent")
        r_mode.pack(fill="x", padx=8, pady=(4, 2))

        ctk.CTkLabel(r_mode, text="🏺 Dekor-Typ:", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#d97706", "#fbbf24")).pack(side="left", padx=(0, 6))
        self.seg_proj_mode = ctk.CTkSegmentedButton(
            r_mode, values=["🏛️ Bildfeld (Seite A / Seite B)", "📜 Umlaufender 360°-Fries"],
            command=self._on_proj_mode_change
        )
        self.seg_proj_mode.set("🏛️ Bildfeld (Seite A / Seite B)")
        self.seg_proj_mode.pack(side="left", fill="x", expand=True, padx=2)

        # Row 0b: Panel Selector Buttons (Side A / Side B / Left / Right)
        self.panel_bar = ctk.CTkFrame(calib_frame, fg_color="transparent")
        self.panel_bar.pack(fill="x", padx=8, pady=(2, 2))

        ctk.CTkLabel(self.panel_bar, text="Ansicht:", font=ctk.CTkFont(size=11), text_color=("#0f172a", "#f8fafc")).pack(side="left", padx=(0, 4))
        self.btn_side_a = ctk.CTkButton(self.panel_bar, text="🏛️ Seite A (Vorderseite)", width=130, height=24, fg_color="#0284c7", command=lambda: self._set_panel_angle(90.0))
        self.btn_side_a.pack(side="left", padx=2)
        self.btn_side_b = ctk.CTkButton(self.panel_bar, text="🏛️ Seite B (Rückseite)", width=130, height=24, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._set_panel_angle(270.0))
        self.btn_side_b.pack(side="left", padx=2)
        self.btn_side_l = ctk.CTkButton(self.panel_bar, text="Henkel L (0°)", width=85, height=24, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._set_panel_angle(0.0))
        self.btn_side_l.pack(side="left", padx=2)
        self.btn_side_r = ctk.CTkButton(self.panel_bar, text="Henkel R (180°)", width=85, height=24, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._set_panel_angle(180.0))
        self.btn_side_r.pack(side="left", padx=2)

        # Row 1: Rotational Axis & Flip orientation
        r0 = ctk.CTkFrame(calib_frame, fg_color="transparent")
        r0.pack(fill="x", padx=8, pady=(2, 2))

        ctk.CTkLabel(r0, text="📐 Achse:", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#0284c7", "#38bdf8")).pack(side="left", padx=(0, 4))
        self.combo_axis = ctk.CTkComboBox(
            r0, values=["Y-Achse (Standard GLTF)", "Z-Achse (CAD / Blender)", "X-Achse", "🤖 Auto-Erkennung"],
            command=self._on_axis_change, width=175
        )
        self.combo_axis.set("Y-Achse (Standard GLTF)")
        self.combo_axis.pack(side="left", padx=2)

        self.btn_flip = ctk.CTkButton(
            r0, text="↕️ Oben/Unten", width=100, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"),
            command=self._on_flip_orientation
        )
        self.btn_flip.pack(side="left", padx=4)

        # Row 2: Azimuth Start-Winkel Offset Slider
        r1 = ctk.CTkFrame(calib_frame, fg_color="transparent")
        r1.pack(fill="x", padx=8, pady=(2, 2))

        ctk.CTkLabel(r1, text="🔄 Start-Winkel (Naht):", font=ctk.CTkFont(size=11), text_color=("#0f172a", "#f8fafc")).pack(side="left", padx=(0, 6))
        self.slider_azimuth = ctk.CTkSlider(r1, from_=0, to=360, number_of_steps=72, command=self._on_azimuth_change)
        self.slider_azimuth.set(0)
        self.slider_azimuth.pack(side="left", fill="x", expand=True, padx=6)

        self.lbl_azimuth_val = ctk.CTkLabel(r1, text="0°", font=ctk.CTkFont(size=11, weight="bold"), width=35, text_color=("#0f172a", "#f8fafc"))
        self.lbl_azimuth_val.pack(side="right", padx=(2, 0))

        # Row 2: Frieze Vertical Zone Framing (Z-Min & Z-Max)
        r2 = ctk.CTkFrame(calib_frame, fg_color="transparent")
        r2.pack(fill="x", padx=8, pady=(2, 2))

        ctk.CTkLabel(r2, text="🏛️ Fries-Höhe:", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#d97706", "#fbbf24")).pack(side="left", padx=(0, 4))
        self.combo_frieze_zone = ctk.CTkComboBox(
            r2, values=[
                "🏛️ Figuren-Bildfries (Bauchzone)",
                "🏺 Gesamtes Gefäß (0% - 100%)",
                "🪞 Oberteil & Mündung (55% - 100%)",
                "🧱 Unterteil & Standring (0% - 35%)"
            ],
            command=self._on_frieze_zone_change, width=195
        )
        self.combo_frieze_zone.set("🏛️ Figuren-Bildfries (Bauchzone)")
        self.combo_frieze_zone.pack(side="left", padx=2)

        ctk.CTkLabel(r2, text="Unten:", font=ctk.CTkFont(size=10), text_color=("#0f172a", "#f8fafc")).pack(side="left", padx=(6, 2))
        self.slider_z_min = ctk.CTkSlider(r2, from_=0.0, to=0.8, number_of_steps=40, width=70, command=self._on_z_min_change)
        self.slider_z_min.set(0.12)
        self.slider_z_min.pack(side="left", padx=2)

        self.lbl_z_min_val = ctk.CTkLabel(r2, text="12%", font=ctk.CTkFont(size=10, weight="bold"), width=28, text_color=("#0f172a", "#f8fafc"))
        self.lbl_z_min_val.pack(side="left", padx=(1, 4))

        ctk.CTkLabel(r2, text="Oben:", font=ctk.CTkFont(size=10), text_color=("#0f172a", "#f8fafc")).pack(side="left", padx=(4, 2))
        self.slider_z_max = ctk.CTkSlider(r2, from_=0.2, to=1.0, number_of_steps=40, width=70, command=self._on_z_max_change)
        self.slider_z_max.set(0.76)
        self.slider_z_max.pack(side="left", padx=2)

        self.lbl_z_max_val = ctk.CTkLabel(r2, text="76%", font=ctk.CTkFont(size=10, weight="bold"), width=28, text_color=("#0f172a", "#f8fafc"))
        self.lbl_z_max_val.pack(side="left", padx=(1, 2))

        # Row 3: Rotation Axis Center Offsets (X & Z)
        r3 = ctk.CTkFrame(calib_frame, fg_color="transparent")
        r3.pack(fill="x", padx=8, pady=(2, 4))

        ctk.CTkLabel(r3, text="🎯 Zentrum: ΔX:", font=ctk.CTkFont(size=10), text_color=("#0f172a", "#f8fafc")).pack(side="left", padx=(0, 2))
        self.slider_cx = ctk.CTkSlider(r3, from_=-0.02, to=0.02, number_of_steps=40, width=95, command=self._on_cx_change)
        self.slider_cx.set(0.0)
        self.slider_cx.pack(side="left", padx=2)

        ctk.CTkLabel(r3, text="ΔZ:", font=ctk.CTkFont(size=10), text_color=("#0f172a", "#f8fafc")).pack(side="left", padx=(6, 2))
        self.slider_cz = ctk.CTkSlider(r3, from_=-0.02, to=0.02, number_of_steps=40, width=95, command=self._on_cz_change)
        self.slider_cz.set(0.0)
        self.slider_cz.pack(side="left", padx=2)

        btn_reset_calib = ctk.CTkButton(r3, text="↺ Reset", width=55, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=self._on_reset_calib)
        btn_reset_calib.pack(side="right", padx=2)

        # Texture Layer Selector (BaseColor vs Normal vs Roughness)
        self.seg_texture_layer = ctk.CTkSegmentedButton(
            col_left, values=["🖼️ Foto-Farbe", "🔮 Normal-Map", "✨ Rauheit / AO"],
            command=self._on_texture_layer_change
        )
        self.seg_texture_layer.set("🖼️ Foto-Farbe")
        self.seg_texture_layer.grid(row=2, column=0, padx=10, pady=4, sticky="ew")

        # Zoom & Control bar for Abrollung
        ctrl_abrollung = ctk.CTkFrame(col_left, fg_color="transparent")
        ctrl_abrollung.grid(row=3, column=0, padx=10, pady=2, sticky="ew")

        ctk.CTkLabel(ctrl_abrollung, text="🔍 Zoom:", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#0284c7", "#38bdf8")).pack(side="left", padx=(0, 4))
        self.btn_zoom_out = ctk.CTkButton(ctrl_abrollung, text="➖", width=28, height=24, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=self._zoom_out)
        self.btn_zoom_out.pack(side="left", padx=2)

        self.slider_abrollung_zoom = ctk.CTkSlider(ctrl_abrollung, from_=1.0, to=4.0, number_of_steps=30, width=120, command=self._on_abrollung_zoom_change)
        self.slider_abrollung_zoom.set(1.0)
        self.slider_abrollung_zoom.pack(side="left", fill="x", expand=True, padx=4)

        self.btn_zoom_in = ctk.CTkButton(ctrl_abrollung, text="➕", width=28, height=24, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=self._zoom_in)
        self.btn_zoom_in.pack(side="left", padx=2)

        self.btn_zoom_reset = ctk.CTkButton(ctrl_abrollung, text="↺ 100%", width=55, height=24, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=self._reset_zoom_and_pan)
        self.btn_zoom_reset.pack(side="left", padx=4)

        self.lbl_zoom_val = ctk.CTkLabel(ctrl_abrollung, text="100%", font=ctk.CTkFont(size=11, weight="bold"), width=40, text_color=("#0f172a", "#f8fafc"))
        self.lbl_zoom_val.pack(side="left", padx=2)

        btn_save_abrollung = ctk.CTkButton(ctrl_abrollung, text="💾 Speichern", width=85, height=24, fg_color="#0284c7", hover_color="#0369a1", text_color="#ffffff", command=self._save_abrollung_image)
        btn_save_abrollung.pack(side="right", padx=5)

        # Canvas for Abrollung
        self.canvas_abrollung = tk.Canvas(col_left, bg="#ffffff", highlightthickness=1, highlightbackground="#e2e8f0", cursor="crosshair")
        self.canvas_abrollung.grid(row=4, column=0, padx=10, pady=4, sticky="nsew")
        self.canvas_abrollung.bind("<Button-1>", self._on_abrollung_mouse_down)
        self.canvas_abrollung.bind("<B1-Motion>", self._on_abrollung_mouse_drag)
        self.canvas_abrollung.bind("<ButtonRelease-1>", self._on_abrollung_mouse_up)
        self.canvas_abrollung.bind("<MouseWheel>", self._on_canvas_mousewheel)
        self.canvas_abrollung.bind("<Button-3>", self._on_pan_start)
        self.canvas_abrollung.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas_abrollung.bind("<Button-2>", self._on_pan_start)
        self.canvas_abrollung.bind("<B2-Motion>", self._on_pan_drag)

        # Crop & Refinement Toolbar
        crop_bar = ctk.CTkFrame(col_left, fg_color=("#f1f5f9", "#0f172a"), corner_radius=6)
        crop_bar.grid(row=5, column=0, padx=10, pady=2, sticky="ew")

        self.btn_crop_selection = ctk.CTkButton(
            crop_bar, text="✂️ Markierten Bildfries freistellen", width=190, height=26,
            fg_color="#0284c7", hover_color="#0369a1", text_color="#ffffff", font=ctk.CTkFont(size=11, weight="bold"),
            command=self._crop_to_selection
        )
        self.btn_crop_selection.pack(side="left", padx=4, pady=3)

        self.btn_reset_crop = ctk.CTkButton(
            crop_bar, text="↺ Gesamtes Bild", width=110, height=26,
            fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), font=ctk.CTkFont(size=11),
            command=self._reset_crop
        )
        self.btn_reset_crop.pack(side="left", padx=4, pady=3)

        self.lbl_crop_status = ctk.CTkLabel(crop_bar, text="💡 Mit Maus Rechteck über Bild ziehen", font=ctk.CTkFont(size=10, slant="italic"), text_color=("#64748b", "#94a3b8"))
        self.lbl_crop_status.pack(side="right", padx=6)

        btn_snap_sam = ctk.CTkButton(
            col_left, text="🚀 Diese 360°-Abrollung an SAM übergeben (Alle Figuren segmentieren)",
            fg_color="#7c3aed", hover_color="#6d28d9", text_color="#ffffff", font=ctk.CTkFont(size=12, weight="bold"),
            command=self._send_abrollung_to_sam
        )
        btn_snap_sam.grid(row=6, column=0, padx=10, pady=8, sticky="ew")

        # Right Column: Profile Analysis & Section Drawing with Visual Frieze Marker
        col_right = ctk.CTkFrame(frame, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        col_right.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        col_right.grid_rowconfigure(3, weight=1)
        col_right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(col_right, text="📐 Archäologische 3D-Profilanalyse & Interaktiver Fries-Marker", font=ctk.CTkFont(size=13, weight="bold"), text_color=("#e11d48", "#f43f5e")).grid(row=0, column=0, padx=10, pady=(6, 2), sticky="w")

        # Interactive Frieze Marker Toolbar
        marker_bar = ctk.CTkFrame(col_right, fg_color=("#f1f5f9", "#0f172a"), corner_radius=6)
        marker_bar.grid(row=1, column=0, padx=10, pady=2, sticky="ew")

        btn_mark_belly = ctk.CTkButton(
            marker_bar, text="🖌️ Figuren-Fries markieren (Bauch)", width=175, height=26,
            fg_color="#0284c7", hover_color="#0369a1", text_color="#ffffff", font=ctk.CTkFont(size=11, weight="bold"),
            command=lambda: self._on_frieze_zone_change("🏛️ Figuren-Bildfries (Bauchzone)")
        )
        btn_mark_belly.pack(side="left", padx=4, pady=3)

        btn_mark_all = ctk.CTkButton(
            marker_bar, text="🏺 Ganzes Gefäß", width=110, height=26,
            fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), font=ctk.CTkFont(size=11),
            command=lambda: self._on_frieze_zone_change("🏺 Gesamtes Gefäß (0% - 100%)")
        )
        btn_mark_all.pack(side="left", padx=4, pady=3)

        ctk.CTkLabel(marker_bar, text="👆 Klick/Ziehen auf Gefäß markiert Fries", font=ctk.CTkFont(size=10, slant="italic"), text_color=("#64748b", "#94a3b8")).pack(side="right", padx=6)

        self.lbl_profile_metrics = ctk.CTkLabel(
            col_right, text="Berechne 3D-Maße...", font=ctk.CTkFont(size=11),
            justify="left", fg_color=("#f1f5f9", "#0f172a"), text_color=("#0f172a", "#f8fafc"), corner_radius=6, padx=8, pady=4
        )
        self.lbl_profile_metrics.grid(row=2, column=0, padx=10, pady=2, sticky="ew")

        self.canvas_section = ctk.CTkLabel(col_right, text="Erzeuge Schnittzeichnung...", fg_color=("#f1f5f9", "#0f172a"), text_color=("#0f172a", "#f8fafc"), corner_radius=8, cursor="hand2")
        self.canvas_section.grid(row=3, column=0, padx=10, pady=4, sticky="nsew")
        self.canvas_section.bind("<Button-1>", self._on_profile_mouse_click)
        self.canvas_section.bind("<B1-Motion>", self._on_profile_mouse_drag)

        export_bar_cva = ctk.CTkFrame(col_right, fg_color="transparent")
        export_bar_cva.grid(row=4, column=0, padx=10, pady=4, sticky="ew")

        btn_cva_svg = ctk.CTkButton(
            export_bar_cva, text="📄 CVA (SVG 1:1)", width=110, height=28,
            fg_color="#0284c7", hover_color="#0369a1", text_color="#ffffff", font=ctk.CTkFont(size=11, weight="bold"),
            command=self._export_cva_svg
        )
        btn_cva_svg.pack(side="left", padx=2, expand=True, fill="x")

        btn_cva_pdf = ctk.CTkButton(
            export_bar_cva, text="📑 CVA (PDF Druck)", width=110, height=28,
            fg_color="#7c3aed", hover_color="#6d28d9", text_color="#ffffff", font=ctk.CTkFont(size=11, weight="bold"),
            command=self._export_cva_pdf
        )
        btn_cva_pdf.pack(side="left", padx=2, expand=True, fill="x")

        btn_cva_png = ctk.CTkButton(
            export_bar_cva, text="🖼️ CVA (PNG 300 DPI)", width=110, height=28,
            fg_color="#059669", hover_color="#047857", text_color="#ffffff", font=ctk.CTkFont(size=11, weight="bold"),
            command=self._export_cva_png
        )
        btn_cva_png.pack(side="left", padx=2, expand=True, fill="x")

        btn_export_3d = ctk.CTkButton(
            col_right, text="📥 3D-LOD Datensatz exportieren (JSON-LD)", height=28,
            fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=self._export_3d_lod
        )
        btn_export_3d.grid(row=5, column=0, padx=10, pady=(2, 8), sticky="ew")

        return frame

    def _on_profile_mouse_click(self, event):
        # Determine click height on vessel section (approx. 10% to 90% in label coordinates)
        lbl_h = max(1, self.canvas_section.winfo_height())
        # Y is 0 at top, lbl_h at bottom
        rel_y = np.clip(1.0 - (event.y / lbl_h), 0.0, 1.0)
        # If click is above midpoint of current selection, adjust top; else adjust bottom
        mid = (self.unroll_z_min + self.unroll_z_max) / 2.0
        if rel_y >= mid:
            self.unroll_z_max = max(self.unroll_z_min + 0.05, min(1.0, rel_y))
            self.slider_z_max.set(self.unroll_z_max)
        else:
            self.unroll_z_min = min(self.unroll_z_max - 0.05, max(0.0, rel_y))
            self.slider_z_min.set(self.unroll_z_min)
        self._apply_live_z_crop()

    def _on_profile_mouse_drag(self, event):
        lbl_h = max(1, self.canvas_section.winfo_height())
        rel_y = np.clip(1.0 - (event.y / lbl_h), 0.0, 1.0)
        mid = (self.unroll_z_min + self.unroll_z_max) / 2.0
        if rel_y >= mid:
            self.unroll_z_max = max(self.unroll_z_min + 0.05, min(1.0, rel_y))
            self.slider_z_max.set(self.unroll_z_max)
        else:
            self.unroll_z_min = min(self.unroll_z_max - 0.05, max(0.0, rel_y))
            self.slider_z_min.set(self.unroll_z_min)
        self._apply_live_z_crop()

    def _canvas_coords_to_image_rect(self, x0: float, y0: float, x1: float, y1: float) -> Optional[Tuple[int, int, int, int]]:
        if self.current_abrollung_pil is None:
            return None
        canvas_w = max(1, self.canvas_abrollung.winfo_width())
        canvas_h = max(1, self.canvas_abrollung.winfo_height())
        zoom = float(self.slider_abrollung_zoom.get())
        iw, ih = self.current_abrollung_pil.size

        base_scale = min(canvas_w / max(1, iw), canvas_h / max(1, ih))
        disp_w = max(1, int(iw * base_scale * zoom))
        disp_h = max(1, int(ih * base_scale * zoom))

        cx = canvas_w // 2 + int(getattr(self, "_abrollung_pan_x", 0))
        cy = canvas_h // 2 + int(getattr(self, "_abrollung_pan_y", 0))

        img_x0 = cx - disp_w / 2.0
        img_y0 = cy - disp_h / 2.0

        min_x, max_x = min(x0, x1), max(x0, x1)
        min_y, max_y = min(y0, y1), max(y0, y1)

        u0 = (min_x - img_x0) / float(disp_w)
        v0 = (min_y - img_y0) / float(disp_h)
        u1 = (max_x - img_x0) / float(disp_w)
        v1 = (max_y - img_y0) / float(disp_h)

        px0 = int(np.clip(u0, 0.0, 1.0) * iw)
        py0 = int(np.clip(v0, 0.0, 1.0) * ih)
        px1 = int(np.clip(u1, 0.0, 1.0) * iw)
        py1 = int(np.clip(v1, 0.0, 1.0) * ih)

        return (px0, py0, px1, py1)

    def _on_abrollung_mouse_down(self, event):
        self._abrollung_crop_start = (event.x, event.y)
        self._abrollung_crop_rect = None
        self.canvas_abrollung.delete("crop_rect")

    def _on_abrollung_mouse_drag(self, event):
        if self._abrollung_crop_start is None or self.current_abrollung_pil is None:
            return
        x0, y0 = self._abrollung_crop_start
        x1, y1 = event.x, event.y

        # Draw live selection box on canvas
        self.canvas_abrollung.delete("crop_rect")
        self.canvas_abrollung.create_rectangle(x0, y0, x1, y1, outline="#38bdf8", width=2, tags="crop_rect")

        rect = self._canvas_coords_to_image_rect(x0, y0, x1, y1)
        if rect:
            px0, py0, px1, py1 = rect
            cw = px1 - px0
            ch = py1 - py0
            self._abrollung_crop_rect = rect
            self.lbl_crop_status.configure(text=f"📐 Auswahl: {cw} × {ch} px | Klick 'Freistellen'")

    def _on_abrollung_mouse_up(self, event):
        if self._abrollung_crop_rect:
            px0, py0, px1, py1 = self._abrollung_crop_rect
            if px1 - px0 > 8 and py1 - py0 > 8:
                self.btn_crop_selection.configure(fg_color="#10b981", hover_color="#059669")
            else:
                self._abrollung_crop_rect = None
                self.canvas_abrollung.delete("crop_rect")
                self.lbl_crop_status.configure(text="💡 Mit Maus Rechteck über Bild ziehen")
                self.btn_crop_selection.configure(fg_color="#0284c7", hover_color="#0369a1")

    def _crop_to_selection(self):
        if self.current_abrollung_pil is None or self._abrollung_crop_rect is None:
            return
        px0, py0, px1, py1 = self._abrollung_crop_rect
        if px1 - px0 <= 8 or py1 - py0 <= 8:
            return

        if self._full_abrollung_backup is None:
            self._full_abrollung_backup = self.current_abrollung_pil.copy()

        box = (px0, py0, px1, py1)
        self.current_abrollung_pil = self.current_abrollung_pil.crop(box)
        self._abrollung_crop_rect = None
        self.canvas_abrollung.delete("crop_rect")
        self.lbl_crop_status.configure(text=f"✂️ Freigestellt: {box[2]-box[0]} × {box[3]-box[1]} px")
        self.btn_crop_selection.configure(fg_color="#0284c7", hover_color="#0369a1")
        self._on_abrollung_zoom_change()

    def _reset_crop(self):
        if self._full_abrollung_backup is not None:
            self.current_abrollung_pil = self._full_abrollung_backup.copy()
            self._full_abrollung_backup = None
        else:
            self._apply_live_z_crop()
        self._abrollung_crop_rect = None
        self.canvas_abrollung.delete("crop_rect")
        self.lbl_crop_status.configure(text="💡 Mit Maus Rechteck über Bild ziehen")
        self.btn_crop_selection.configure(fg_color="#0284c7", hover_color="#0369a1")
        self._on_abrollung_zoom_change()


    def _zoom_in(self):
        cur = float(self.slider_abrollung_zoom.get())
        new_val = min(4.0, cur + 0.25)
        self.slider_abrollung_zoom.set(new_val)
        self._on_abrollung_zoom_change()

    def _zoom_out(self):
        cur = float(self.slider_abrollung_zoom.get())
        new_val = max(1.0, cur - 0.25)
        self.slider_abrollung_zoom.set(new_val)
        self._on_abrollung_zoom_change()

    def _reset_zoom_and_pan(self):
        self.slider_abrollung_zoom.set(1.0)
        self._abrollung_pan_x = 0
        self._abrollung_pan_y = 0
        self._on_abrollung_zoom_change()

    def _on_canvas_mousewheel(self, event):
        delta = getattr(event, "delta", 0)
        cur = float(self.slider_abrollung_zoom.get())
        if delta > 0:
            new_val = min(4.0, cur + 0.20)
        else:
            new_val = max(1.0, cur - 0.20)
        self.slider_abrollung_zoom.set(new_val)
        self._on_abrollung_zoom_change()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y)

    def _on_pan_drag(self, event):
        if self._pan_start is not None:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self._abrollung_pan_x = getattr(self, "_abrollung_pan_x", 0) + dx
            self._abrollung_pan_y = getattr(self, "_abrollung_pan_y", 0) + dy
            self._pan_start = (event.x, event.y)
            self._on_abrollung_zoom_change()

    def _on_abrollung_zoom_change(self, val=None):
        canvas_w = max(450, self.canvas_abrollung.winfo_width())
        canvas_h = max(380, self.canvas_abrollung.winfo_height())
        is_dark = hasattr(self, "switch_dark") and self.switch_dark.get() == 1

        if self.current_abrollung_pil is None:
            self.canvas_abrollung.delete("all")
            accent = "#38bdf8" if is_dark else "#0284c7"
            muted = "#94a3b8" if is_dark else "#64748b"
            card_bg = "#1e293b" if is_dark else "#f8fafc"
            border_col = "#334155" if is_dark else "#e2e8f0"

            rw, rh = 440, 160
            rx0, ry0 = (canvas_w - rw) // 2, (canvas_h - rh) // 2
            rx1, ry1 = rx0 + rw, ry0 + rh
            self.canvas_abrollung.create_rectangle(rx0, ry0, rx1, ry1, fill=card_bg, outline=border_col, width=1, tags="empty_card")

            self.canvas_abrollung.create_text(
                canvas_w // 2, canvas_h // 2 - 35,
                text="🏺", font=("Segoe UI Emoji", 34), fill=accent, tags="empty_card"
            )
            self.canvas_abrollung.create_text(
                canvas_w // 2, canvas_h // 2 + 10,
                text="Keine 3D-Abrollung aktiv", font=("Segoe UI", 13, "bold"), fill=accent, tags="empty_card"
            )
            self.canvas_abrollung.create_text(
                canvas_w // 2, canvas_h // 2 + 38,
                text="Wähle oben ein Gefäß-Preset (z. B. Amphora 38) oder öffne eine .glb-Datei",
                font=("Segoe UI", 10), fill=muted, tags="empty_card"
            )
            return

        self.canvas_abrollung.delete("empty_card")
        zoom = float(self.slider_abrollung_zoom.get())
        if hasattr(self, "lbl_zoom_val"):
            self.lbl_zoom_val.configure(text=f"{int(zoom * 100)}%")

        img = self.current_abrollung_pil
        iw, ih = img.size

        # Fit image to fill the canvas viewport generously
        base_scale = min(canvas_w / max(1, iw), canvas_h / max(1, ih))
        disp_w = max(10, int(iw * base_scale * zoom))
        disp_h = max(10, int(ih * base_scale * zoom))

        disp_img = img.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self._tk_abrollung_img = ImageTk.PhotoImage(disp_img)

        cx = canvas_w // 2 + int(getattr(self, "_abrollung_pan_x", 0))
        cy = canvas_h // 2 + int(getattr(self, "_abrollung_pan_y", 0))

        self.canvas_abrollung.delete("abrollung_img")
        self._abrollung_img_id = self.canvas_abrollung.create_image(cx, cy, image=self._tk_abrollung_img, anchor="center", tags="abrollung_img")
        self.canvas_abrollung.tag_lower("abrollung_img")

    def _apply_azimuth_and_z_crop(self):
        base = getattr(self, "_base_360_master", None)
        if base is not None:
            w, h = base.size
            shift_px = int((self.unroll_azimuth_offset / 360.0) * w)
            if shift_px != 0:
                rolled_arr = np.roll(np.array(base), -shift_px, axis=1)
                full_img = Image.fromarray(rolled_arr)
            else:
                full_img = base

            if self.unroll_projection_mode == "bildfeld":
                # Bildfeld focuses on the front 180° field centered on the azimuth angle
                half_w = w // 4
                full_img = full_img.crop((half_w, 0, 3 * half_w, h))

            self._master_full_image = full_img
            self._apply_live_z_crop()

    def _apply_live_z_crop(self):
        master = getattr(self, "_master_full_image", None)
        if master is not None:
            iw, ih = master.size
            # Top is y0, Bottom is y1 (top of vase is 1.0, bottom is 0.0)
            y0 = max(0, min(ih - 4, int((1.0 - self.unroll_z_max) * ih)))
            y1 = max(y0 + 4, min(ih, int((1.0 - self.unroll_z_min) * ih)))
            self.current_abrollung_pil = master.crop((0, y0, iw, y1))
            self._on_abrollung_zoom_change()
            self._schedule_section_drawing_fast()

            # Update percentage indicator labels
            if hasattr(self, "lbl_z_min_val"):
                self.lbl_z_min_val.configure(text=f"{int(round(self.unroll_z_min * 100))}%")
            if hasattr(self, "lbl_z_max_val"):
                self.lbl_z_max_val.configure(text=f"{int(round(self.unroll_z_max * 100))}%")

            if self.current_abrollung_pil is not None:
                w, h = self.current_abrollung_pil.size
                layer_label = "Normal-Map" if self.current_texture_layer == "normal" else ("Rauheit/AO" if self.current_texture_layer == "roughness" else "Fotografische Farbe")
                mode_label = "🏛️ Bildfeld" if self.unroll_projection_mode == "bildfeld" else f"📜 360°-Abrollung ({self.unroll_axis}-Achse)"
                self.lbl_abrollung_title.configure(text=f"{mode_label}: {layer_label} ({w} × {h} px | {int(round(self.unroll_z_min * 100))}% - {int(round(self.unroll_z_max * 100))}% Höhe)")

    def _schedule_section_drawing_fast(self):
        if getattr(self, "_sec_debounce_job", None) is not None:
            self.after_cancel(self._sec_debounce_job)
        self._sec_debounce_job = self.after(35, self._update_section_drawing_fast)

    def _update_section_drawing_fast(self):
        self._sec_debounce_job = None
        if self.current_glb_profile is not None:
            buf = profile_analyzer.generate_section_drawing(
                self.current_glb_profile,
                frieze_z_min_pct=self.unroll_z_min,
                frieze_z_max_pct=self.unroll_z_max,
                dpi=100
            )
            sec_img = Image.open(buf).convert("RGB")
            lbl_sw = max(400, self.canvas_section.winfo_width())
            lbl_sh = max(400, self.canvas_section.winfo_height())
            self._display_image_on_label(sec_img, self.canvas_section, (lbl_sw, lbl_sh))

    def _on_proj_mode_change(self, choice: str):
        if "Bildfeld" in choice:
            self.unroll_projection_mode = "bildfeld"
        else:
            self.unroll_projection_mode = "frieze"
        self._apply_azimuth_and_z_crop()

    def _set_panel_angle(self, angle: float):
        self.unroll_azimuth_offset = angle
        self.slider_azimuth.set(angle)
        self.lbl_azimuth_val.configure(text=f"{int(angle)}°")
        self.btn_side_a.configure(fg_color="#0284c7" if angle == 90.0 else "#334155")
        self.btn_side_b.configure(fg_color="#0284c7" if angle == 270.0 else "#334155")
        self.btn_side_l.configure(fg_color="#0284c7" if angle == 0.0 else "#334155")
        self.btn_side_r.configure(fg_color="#0284c7" if angle == 180.0 else "#334155")
        self._apply_azimuth_and_z_crop()

    def _on_axis_change(self, choice: str):
        if "Y" in choice:
            self.unroll_axis = "Y"
        elif "Z" in choice:
            self.unroll_axis = "Z"
        elif "X" in choice:
            self.unroll_axis = "X"
        else:
            self.unroll_axis = "auto"
        self._schedule_debounced_render(fast_preview=False)

    def _on_flip_orientation(self):
        self.unroll_flip = not self.unroll_flip
        self._schedule_debounced_render(fast_preview=False)

    def _on_azimuth_change(self, val=None):
        self.unroll_azimuth_offset = float(self.slider_azimuth.get())
        self.lbl_azimuth_val.configure(text=f"{int(self.unroll_azimuth_offset)}°")
        self._apply_azimuth_and_z_crop()

    def _on_z_min_change(self, val=None):
        self.unroll_z_min = float(self.slider_z_min.get())
        if self.unroll_z_min >= self.unroll_z_max - 0.02:
            self.unroll_z_min = max(0.0, self.unroll_z_max - 0.02)
            self.slider_z_min.set(self.unroll_z_min)
        self._apply_live_z_crop()

    def _on_z_max_change(self, val=None):
        self.unroll_z_max = float(self.slider_z_max.get())
        if self.unroll_z_max <= self.unroll_z_min + 0.02:
            self.unroll_z_max = min(1.0, self.unroll_z_min + 0.02)
            self.slider_z_max.set(self.unroll_z_max)
        self._apply_live_z_crop()

    def _on_cx_change(self, val=None):
        self.unroll_cx_offset = float(self.slider_cx.get())
        self._schedule_debounced_render(fast_preview=True)

    def _on_cz_change(self, val=None):
        self.unroll_cz_offset = float(self.slider_cz.get())
        self._schedule_debounced_render(fast_preview=True)

    def _on_reset_calib(self):
        self.unroll_cx_offset = 0.0
        self.unroll_cz_offset = 0.0
        self.unroll_azimuth_offset = 0.0
        self.unroll_z_min = 0.05
        self.unroll_z_max = 0.95
        self.slider_cx.set(0.0)
        self.slider_cz.set(0.0)
        self.slider_azimuth.set(0.0)
        self.lbl_azimuth_val.configure(text="0°")
        self.slider_z_min.set(0.05)
        self.slider_z_max.set(0.95)
        self.combo_frieze_zone.set("🏛️ Figuren-Bildfries (Bauchzone)")
        self._schedule_debounced_render(fast_preview=False)

    def _on_frieze_zone_change(self, choice: str):
        if "Gesamtes" in choice:
            self.unroll_z_min = 0.0
            self.unroll_z_max = 1.0
        elif "Oberteil" in choice:
            self.unroll_z_min = 0.55
            self.unroll_z_max = 1.0
        elif "Unterteil" in choice:
            self.unroll_z_min = 0.0
            self.unroll_z_max = 0.35
        else: # Figuren-Bildfries (Bauchzone)
            self.unroll_z_min = 0.08
            self.unroll_z_max = 0.88
        self.slider_z_min.set(self.unroll_z_min)
        self.slider_z_max.set(self.unroll_z_max)
        self._apply_live_z_crop()

    def _on_texture_layer_change(self, choice: str):
        if "Normal" in choice:
            self.current_texture_layer = "normal"
        elif "Rauheit" in choice:
            self.current_texture_layer = "roughness"
        else:
            self.current_texture_layer = "base_color"
        self._schedule_debounced_render(fast_preview=False)

    def _schedule_debounced_render(self, fast_preview=False, delay_ms=100):
        if self._unroll_debounce_job is not None:
            self.after_cancel(self._unroll_debounce_job)
            self._unroll_debounce_job = None
            
        unroll_full = getattr(self, "_unroll_full_job", None)
        if unroll_full is not None:
            self.after_cancel(unroll_full)
            self._unroll_full_job = None

        if fast_preview:
            self._unroll_debounce_job = self.after(30, lambda: self._render_current_abrollung(max_dim=600))
            self._unroll_full_job = self.after(220, lambda: self._render_current_abrollung(max_dim=3600))
        else:
            self._unroll_debounce_job = self.after(delay_ms, lambda: self._render_current_abrollung(max_dim=3600))

    def _render_current_abrollung(self, max_dim: int = 3600):
        if self.current_glb_info is not None:
            # Always render full vessel height (0.0 to 1.0) into base 360 image at Ultra-HD resolution
            base_360 = glb_processor.get_abrollung_image(
                self.current_glb_info,
                layer=self.current_texture_layer,
                axis=self.unroll_axis,
                flip_vertical=self.unroll_flip,
                azimuth_offset_deg=0.0,
                cx_offset=self.unroll_cx_offset,
                cz_offset=self.unroll_cz_offset,
                z_crop=(0.0, 1.0),
                projection_mode="frieze",
                max_dimension=max_dim
            )

            self._base_360_master = base_360
            self._apply_azimuth_and_z_crop()

            if self.current_abrollung_pil is not None:
                w, h = self.current_abrollung_pil.size
                layer_label = "Normal-Map" if self.current_texture_layer == "normal" else ("Rauheit/AO" if self.current_texture_layer == "roughness" else "Fotografische Farbe")
                mode_label = "🏛️ Bildfeld" if self.unroll_projection_mode == "bildfeld" else f"📜 360°-Abrollung ({self.unroll_axis}-Achse)"
                self.lbl_abrollung_title.configure(text=f"{mode_label}: {layer_label} ({w} × {h} px)")

    def _load_sample_3d(self, shape_name: str):
        def _task():
            if "Aryballos" in shape_name:
                ary_path = os.path.join(BASE_DIR, "models", "Aryballos_29_gameready.glb")
                if os.path.exists(ary_path):
                    with open(ary_path, "rb") as f:
                        glb_bytes = f.read()
                    filename = "Aryballos_29_gameready.glb"
                    axis = "Y"
                    azimuth = 270.0
                    z_min = 0.12
                    z_max = 0.76
                    proj_mode = "frieze"
                else:
                    glb_bytes = glb_processor.create_sample_3d_vase("Lekythos")
                    filename = "aryballos_3d.glb"
                    axis = "Y"
                    azimuth = 270.0
                    z_min = 0.12
                    z_max = 0.76
                    proj_mode = "frieze"
            elif "Pelike" in shape_name:
                pel_path = os.path.join(BASE_DIR, "models", "Pelike_574_gameready.glb")
                if os.path.exists(pel_path):
                    with open(pel_path, "rb") as f:
                        glb_bytes = f.read()
                    filename = "Pelike_574_gameready.glb"
                    axis = "Y"
                    azimuth = 90.0
                    z_min = 0.08
                    z_max = 0.88
                    proj_mode = "bildfeld"
                else:
                    glb_bytes = glb_processor.create_sample_3d_vase("Amphora")
                    filename = "pelike_3d.glb"
                    axis = "Y"
                    azimuth = 0.0
                    z_min = 0.15
                    z_max = 0.80
                    proj_mode = "bildfeld"
            else:
                glb_bytes = glb_processor.create_sample_3d_vase(shape_name)
                filename = f"{shape_name.lower()}_3d.glb"
                axis = "Z"
                azimuth = 0.0
                z_min = 0.10
                z_max = 0.85
                proj_mode = "frieze"

            info = glb_processor.load_glb(glb_bytes)
            profile = profile_analyzer.analyze_mesh(info["mesh"])

            def _apply_on_main():
                self.current_glb_bytes = glb_bytes
                self.current_glb_filename = filename
                self.unroll_axis = axis
                self.unroll_azimuth_offset = azimuth
                self.unroll_z_min = z_min
                self.unroll_z_max = z_max
                self.unroll_projection_mode = proj_mode
                self.seg_proj_mode.set("🏛️ Bildfeld (Seite A / Seite B)" if proj_mode == "bildfeld" else "📜 Umlaufender 360°-Fries")
                self.combo_axis.set(f"{axis}-Achse (Standard GLTF)" if axis == "Y" else f"{axis}-Achse (CAD / Blender)")
                self._set_panel_angle(azimuth)
                self.slider_z_min.set(z_min)
                self.slider_z_max.set(z_max)
                self.current_glb_info = info
                self.current_glb_mesh = info["mesh"]
                self.current_glb_proxy = info.get("proxy_mesh")
                self.current_glb_texture = info.get("texture")
                self.current_glb_uv = info.get("uv")
                self.current_glb_profile = profile
                self._full_abrollung_backup = None
                self._abrollung_crop_rect = None
                self._render_current_abrollung()
                self._update_3d_ui()

            self.after(0, _apply_on_main)

        threading.Thread(target=_task, daemon=True).start()

    def _open_glb_file(self):
        path = filedialog.askopenfilename(filetypes=[("3D GLB Models", "*.glb *.gltf")])
        if path:
            def _task():
                with open(path, "rb") as f:
                    glb_bytes = f.read()
                filename = os.path.basename(path)
                info = glb_processor.load_glb(glb_bytes)
                det = info.get("detected_axis", "Y")
                profile = profile_analyzer.analyze_mesh(info["mesh"])

                def _apply_on_main():
                    self.current_glb_bytes = glb_bytes
                    self.current_glb_filename = filename
                    self.current_glb_info = info
                    self.current_glb_mesh = info["mesh"]
                    self.current_glb_proxy = info.get("proxy_mesh")
                    self.current_glb_texture = info.get("texture")
                    self.current_glb_uv = info.get("uv")
                    self.unroll_axis = det
                    self.combo_axis.set(f"{det}-Achse" if det in ["X", "Y", "Z"] else "🤖 Auto-Erkennung")
                    self.current_glb_profile = profile
                    self._full_abrollung_backup = None
                    self._abrollung_crop_rect = None
                    self._render_current_abrollung()
                    self._update_3d_ui()

                self.after(0, _apply_on_main)

            threading.Thread(target=_task, daemon=True).start()

    def _save_abrollung_image(self):
        if self.current_abrollung_pil is not None:
            save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG Bild", "*.jpg"), ("PNG Bild", "*.png")])
            if save_path:
                self.current_abrollung_pil.save(save_path, quality=95)
                self.show_toast(f"360°-Abrollung gespeichert: {os.path.basename(save_path)}", "success")
        else:
            self.show_toast("Bitte laden oder erzeugen Sie zuerst eine 3D-Abrollung.", "warning")

    def _send_abrollung_to_sam(self):
        target_img = getattr(self, "current_abrollung_pil", None)
        if target_img is None:
            self.show_toast("Bitte laden oder erzeugen Sie zuerst eine 3D-Abrollung.", "warning")
            return

        out_dir = os.path.join(BASE_DIR, "pics")
        os.makedirs(out_dir, exist_ok=True)
        fname = f"3d_abrollung_{self.current_glb_filename or 'vase'}.png"
        export_path = os.path.normpath(os.path.join(out_dir, fname))
        target_img.save(export_path)

        self.select_view("sam")
        existing_norms = [os.path.normpath(p) for p in self.sam_image_paths]
        if export_path not in existing_norms:
            self.sam_image_paths.insert(0, export_path)
            self.sam_current_idx = 0
        else:
            self.sam_current_idx = existing_norms.index(export_path)

        self._refresh_sam_image_combo()
        self._load_sam_image_path(export_path)
        self.show_toast("360°-Abrollung an SAM-Studio übertragen!", "success")

    def _update_3d_ui(self):
        if self.current_glb_profile is None:
            return

        # Display 360° Abrollung
        if self.current_abrollung_pil is not None:
            w, h = self.current_abrollung_pil.size
            layer_label = "Normal-Map" if self.current_texture_layer == "normal" else ("Rauheit/AO" if self.current_texture_layer == "roughness" else "Fotografische Farbe")
            self.lbl_abrollung_title.configure(text=f"📜 360°-Abrollung: {layer_label} ({w} × {h} px)")
            self._on_abrollung_zoom_change()

        p = self.current_glb_profile
        shape = p["suggested_shape"]
        skos = skos_client.get_concept_by_label(shape)
        uri = skos.uri if skos else "N/A"

        metrics_text = (
            f"🏺 Erkannte Form: {shape} ({p['morphological_confidence']*100:.1f}% Konfidenz)\n"
            f"🔗 HECTOR SKOS URI: {uri}\n"
            f"📏 Gesamthöhe: {p['height']} cm | Max. Bauch-Ø: {p['max_diameter']} cm | Mündungs-Ø: {p['rim_diameter']} cm\n"
            f"📐 Schlankheit (H/D): {p['slenderness_ratio']} | Standring-Ø: {p['base_diameter']} cm | Volumen: ~{p['estimated_volume_liters']} L"
        )
        self.lbl_profile_metrics.configure(text=metrics_text)

        # Generate section drawing with visual frieze band
        buf = profile_analyzer.generate_section_drawing(
            p,
            frieze_z_min_pct=self.unroll_z_min,
            frieze_z_max_pct=self.unroll_z_max
        )
        sec_img = Image.open(buf).convert("RGB")
        self._display_image_on_label(sec_img, self.canvas_section, (620, 560))

    def _export_cva_svg(self):
        if self.current_glb_profile is None:
            self.show_toast("Bitte laden Sie zuerst ein 3D-Gefäßmodell.", "warning")
            return
        p = self.current_glb_profile
        base_name = os.path.splitext(self.current_glb_filename)[0] if self.current_glb_filename else "cva_vase"
        save_path = filedialog.asksaveasfilename(
            initialfile=f"{base_name}_cva_1zu1.svg",
            defaultextension=".svg",
            filetypes=[("Scalable Vector Graphics (SVG)", "*.svg")]
        )
        if save_path:
            svg_data = profile_analyzer.export_cva_svg(p)
            with open(save_path, "wb") as f:
                f.write(svg_data)
            self.show_toast(f"CVA-Vektorzeichnung (SVG 1:1) gespeichert: {os.path.basename(save_path)}", "success")

    def _export_cva_pdf(self):
        if self.current_glb_profile is None:
            self.show_toast("Bitte laden Sie zuerst ein 3D-Gefäßmodell.", "warning")
            return
        p = self.current_glb_profile
        base_name = os.path.splitext(self.current_glb_filename)[0] if self.current_glb_filename else "cva_vase"
        save_path = filedialog.asksaveasfilename(
            initialfile=f"{base_name}_cva_tafel.pdf",
            defaultextension=".pdf",
            filetypes=[("PDF Drucktafel", "*.pdf")]
        )
        if save_path:
            pdf_data = profile_analyzer.export_cva_pdf(p)
            with open(save_path, "wb") as f:
                f.write(pdf_data)
            self.show_toast(f"CVA-Drucktafel (PDF) gespeichert: {os.path.basename(save_path)}", "success")

    def _export_cva_png(self):
        if self.current_glb_profile is None:
            self.show_toast("Bitte laden Sie zuerst ein 3D-Gefäßmodell.", "warning")
            return
        p = self.current_glb_profile
        base_name = os.path.splitext(self.current_glb_filename)[0] if self.current_glb_filename else "cva_vase"
        save_path = filedialog.asksaveasfilename(
            initialfile=f"{base_name}_cva_300dpi.png",
            defaultextension=".png",
            filetypes=[("PNG Bilddatei (300 DPI)", "*.png")]
        )
        if save_path:
            buf = profile_analyzer.generate_cva_plate(p, dpi=300, output_format="png")
            with open(save_path, "wb") as f:
                f.write(buf.getvalue())
            self.show_toast(f"CVA-Publikationstafel (300 DPI PNG) gespeichert: {os.path.basename(save_path)}", "success")

    def _export_3d_lod(self):
        if self.current_glb_profile is None:
            self.show_toast("Bitte laden Sie zuerst ein 3D-Gefäßmodell.", "warning")
            return
        p = self.current_glb_profile
        skos = skos_client.get_concept_by_label(p["suggested_shape"])
        item = {
            "id": os.path.splitext(self.current_glb_filename)[0],
            "name": f"3D-Vase: {p['suggested_shape']}",
            "shape_label": p["suggested_shape"],
            "shape_uri": skos.uri if skos else "",
            "model_3d": self.current_glb_filename,
            "dimensions_3d": p
        }
        jsonld_data = LODExporter.to_jsonld([item])
        save_path = filedialog.asksaveasfilename(defaultextension=".jsonld", filetypes=[("JSON-LD", "*.jsonld")])
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(jsonld_data, f, indent=2, ensure_ascii=False)
            self.show_toast(f"3D-LOD Datensatz gespeichert: {os.path.basename(save_path)}", "success")

    # ---------------------------------------------------------
    # VIEW 2: SAM Segmentierungs & Trainings-Studio
    # ---------------------------------------------------------
    def _build_sam_view(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # Header Bar: Batch Import & Navigation
        top_bar = ctk.CTkFrame(frame, height=52, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        lbl = ctk.CTkLabel(top_bar, text="✨ SAM Studio & Batch-Annotation", font=ctk.CTkFont(size=14, weight="bold"), text_color=("#0f172a", "#f8fafc"))
        lbl.pack(side="left", padx=12, pady=8)

        # Folder Import button (Batch)
        btn_import_dir = ctk.CTkButton(top_bar, text="📁 Ordner laden (Batch)", command=self._import_sam_folder, width=150, fg_color=("#0284c7", "#334155"), hover_color=("#0369a1", "#475569"))
        btn_import_dir.pack(side="left", padx=4, pady=8)

        # Multi-Files Import button
        btn_import_files = ctk.CTkButton(top_bar, text="📂 Bilder wählen", command=self._open_sam_images_batch, width=130, fg_color=("#0284c7", "#334155"), hover_color=("#0369a1", "#475569"))
        btn_import_files.pack(side="left", padx=4, pady=8)

        # Navigation & Counter on right side
        self.btn_sam_next = ctk.CTkButton(top_bar, text="▶️ Weiter", width=80, height=32, fg_color=("#0284c7", "#334155"), hover_color=("#0369a1", "#475569"), command=self._next_sam_image)
        self.btn_sam_next.pack(side="right", padx=6, pady=8)

        self.combo_pics_sam = ctk.CTkComboBox(top_bar, values=["Keine Bilder"], command=self._on_select_pic_sam_combo, width=250)
        self.combo_pics_sam.pack(side="right", padx=4, pady=8)

        self.btn_sam_prev = ctk.CTkButton(top_bar, text="◀️ Zurück", width=80, height=32, fg_color=("#0284c7", "#334155"), hover_color=("#0369a1", "#475569"), command=self._prev_sam_image)
        self.btn_sam_prev.pack(side="right", padx=4, pady=8)

        self.lbl_sam_img_counter = ctk.CTkLabel(top_bar, text="0 / 0", font=ctk.CTkFont(size=12, weight="bold"), text_color=("#0284c7", "#38bdf8"))
        self.lbl_sam_img_counter.pack(side="right", padx=8, pady=8)

        # Left Column: Image Canvas & Interactive Tool Suite
        col_left = ctk.CTkFrame(frame, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        col_left.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        col_left.grid_rowconfigure(1, weight=1)
        col_left.grid_columnconfigure(0, weight=1)

        # Top Zoom & Header Toolbar
        top_left = ctk.CTkFrame(col_left, fg_color="transparent")
        top_left.grid(row=0, column=0, sticky="ew", padx=10, pady=6)

        ctk.CTkLabel(top_left, text="🖼️ SAM Leinwand", font=ctk.CTkFont(size=13, weight="bold"), text_color=("#7c3aed", "#c084fc")).pack(side="left")

        # View Mode Switcher [🖼️ Original, 🎭 Figuren, 🛡️ Figuren-Maske]
        self.seg_view_mode = ctk.CTkSegmentedButton(
            top_left,
            values=["🖼️ Original", "🎭 Figuren", "🛡️ Figuren-Maske"],
            command=self._on_change_sam_view_mode
        )
        self.seg_view_mode.set("🎭 Figuren")
        self.seg_view_mode.pack(side="left", padx=(15, 5))

        # Zoom Bar
        self.slider_sam_zoom = ctk.CTkSlider(top_left, from_=0.5, to=4.0, number_of_steps=70, width=110, command=self._on_sam_zoom_change)
        self.slider_sam_zoom.set(1.0)
        self.slider_sam_zoom.pack(side="right", padx=5)

        self.lbl_sam_zoom_val = ctk.CTkLabel(top_left, text="100%", width=36, font=ctk.CTkFont(size=11), text_color=("#0f172a", "#f8fafc"))
        self.lbl_sam_zoom_val.pack(side="right", padx=2)

        ctk.CTkButton(top_left, text="➕", width=28, height=26, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._step_sam_zoom(1.25)).pack(side="right", padx=2)
        ctk.CTkButton(top_left, text="➖", width=28, height=26, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._step_sam_zoom(0.80)).pack(side="right", padx=2)
        ctk.CTkButton(top_left, text="🔄 100%", width=55, height=26, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=self._reset_sam_zoom).pack(side="right", padx=2)

        # Interactive Canvas
        self.canvas_sam = tk.Canvas(col_left, bg="#ffffff", highlightthickness=1, highlightbackground="#e2e8f0", cursor="crosshair")
        self.canvas_sam.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        self._sam_pan_x = 0
        self._sam_pan_y = 0
        self._sam_pan_start = None
        self._sam_box_start = None
        self._sam_box_rect = None
        self._sam_temp_img = None
        self.sam_overlay_pil = None
        self.sam_struct_mask_pil = None

        self.canvas_sam.bind("<Configure>", self._redraw_sam_canvas)
        self.canvas_sam.bind("<MouseWheel>", self._on_sam_mousewheel)
        self.canvas_sam.bind("<Button-4>", self._on_sam_mousewheel)
        self.canvas_sam.bind("<Button-5>", self._on_sam_mousewheel)
        self.canvas_sam.bind("<ButtonPress-1>", self._on_sam_mouse_down)
        self.canvas_sam.bind("<B1-Motion>", self._on_sam_mouse_drag)
        self.canvas_sam.bind("<ButtonRelease-1>", self._on_sam_mouse_up)
        self.canvas_sam.bind("<ButtonPress-2>", self._on_sam_pan_down)
        self.canvas_sam.bind("<ButtonPress-3>", self._on_sam_pan_down)
        self.canvas_sam.bind("<B2-Motion>", self._on_sam_pan_drag)
        self.canvas_sam.bind("<B3-Motion>", self._on_sam_pan_drag)
        self.canvas_sam.bind("<Delete>", lambda e: self._delete_current_motif())

        # Tool Mode & Model Bar (On Left Side)
        tool_panel = ctk.CTkFrame(col_left, fg_color=("#f1f5f9", "#0f172a"), corner_radius=8)
        tool_panel.grid(row=2, column=0, sticky="ew", padx=10, pady=4)

        ctk.CTkLabel(tool_panel, text="🧠 Modell:", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#0f172a", "#f8fafc")).pack(side="left", padx=(8, 3), pady=4)
        self.combo_sam_backend = ctk.CTkComboBox(
            tool_panel,
            values=["🧠 Meta SAM (sam_b.pt)", "⚡ MobileSAM (ViT)"],
            width=185,
            command=self._on_change_sam_backend
        )
        self.combo_sam_backend.set("🧠 Meta SAM (sam_b.pt)")
        self.combo_sam_backend.pack(side="left", padx=(0, 6), pady=4)

        ctk.CTkLabel(tool_panel, text="🎨 Stil:", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#0f172a", "#f8fafc")).pack(side="left", padx=(4, 2), pady=4)
        self.combo_sam_style = ctk.CTkComboBox(
            tool_panel,
            values=["Auto (AI)", "Attisch Rotfigurig", "Attisch Schwarzfigurig", "Korinthisch", "Geometrisch", "Weißgrundig"],
            width=135,
            command=self._on_change_sam_style
        )
        self.combo_sam_style.set("Auto (AI)")
        self.combo_sam_style.pack(side="left", padx=(0, 6), pady=4)

        ctk.CTkLabel(tool_panel, text="🛠️ Werkzeug:", font=ctk.CTkFont(size=11, weight="bold"), text_color=("#0f172a", "#f8fafc")).pack(side="left", padx=(4, 4), pady=4)

        self.seg_mode_btn = ctk.CTkSegmentedButton(
            tool_panel,
            values=["🪄 Zauberstab", "🎯 Box-Auswahl", "✏️ Stift (Zeichnen)", "🧹 Radierer (Stift)", "🔗 Verbinden"],
            command=self._on_change_seg_mode
        )
        self.seg_mode_btn.set("🪄 Zauberstab")
        self.seg_mode_btn.pack(side="left", fill="x", expand=True, padx=4, pady=4)

        # Quick Actions & Brush Bar (On Left Side)
        act_panel = ctk.CTkFrame(col_left, fg_color="transparent")
        act_panel.grid(row=3, column=0, sticky="ew", padx=10, pady=3)

        self.chk_sam_constraint = ctk.CTkCheckBox(act_panel, text="🛡️ Tongrund-Sperre", font=ctk.CTkFont(size=11))
        self.chk_sam_constraint.select()
        self.chk_sam_constraint.pack(side="left", padx=(2, 4))

        self.btn_invert_mask = ctk.CTkButton(
            act_panel, text="🔄 Invertieren", width=95, height=30,
            fg_color=("#0284c7", "#334155"), hover_color=("#0369a1", "#475569"), text_color="#ffffff",
            font=ctk.CTkFont(size=11, weight="bold"),
            command=self._toggle_invert_mask
        )
        self.btn_invert_mask.pack(side="left", padx=(2, 6))

        ctk.CTkLabel(act_panel, text="🖌️ Pinsel:", font=ctk.CTkFont(size=11), text_color=("#475569", "#94a3b8")).pack(side="left", padx=(2, 2))
        self.slider_brush_size = ctk.CTkSlider(act_panel, from_=3, to=40, number_of_steps=37, width=70, command=self._on_brush_size_change)
        self.slider_brush_size.set(12)
        self.slider_brush_size.pack(side="left", padx=2)

        self.lbl_brush_size = ctk.CTkLabel(act_panel, text="12px", width=30, font=ctk.CTkFont(size=11), text_color=("#0f172a", "#f8fafc"))
        self.lbl_brush_size.pack(side="left", padx=(0, 4))

        ctk.CTkButton(act_panel, text="➕ +5px", width=55, height=30, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._adjust_current_mask(dilate=True)).pack(side="left", padx=2)
        ctk.CTkButton(act_panel, text="➖ -5px", width=55, height=30, fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._adjust_current_mask(dilate=False)).pack(side="left", padx=2)
        ctk.CTkButton(act_panel, text="🗑️ Löschen", fg_color="#e11d48", hover_color="#be123c", text_color="#ffffff", width=75, height=30, command=self._delete_current_motif).pack(side="left", padx=2)

        btn_run_all = ctk.CTkButton(
            act_panel, text="🚀 Alle Figuren automatisch", height=30,
            fg_color="#7c3aed", hover_color="#6d28d9", text_color="#ffffff", font=ctk.CTkFont(size=11, weight="bold"),
            command=self._run_sam_segmentation
        )
        btn_run_all.pack(side="right", padx=2, fill="x", expand=True)

        self.lbl_sam_status = ctk.CTkLabel(
            col_left,
            text="🪄 Zauberstab aktiv: Klicke auf eine Figur auf der Leinwand, um die KI-Segmentierung punktgenau anzustoßen!",
            font=ctk.CTkFont(size=11), text_color=("#0284c7", "#38bdf8")
        )
        self.lbl_sam_status.grid(row=4, column=0, padx=10, pady=2, sticky="w")

        # Right Column: Multi-Motif Inspector (Stacked Cards & Active Learning)
        col_right = ctk.CTkFrame(frame, fg_color=THEME["bg_card"], corner_radius=8, border_width=1, border_color=THEME["border_subtle"])
        col_right.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        col_right.grid_rowconfigure(2, weight=1)
        col_right.grid_columnconfigure(0, weight=1)

        # Header Bar with Controls
        header_right = ctk.CTkFrame(col_right, fg_color="transparent")
        header_right.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 4))

        self.lbl_motifs_title = ctk.CTkLabel(
            header_right,
            text="🏷️ Motiv-Prüfung (0 Figuren)",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=THEME["text_accent"]
        )
        self.lbl_motifs_title.pack(side="left")

        # Multi-Screen Button & SKOS Button
        btn_popout_motifs = ctk.CTkButton(
            header_right, text="🪟 Multi-Screen",
            font=ctk.CTkFont(size=10, weight="bold"), width=95, height=26,
            fg_color=THEME["primary"], hover_color=THEME["primary_hover"], text_color="#ffffff",
            command=self._open_multi_screen_motif_window
        )
        btn_popout_motifs.pack(side="right", padx=(4, 0))

        btn_open_skos = ctk.CTkButton(
            header_right, text="🌳 SKOS-Editor",
            font=ctk.CTkFont(size=10, weight="bold"), width=95, height=26,
            fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"],
            command=self._open_multi_screen_skos_window
        )
        btn_open_skos.pack(side="right", padx=(4, 0))

        # Quick Batch Actions Bar
        batch_act_bar = ctk.CTkFrame(col_right, fg_color="transparent")
        batch_act_bar.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 4))

        btn_apply_all_ai = ctk.CTkButton(
            batch_act_bar, text="✨ Alle KI-Vorschläge anwenden",
            height=26, font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#7c3aed", hover_color="#6d28d9", text_color="#ffffff",
            command=self._apply_all_ai_suggestions
        )
        btn_apply_all_ai.pack(side="left", fill="x", expand=True, padx=(0, 3))

        btn_save_all_m = ctk.CTkButton(
            batch_act_bar, text="💾 Alle verifizieren & speichern",
            height=26, font=ctk.CTkFont(size=10, weight="bold"),
            fg_color="#10b981", hover_color="#059669", text_color="#ffffff",
            command=self._save_all_verified_motifs
        )
        btn_save_all_m.pack(side="right", fill="x", expand=True, padx=(3, 0))

        # Scrollable Container with Stacked Motif Cards (untereinander & zoombar)
        self.scroll_motifs = ctk.CTkScrollableFrame(col_right, fg_color="transparent")
        self.scroll_motifs.grid(row=2, column=0, sticky="nsew", padx=6, pady=4)

        self.lbl_empty_motifs = ctk.CTkLabel(
            self.scroll_motifs,
            text="💡 Keine Motive geladen.\nNutze den Zauberstab (Klick auf Figur), Box-Auswahl\noder '🚀 Alle Figuren automatisch', um Figuren zu extrahieren!",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_muted"],
            justify="center"
        )
        self.lbl_empty_motifs.pack(pady=40)

        # Dataset Export Toolbar at Bottom
        dl_frame = ctk.CTkFrame(col_right, fg_color=THEME["bg_card_inner"], corner_radius=6, border_width=1, border_color=THEME["border_subtle"])
        dl_frame.grid(row=3, column=0, padx=10, pady=(4, 8), sticky="ew")

        self.lbl_pool_counter = ctk.CTkLabel(
            dl_frame, text="📦 Pool: 0", font=ctk.CTkFont(size=11, weight="bold"), text_color=THEME["text_accent"]
        )
        self.lbl_pool_counter.pack(side="left", padx=8, pady=4)

        self.btn_export_zip = ctk.CTkButton(dl_frame, text="📦 ZIP-Export", width=95, height=26, font=ctk.CTkFont(size=10), fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"], command=self._export_train_zip)
        self.btn_export_zip.pack(side="left", padx=2, pady=4)

        self.btn_export_coco = ctk.CTkButton(dl_frame, text="📄 COCO JSON", width=95, height=26, font=ctk.CTkFont(size=10), fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"], command=self._export_train_coco)
        self.btn_export_coco.pack(side="left", padx=2, pady=4)

        self.btn_clear_pool = ctk.CTkButton(dl_frame, text="🗑️ Leeren", width=75, height=26, font=ctk.CTkFont(size=10), fg_color="#ef4444", hover_color="#dc2626", text_color="#ffffff", command=self._clear_train_pool)
        self.btn_clear_pool.pack(side="right", padx=6, pady=4)

        return frame

    def _get_pic_list(self) -> List[str]:
        if os.path.exists(PICS_DIR):
            return [f for f in os.listdir(PICS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.tif', '.bmp'))]
        return []

    def _open_sam_image(self):
        self._open_sam_images_batch()

    def _init_sam_image_list(self):
        self.sam_image_paths = []
        self.sam_current_idx = 0
        if os.path.exists(PICS_DIR):
            found = [
                os.path.join(PICS_DIR, f)
                for f in os.listdir(PICS_DIR)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.tif', '.bmp'))
            ]
            self.sam_image_paths = sorted(found)
        self._refresh_sam_image_combo()

    def _load_default_sam_image(self):
        self._init_sam_image_list()
        if self.sam_image_paths:
            self._load_sam_image_at_index(0)

    def _refresh_sam_image_combo(self):
        total = len(self.sam_image_paths)
        if total > 0:
            options = [f"[{i+1}/{total}] {os.path.basename(p)}" for i, p in enumerate(self.sam_image_paths)]
            self.combo_pics_sam.configure(values=options)
            cur = min(self.sam_current_idx, total - 1)
            self.combo_pics_sam.set(options[cur])
            self.lbl_sam_img_counter.configure(text=f"{cur+1} / {total}")
        else:
            self.combo_pics_sam.configure(values=["Keine Bilder geladen"])
            self.combo_pics_sam.set("Keine Bilder geladen")
            self.lbl_sam_img_counter.configure(text="0 / 0")

    def _load_sam_image_at_index(self, idx: int):
        if not self.sam_image_paths:
            return
        self.sam_current_idx = max(0, min(idx, len(self.sam_image_paths) - 1))
        self._refresh_sam_image_combo()
        self._load_sam_image_path(self.sam_image_paths[self.sam_current_idx])

    def _prev_sam_image(self):
        if self.sam_image_paths and self.sam_current_idx > 0:
            self._load_sam_image_at_index(self.sam_current_idx - 1)

    def _next_sam_image(self):
        if self.sam_image_paths and self.sam_current_idx < len(self.sam_image_paths) - 1:
            self._load_sam_image_at_index(self.sam_current_idx + 1)

    def _on_select_pic_sam_combo(self, choice: str):
        if not self.sam_image_paths or choice.startswith("Keine"):
            return
        for i, p in enumerate(self.sam_image_paths):
            tag = f"[{i+1}/{len(self.sam_image_paths)}]"
            if choice.startswith(tag) or os.path.basename(p) in choice:
                self._load_sam_image_at_index(i)
                break

    def _open_sam_images_batch(self):
        files = filedialog.askopenfilenames(
            title="Vasenbilder auswählen (Batch)",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.tif *.bmp")]
        )
        if files:
            self.sam_image_paths = list(files)
            self.sam_current_idx = 0
            self._load_sam_image_at_index(0)

    def _import_sam_folder(self):
        folder = filedialog.askdirectory(title="Ordner mit Vasenbildern auswählen")
        if folder and os.path.exists(folder):
            found = []
            for root, _, filenames in os.walk(folder):
                for f in filenames:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.tif', '.bmp')):
                        found.append(os.path.join(root, f))
            if found:
                self.sam_image_paths = sorted(found)
                self.sam_current_idx = 0
                self._load_sam_image_at_index(0)
            else:
                self.lbl_sam_status.configure(text="⚠️ Keine Bilddateien (.jpg, .png, etc.) in diesem Ordner gefunden.")

    def _load_default_sam_image(self):
        self._init_sam_image_list()
        if self.sam_image_paths:
            self._load_sam_image_at_index(0)

    def _on_sam_zoom_change(self, val=None):
        zoom = float(self.slider_sam_zoom.get())
        if hasattr(self, "lbl_sam_zoom_val"):
            self.lbl_sam_zoom_val.configure(text=f"{int(zoom * 100)}%")
        self._redraw_sam_canvas()

    def _step_sam_zoom(self, factor: float):
        curr = float(self.slider_sam_zoom.get())
        new_val = np.clip(curr * factor, 0.5, 4.0)
        self.slider_sam_zoom.set(new_val)
        self._on_sam_zoom_change()

    def _reset_sam_zoom(self):
        self.slider_sam_zoom.set(1.0)
        self._sam_pan_x = 0
        self._sam_pan_y = 0
        self._on_sam_zoom_change()

    def _on_sam_mousewheel(self, event):
        delta = 1.15 if (event.delta > 0 or event.num == 4) else 0.85
        self._step_sam_zoom(delta)

    def _on_sam_pan_down(self, event):
        self._sam_pan_start = (event.x, event.y)

    def _on_sam_pan_drag(self, event):
        if self._sam_pan_start is not None:
            dx = event.x - self._sam_pan_start[0]
            dy = event.y - self._sam_pan_start[1]
            self._sam_pan_x += dx
            self._sam_pan_y += dy
            self._sam_pan_start = (event.x, event.y)
            self._redraw_sam_canvas()

    def _sam_canvas_coords_to_image_rect(self, x0: float, y0: float, x1: float, y1: float) -> Optional[Tuple[int, int, int, int]]:
        if self.sam_current_pil is None:
            return None
        canvas_w = max(1, self.canvas_sam.winfo_width())
        canvas_h = max(1, self.canvas_sam.winfo_height())
        zoom = float(self.slider_sam_zoom.get())
        iw, ih = self.sam_current_pil.size

        base_scale = min(canvas_w / max(1, iw), canvas_h / max(1, ih))
        disp_w = max(1, int(iw * base_scale * zoom))
        disp_h = max(1, int(ih * base_scale * zoom))

        cx = canvas_w // 2 + self._sam_pan_x
        cy = canvas_h // 2 + self._sam_pan_y

        img_x0 = cx - disp_w / 2.0
        img_y0 = cy - disp_h / 2.0

        min_x, max_x = min(x0, x1), max(x0, x1)
        min_y, max_y = min(y0, y1), max(y0, y1)

        u0 = (min_x - img_x0) / float(disp_w)
        v0 = (min_y - img_y0) / float(disp_h)
        u1 = (max_x - img_x0) / float(disp_w)
        v1 = (max_y - img_y0) / float(disp_h)

        px0 = int(np.clip(u0, 0.0, 1.0) * iw)
        py0 = int(np.clip(v0, 0.0, 1.0) * ih)
        px1 = int(np.clip(u1, 0.0, 1.0) * iw)
        py1 = int(np.clip(v1, 0.0, 1.0) * ih)

        return (px0, py0, px1, py1)

    def _sam_canvas_coords_to_image_point(self, cx_val: float, cy_val: float) -> Optional[Tuple[int, int]]:
        if self.sam_current_pil is None:
            return None
        canvas_w = max(1, self.canvas_sam.winfo_width())
        canvas_h = max(1, self.canvas_sam.winfo_height())
        zoom = float(self.slider_sam_zoom.get())
        iw, ih = self.sam_current_pil.size

        base_scale = min(canvas_w / max(1, iw), canvas_h / max(1, ih))
        disp_w = max(1, int(iw * base_scale * zoom))
        disp_h = max(1, int(ih * base_scale * zoom))

        cx = canvas_w // 2 + self._sam_pan_x
        cy = canvas_h // 2 + self._sam_pan_y

        img_x0 = cx - disp_w / 2.0
        img_y0 = cy - disp_h / 2.0

        u = (cx_val - img_x0) / float(disp_w)
        v = (cy_val - img_y0) / float(disp_h)
        if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
            return (int(u * iw), int(v * ih))
        return None

    def _on_brush_size_change(self, val=None):
        if hasattr(self, "lbl_brush_size") and hasattr(self, "slider_brush_size"):
            self.lbl_brush_size.configure(text=f"{int(float(self.slider_brush_size.get()))}px")

    def _on_sam_mouse_down(self, event):
        curr_mode = self.seg_mode_btn.get() if hasattr(self, "seg_mode_btn") else "🪄 Zauberstab"

        if curr_mode in ["✏️ Stift (Zeichnen)", "🧹 Radierer (Stift)"]:
            self._brush_canvas_pts = [(event.x, event.y)]
            self._sam_box_rect = None
            self.canvas_sam.delete("sam_brush")
            r = max(2, int(float(self.slider_brush_size.get()) * float(self.slider_sam_zoom.get()) * 0.5))
            color = "#10b981" if curr_mode == "✏️ Stift (Zeichnen)" else "#ef4444"
            self.canvas_sam.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill=color, outline="", tags="sam_brush")
            return

        self._brush_canvas_pts = []
        self._sam_box_start = (event.x, event.y)
        self._sam_box_rect = None
        self.canvas_sam.delete("sam_box")

    def _on_sam_mouse_drag(self, event):
        if self.sam_current_pil is None:
            return

        curr_mode = self.seg_mode_btn.get() if hasattr(self, "seg_mode_btn") else "🪄 Zauberstab"

        if curr_mode in ["✏️ Stift (Zeichnen)", "🧹 Radierer (Stift)"]:
            if hasattr(self, "_brush_canvas_pts") and self._brush_canvas_pts:
                p_prev = self._brush_canvas_pts[-1]
                self._brush_canvas_pts.append((event.x, event.y))
                r = max(2, int(float(self.slider_brush_size.get()) * float(self.slider_sam_zoom.get()) * 0.5))
                color = "#10b981" if curr_mode == "✏️ Stift (Zeichnen)" else "#ef4444"
                self.canvas_sam.create_line(p_prev[0], p_prev[1], event.x, event.y, fill=color, width=r*2, capstyle="round", joinstyle="round", tags="sam_brush")
            return

        if self._sam_box_start is not None:
            x0, y0 = self._sam_box_start
            x1, y1 = event.x, event.y
            self.canvas_sam.delete("sam_box")
            self.canvas_sam.create_rectangle(x0, y0, x1, y1, outline="#38bdf8", width=2, tags="sam_box")
            self._sam_box_rect = self._sam_canvas_coords_to_image_rect(x0, y0, x1, y1)

    def _on_change_sam_backend(self, choice: str):
        if "Meta SAM" in choice:
            sam_segmenter.set_backend("meta_sam")
            msg = "🧠 Meta SAM 1 (sam_b.pt) aktiv: Höchste Detailtreue bei feinen Pinsel- und Relieflinien."
        elif "MobileSAM" in choice:
            sam_segmenter.set_backend("mobilesam")
            msg = "⚡ MobileSAM aktiv (Vision Transformer): Schnell & ressourcenschonend."
        else:
            sam_segmenter.set_backend("meta_sam")
            msg = "🧠 Meta SAM 1 (sam_b.pt) aktiv."
        if hasattr(self, "lbl_sam_status"):
            self.lbl_sam_status.configure(text=msg)

    def _on_change_seg_mode(self, mode_val: str):
        if hasattr(self, "lbl_sam_status"):
            if mode_val == "🪄 Zauberstab":
                self.lbl_sam_status.configure(text="🪄 ZAUBERSTAB: Klicke auf eine beliebige Figur auf der Leinwand, um die KI-Segmentierung punktgenau anzustoßen.")
            elif mode_val == "🎯 Box-Auswahl":
                self.lbl_sam_status.configure(text="🎯 BOX-AUSWAHL: Ziehe eine Box mit der Maus, um den Bereich per KI zu segmentieren.")
            elif mode_val == "✏️ Stift (Zeichnen)":
                self.lbl_sam_status.configure(text="✏️ STIFT: Zeichne mit gehaltener linker Maustaste, um Bereiche zur aktiven Maske hinzuzufügen.")
            elif mode_val == "🧹 Radierer (Stift)":
                self.lbl_sam_status.configure(text="🧹 RADIERER: Radiere mit gehaltener linker Maustaste unerwünschte Teile aus der aktiven Maske weg.")
            elif mode_val == "🔗 Verbinden":
                self.lbl_sam_status.configure(text="🔗 VERSCHMELZEN: Klicke auf ein zweites Motiv auf der Leinwand, um es mit dem aktiven Motiv zu verbinden.")

    def _toggle_merge_mode(self):
        self.seg_mode_btn.set("🔗 Verbinden")
        self._on_change_seg_mode("🔗 Verbinden")

    def _get_active_motif_index(self) -> Optional[int]:
        if not self.sam_detected_motifs:
            return None
        choice = self.combo_motifs.get()
        if not choice or "#" not in choice:
            return None
        try:
            idx = int(choice.split("#")[1].split(" ")[0]) - 1
            if 0 <= idx < len(self.sam_detected_motifs):
                return idx
        except Exception:
            pass
        return None

    def _adjust_current_mask(self, dilate: bool = True):
        idx = self._get_active_motif_index()
        if idx is None or self.sam_current_pil is None:
            return
        m = self.sam_detected_motifs[idx]
        if dilate:
            new_mask = sam_segmenter.dilate_mask(m["mask"], pixels=6)
        else:
            new_mask = sam_segmenter.erode_mask(m["mask"], pixels=6)

        updated = sam_segmenter.recalculate_motif(self.sam_current_pil, new_mask, m["id"])
        if updated:
            self.sam_detected_motifs[idx] = updated
            overlay = sam_segmenter.draw_segmentation_overlay(self.sam_current_pil, self.sam_detected_motifs, selected_id=updated["id"])
            self.sam_overlay_pil = overlay
            self._redraw_sam_canvas()
            opt_str = f"Motiv #{updated['id']+1} ({updated['area_pct']}% Fläche)"
            self._update_motif_combobox_list(select_idx=idx)
            self._on_select_motif(opt_str)

    def _delete_current_motif(self):
        idx = self._get_active_motif_index()
        if idx is None:
            return
        del self.sam_detected_motifs[idx]
        for i, m in enumerate(self.sam_detected_motifs):
            m["id"] = i

        if self.sam_current_pil is not None:
            overlay = sam_segmenter.draw_segmentation_overlay(self.sam_current_pil, self.sam_detected_motifs)
            self.sam_overlay_pil = overlay
            self._redraw_sam_canvas()

        self._update_motif_combobox_list(select_idx=max(0, idx - 1) if self.sam_detected_motifs else None)

    def _merge_motif_with(self, target_idx: int):
        active_idx = self._get_active_motif_index()
        if active_idx is None or target_idx == active_idx or self.sam_current_pil is None:
            return
        if target_idx < 0 or target_idx >= len(self.sam_detected_motifs):
            return

        m_active = self.sam_detected_motifs[active_idx]
        m_target = self.sam_detected_motifs[target_idx]

        merged_mask = sam_segmenter.merge_masks(m_active["mask"], m_target["mask"])
        updated = sam_segmenter.recalculate_motif(self.sam_current_pil, merged_mask, m_active["id"])
        if updated:
            self.sam_detected_motifs[active_idx] = updated
            del self.sam_detected_motifs[target_idx]
            for i, m in enumerate(self.sam_detected_motifs):
                m["id"] = i

            new_active_idx = updated["id"]
            overlay = sam_segmenter.draw_segmentation_overlay(self.sam_current_pil, self.sam_detected_motifs, selected_id=new_active_idx)
            self.sam_overlay_pil = overlay
            self._redraw_sam_canvas()
            self._update_motif_combobox_list(select_idx=new_active_idx)
            self._on_select_motif(f"Motiv #{new_active_idx+1} ({self.sam_detected_motifs[new_active_idx]['area_pct']}% Fläche)")

        # Switch back to Zauberstab mode
        self.seg_mode_btn.set("🪄 Zauberstab")
        self._on_change_seg_mode("🪄 Zauberstab")

    def _update_motif_combobox_list(self, select_idx: Optional[int] = None):
        if self.sam_detected_motifs:
            options = [
                f"{m.get('label') or ('Motiv #' + str(m.get('id', 0) + 1))} ({m.get('area_pct', 0)}% Fläche)"
                for m in self.sam_detected_motifs
            ]
            self.combo_motifs.configure(values=options)
            if select_idx is not None and 0 <= select_idx < len(options):
                self.combo_motifs.set(options[select_idx])
            else:
                self.combo_motifs.set(options[0])
        else:
            self.combo_motifs.configure(values=["Keine Motive erkannt"])
            self.combo_motifs.set("Keine Motive erkannt")
            self.canvas_cutout.configure(image=None, text="Kein Motiv")
            if hasattr(self, "canvas_struct_mask"):
                self.canvas_struct_mask.configure(image=None, text="Keine Maske")
            self.lbl_ai_suggestion.configure(text="KI-Vorschlag: -\nKonfidenz: -\nSKOS: -")

    def _on_sam_mouse_up(self, event):
        curr_mode = self.seg_mode_btn.get() if hasattr(self, "seg_mode_btn") else "🪄 Zauberstab"

        # Mode: ✏️ Stift (Zeichnen) or 🧹 Radierer (Stift)
        if curr_mode in ["✏️ Stift (Zeichnen)", "🧹 Radierer (Stift)"]:
            self.canvas_sam.delete("sam_brush")
            if hasattr(self, "_brush_canvas_pts") and self._brush_canvas_pts and self.sam_current_pil is not None:
                active_idx = self._get_active_motif_index()
                if active_idx is not None:
                    img_pts = []
                    for cx, cy in self._brush_canvas_pts:
                        pt = self._sam_canvas_coords_to_image_point(cx, cy)
                        if pt is not None:
                            img_pts.append(pt)

                    if img_pts:
                        m = self.sam_detected_motifs[active_idx]
                        val = 1 if curr_mode == "✏️ Stift (Zeichnen)" else 0
                        brush_r = int(float(self.slider_brush_size.get()))
                        new_mask = sam_segmenter.draw_stroke_on_mask(m["mask"], img_pts, value=val, radius=brush_r)
                        updated = sam_segmenter.recalculate_motif(self.sam_current_pil, new_mask, m["id"])
                        if updated:
                            self.sam_detected_motifs[active_idx] = updated
                            overlay = sam_segmenter.draw_segmentation_overlay(self.sam_current_pil, self.sam_detected_motifs, selected_id=m["id"])
                            self.sam_overlay_pil = overlay
                            self._redraw_sam_canvas()
                            self._update_motif_combobox_list(select_idx=active_idx)
                            self._on_select_motif(f"Motiv #{m['id']+1} ({updated['area_pct']}% Fläche)")
                        else:
                            # Completely erased
                            self._delete_current_motif()
            self._brush_canvas_pts = []
            return

        # Check if user dragged a rectangle box for Box-Auswahl
        if self._sam_box_rect:
            px0, py0, px1, py1 = self._sam_box_rect
            if (px1 - px0) > 12 and (py1 - py0) > 12:
                if (curr_mode in ["🎯 Box-Auswahl", "🪄 Zauberstab"]) and self.sam_current_pil is not None:
                    self.lbl_sam_status.configure(text=f"⏳ Segmentiere Box ({px0},{py0}) bis ({px1},{py1}) mit Stil-Maskierung...")
                    style_val = self.combo_sam_style.get() if hasattr(self, "combo_sam_style") else "Auto (AI)"
                    style_str = "auto" if "Auto" in style_val else style_val
                    use_const = bool(self.chk_sam_constraint.get()) if hasattr(self, "chk_sam_constraint") else True
                    
                    def _task():
                        invert_flag = getattr(self, "sam_mask_inverted", False)
                        motif = sam_segmenter.segment_at_point_or_box(
                            self.sam_current_pil,
                            box=(px0, py0, px1, py1),
                            style=style_str,
                            use_style_constraint=use_const,
                            invert=invert_flag
                        )
                        if motif:
                            self.sam_detected_motifs.append(motif)
                            for idx, m in enumerate(self.sam_detected_motifs):
                                m["id"] = idx

                            overlay = sam_segmenter.draw_segmentation_overlay(self.sam_current_pil, self.sam_detected_motifs, selected_id=motif["id"])
                            self.sam_overlay_pil = overlay

                            def _apply():
                                self._redraw_sam_canvas()
                                self._render_all_motif_cards()
                                self.lbl_sam_status.configure(text=f"✨ Box #{motif['id']+1} erfolgreich segmentiert!")
                            self.after(0, _apply)
                    threading.Thread(target=_task, daemon=True).start()
                    return

        # If it was a point click on the canvas (not a drag box)
        pt = self._sam_canvas_coords_to_image_point(event.x, event.y)
        if pt is None or self.sam_current_pil is None:
            return

        px, py = pt

        # Mode: 🔗 Verbinden (Click on target motif to merge)
        if curr_mode == "🔗 Verbinden" and self.sam_detected_motifs:
            for m in self.sam_detected_motifs:
                if 0 <= py < m["mask"].shape[0] and 0 <= px < m["mask"].shape[1]:
                    if m["mask"][py, px]:
                        self._merge_motif_with(m["id"])
                        return

        # Mode: 🪄 Zauberstab (Magic Wand click segmentation)
        if curr_mode == "🪄 Zauberstab":
            existing_idx = None
            if self.sam_detected_motifs:
                for m in self.sam_detected_motifs:
                    if 0 <= py < m["mask"].shape[0] and 0 <= px < m["mask"].shape[1]:
                        if m["mask"][py, px]:
                            existing_idx = m["id"]
                            break

            if existing_idx is not None:
                return

            self.lbl_sam_status.configure(text=f"🪄 Zauberstab: Analysiere Figur an Position ({px}, {py}) mit Stil-Maskierung...")
            style_val = self.combo_sam_style.get() if hasattr(self, "combo_sam_style") else "Auto (AI)"
            style_str = "auto" if "Auto" in style_val else style_val
            use_const = bool(self.chk_sam_constraint.get()) if hasattr(self, "chk_sam_constraint") else True

            def _task():
                invert_flag = getattr(self, "sam_mask_inverted", False)
                motif = sam_segmenter.segment_at_point_or_box(
                    self.sam_current_pil,
                    point=(px, py),
                    style=style_str,
                    use_style_constraint=use_const,
                    invert=invert_flag
                )
                if motif:
                    self.sam_detected_motifs.append(motif)
                    for idx, m in enumerate(self.sam_detected_motifs):
                        m["id"] = idx

                    overlay = sam_segmenter.draw_segmentation_overlay(self.sam_current_pil, self.sam_detected_motifs, selected_id=motif["id"])
                    self.sam_overlay_pil = overlay

                    def _apply():
                        self.sam_struct_mask_pil = None
                        self._update_struct_mask_preview()
                        self._redraw_sam_canvas()
                        self._render_all_motif_cards()
                        self.lbl_sam_status.configure(text=f"✨ Figur #{motif['id']+1} per Zauberstab erfolgreich segmentiert!")
                    self.after(0, _apply)
                else:
                    def _fail():
                        self.lbl_sam_status.configure(text="⚠️ Keine Figur an dieser Stelle erkannt. Versuche mit dem Stift zu zeichnen oder eine Box aufzuziehen.")
                    self.after(0, _fail)

            threading.Thread(target=_task, daemon=True).start()
            return

        # Default fallback: Select clicked motif
        if self.sam_detected_motifs:
            for m in self.sam_detected_motifs:
                if 0 <= py < m["mask"].shape[0] and 0 <= px < m["mask"].shape[1]:
                    if m["mask"][py, px]:
                        label_str = m.get("label") or f"Motiv #{m['id']+1}"
                        opt_str = f"{label_str} ({m.get('area_pct', 0)}% Fläche)"
                        self.combo_motifs.set(opt_str)
                        self._on_select_motif(opt_str)
                        break

    def _on_change_sam_view_mode(self, mode_val: str):
        self._redraw_sam_canvas()

    def _on_change_sam_style(self, choice: str = ""):
        self.sam_struct_mask_pil = None
        self._update_struct_mask_preview()
        if hasattr(self, "seg_view_mode") and "Maske" in self.seg_view_mode.get():
            self._redraw_sam_canvas()

    def _toggle_invert_mask(self):
        self.sam_mask_inverted = not getattr(self, "sam_mask_inverted", False)
        self.sam_struct_mask_pil = None
        self._update_struct_mask_preview()
        if hasattr(self, "seg_view_mode") and "Maske" in self.seg_view_mode.get():
            self._redraw_sam_canvas()
        state = "Invertiert (Hintergrund)" if self.sam_mask_inverted else "Normal (Figuren)"
        self.lbl_sam_status.configure(text=f"🔄 Maskierung umgekehrt: {state}")

    def _get_or_create_struct_mask_visual(self) -> Optional[Image.Image]:
        if self.sam_current_pil is None:
            return None
        if self.sam_struct_mask_pil is not None:
            return self.sam_struct_mask_pil

        style_val = self.combo_sam_style.get() if hasattr(self, "combo_sam_style") else "Auto (AI)"
        style_str = "auto" if "Auto" in style_val else style_val
        invert_flag = getattr(self, "sam_mask_inverted", False)

        # Prioritize pixel-perfect AI segmented motifs if already detected
        if self.sam_detected_motifs:
            combined = np.zeros((self.sam_current_pil.height, self.sam_current_pil.width), dtype=np.uint8)
            for m in self.sam_detected_motifs:
                if "mask" in m:
                    combined[m["mask"]] = 255
            raw_mask = combined if not invert_flag else cv2.bitwise_not(combined)
        else:
            raw_mask = sam_segmenter.extract_structure_mask(self.sam_current_pil, style=style_str, invert=invert_flag)
        
        h, w = raw_mask.shape
        img_np = np.array(self.sam_current_pil.convert("RGB"))
        
        # Highlight detected figures with a luminous turquoise overlay on the real photograph
        vis = img_np.copy()
        fg_indices = raw_mask > 0
        if np.any(fg_indices):
            glow = np.full_like(vis, [0, 220, 255]) # Radiant cyan/turquoise
            vis[fg_indices] = cv2.addWeighted(img_np[fg_indices], 0.35, glow[fg_indices], 0.65, 0)
            # Add thin white boundary contour around figures
            contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 2)
        
        self.sam_struct_mask_pil = Image.fromarray(vis)
        return self.sam_struct_mask_pil

    def _update_struct_mask_preview(self):
        if self.sam_current_pil is None:
            return
        struct_img = self._get_or_create_struct_mask_visual()
        if struct_img is not None and hasattr(self, "canvas_struct_mask"):
            self._display_image_on_label(struct_img, self.canvas_struct_mask, (160, 160))

    def _redraw_sam_canvas(self, event=None):
        mode = self.seg_view_mode.get() if hasattr(self, "seg_view_mode") else "🎭 Figuren"
        
        if mode == "🖼️ Original":
            target_img = self.sam_current_pil
        elif "Maske" in mode:
            target_img = self._get_or_create_struct_mask_visual()
        else: # "🎭 Figuren"
            target_img = self.sam_overlay_pil if self.sam_overlay_pil is not None else self.sam_current_pil

        canvas_w = max(1, self.canvas_sam.winfo_width())
        canvas_h = max(1, self.canvas_sam.winfo_height())
        is_dark = hasattr(self, "switch_dark") and self.switch_dark.get() == 1

        if target_img is None:
            self.canvas_sam.delete("all")
            accent = "#c084fc" if is_dark else "#7c3aed"
            muted = "#94a3b8" if is_dark else "#64748b"
            card_bg = "#1e293b" if is_dark else "#f8fafc"
            border_col = "#334155" if is_dark else "#e2e8f0"

            rw, rh = 440, 160
            rx0, ry0 = (canvas_w - rw) // 2, (canvas_h - rh) // 2
            rx1, ry1 = rx0 + rw, ry0 + rh
            self.canvas_sam.create_rectangle(rx0, ry0, rx1, ry1, fill=card_bg, outline=border_col, width=1, tags="empty_card")

            self.canvas_sam.create_text(
                canvas_w // 2, canvas_h // 2 - 35,
                text="✨", font=("Segoe UI Emoji", 34), fill=accent, tags="empty_card"
            )
            self.canvas_sam.create_text(
                canvas_w // 2, canvas_h // 2 + 10,
                text="Kein Bild zur Segmentierung geladen", font=("Segoe UI", 13, "bold"), fill=accent, tags="empty_card"
            )
            self.canvas_sam.create_text(
                canvas_w // 2, canvas_h // 2 + 38,
                text="Wähle ein Bild aus 'pics/' oder sende eine 3D-Abrollung aus dem 3D-Studio",
                font=("Segoe UI", 10), fill=muted, tags="empty_card"
            )
            return

        self.canvas_sam.delete("empty_card")
        zoom = float(self.slider_sam_zoom.get())
        iw, ih = target_img.size

        base_scale = min(canvas_w / max(1, iw), canvas_h / max(1, ih))
        disp_w = max(1, int(iw * base_scale * zoom))
        disp_h = max(1, int(ih * base_scale * zoom))

        cx = canvas_w // 2 + self._sam_pan_x
        cy = canvas_h // 2 + self._sam_pan_y

        resized = target_img.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
        self._sam_temp_img = ImageTk.PhotoImage(resized)

        self.canvas_sam.delete("all")
        self.canvas_sam.create_image(cx, cy, image=self._sam_temp_img, anchor="center")

    def _load_sam_image_path(self, path: str):
        self.sam_current_path = path
        self.sam_current_pil = Image.open(path).convert("RGB")
        self.sam_overlay_pil = None
        self.sam_struct_mask_pil = None
        self.sam_mask_inverted = False
        self.sam_detected_motifs = []

        # Auto-detect style from metadata or physical contrast
        txt_path = os.path.splitext(path)[0] + ".txt"
        style_set = False
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read().lower()
                if "schwarzfigurig" in txt:
                    self.combo_sam_style.set("Attisch Schwarzfigurig")
                    style_set = True
                elif "rotfigurig" in txt:
                    self.combo_sam_style.set("Attisch Rotfigurig")
                    style_set = True
            except Exception:
                pass
        
        if not style_set:
            try:
                tech = sam_segmenter.detect_ceramic_technique(self.sam_current_pil)
                self.combo_sam_style.set(tech)
            except Exception:
                self.combo_sam_style.set("Auto (AI)")

        self._reset_sam_zoom()
        self._render_all_motif_cards()
        self._update_struct_mask_preview()
        self._redraw_sam_canvas()

    def _run_sam_segmentation(self):
        if self.sam_current_pil is None:
            messagebox.showwarning("Kein Bild", "Bitte laden Sie zuerst ein Vasenbild.")
            return

        style_val = self.combo_sam_style.get() if hasattr(self, "combo_sam_style") else "Auto (AI)"
        style_str = "auto" if "Auto" in style_val else style_val
        use_const = bool(self.chk_sam_constraint.get()) if hasattr(self, "chk_sam_constraint") else True

        self.lbl_sam_status.configure(text=f"⏳ Führe archäologische Segmentierung durch ({style_val}, Tongrund-Sperre: {use_const})...")

        def _task():
            invert_flag = getattr(self, "sam_mask_inverted", False)
            motifs = sam_segmenter.segment_all(
                self.sam_current_pil,
                style=style_str,
                use_style_constraint=use_const,
                invert=invert_flag
            )
            self.sam_detected_motifs = motifs

            if motifs:
                overlay = sam_segmenter.draw_segmentation_overlay(self.sam_current_pil, motifs, selected_id=0)
                self.sam_overlay_pil = overlay

                def _apply_on_main():
                    self.sam_struct_mask_pil = None
                    self._update_struct_mask_preview()
                    self._redraw_sam_canvas()
                    self._render_all_motif_cards()
                    self.lbl_sam_status.configure(text=f"✨ {len(motifs)} Figuren & Sub-Komponenten erfolgreich extrahiert!")

                self.after(0, _apply_on_main)
            else:
                def _fail():
                    self._render_all_motif_cards()
                    self.lbl_sam_status.configure(text="⚠️ Keine Figuren erkannt. Nutzen Sie den Zauberstab oder die Box-Auswahl.")
                self.after(0, _fail)

        threading.Thread(target=_task, daemon=True).start()

    def _render_all_motif_cards(self):
        """Renders stacked, zoomable motif cards for all detected figures."""
        if not hasattr(self, "scroll_motifs"):
            return

        for w in self.scroll_motifs.winfo_children():
            w.destroy()
        self._motif_cards.clear()

        count = len(self.sam_detected_motifs)
        if hasattr(self, "lbl_motifs_title"):
            self.lbl_motifs_title.configure(text=f"🏷️ Motiv-Prüfung ({count} Figuren)")

        if count == 0:
            lbl_empty = ctk.CTkLabel(
                self.scroll_motifs,
                text="💡 Keine Motive geladen.\nNutze den Zauberstab (Klick auf Figur), Box-Auswahl\noder '🚀 Alle Figuren automatisch', um Figuren zu extrahieren!",
                font=ctk.CTkFont(size=12),
                text_color=THEME["text_muted"],
                justify="center"
            )
            lbl_empty.pack(pady=40)
        else:
            for idx, m in enumerate(self.sam_detected_motifs):
                card = MotifCard(self.scroll_motifs, m, idx, self)
                card.pack(fill="x", pady=6, padx=4)
                self._motif_cards.append(card)

        # Update floating popup window if open
        if getattr(self, "_motif_window_toplevel", None) and self._motif_window_toplevel.winfo_exists():
            try:
                self._motif_window_toplevel._populate()
            except Exception:
                pass

    def _apply_all_ai_suggestions(self):
        """Applies AI predicted labels on all motif cards with 1 click."""
        if not self._motif_cards:
            self.show_toast("Keine Motive zum Übernehmen vorhanden.", "warning")
            return
        for card in self._motif_cards:
            ai_lbl, _, _ = card._predict_ai_motif()
            card._set_label(ai_lbl)
        if getattr(self, "_motif_window_toplevel", None) and self._motif_window_toplevel.winfo_exists():
            for card in self._motif_window_toplevel._cards:
                ai_lbl, _, _ = card._predict_ai_motif()
                card._set_label(ai_lbl)
        self.show_toast("Alle KI-Vorschläge auf die Figuren angewendet!", "success")

    def _save_all_verified_motifs(self):
        """Saves all currently configured motifs to the training pool."""
        if not self._motif_cards:
            self.show_toast("Keine Motive zum Speichern vorhanden.", "warning")
            return
        for card in self._motif_cards:
            card._on_save()
        self.lbl_pool_counter.configure(text=f"📦 Pool: {len(self.training_dataset)}")
        self.show_toast(f"{len(self._motif_cards)} Figuren im Trainings-Pool gespeichert!", "success")

    def _on_skos_vocab_changed(self):
        """Called automatically whenever the SKOS vocabulary is updated or saved."""
        def _update():
            for card in getattr(self, "_motif_cards", []):
                try:
                    card.refresh_skos_dropdown()
                except Exception:
                    pass
            if getattr(self, "_motif_window_toplevel", None) and self._motif_window_toplevel.winfo_exists():
                try:
                    self._motif_window_toplevel.refresh_dropdowns()
                except Exception:
                    pass
            if hasattr(self, "combo_verify"):
                try:
                    labels = skos_client.get_all_labels_for_selection()
                    self.combo_verify.configure(values=labels[:150])
                except Exception:
                    pass
        self.after(0, _update)

    def _open_multi_screen_motif_window(self):
        """Opens floating multi-screen window for Motif Verification."""
        if self._motif_window_toplevel is None or not self._motif_window_toplevel.winfo_exists():
            self._motif_window_toplevel = MotifVerificationWindow(self, app=self)
        else:
            self._motif_window_toplevel.lift()
            self._motif_window_toplevel.focus_force()

    def _open_multi_screen_skos_window(self):
        """Opens floating multi-screen window for the SKOS Editor."""
        if self._skos_window_toplevel is None or not self._skos_window_toplevel.winfo_exists():
            self._skos_window_toplevel = SKOSEditorWindow(self, app=self)
        else:
            self._skos_window_toplevel.lift()
            self._skos_window_toplevel.focus_force()

    def open_skos_editor_with_term(self, term: str):
        """Cross-link: Jumps to or opens SKOS Editor with a given term pre-filled."""
        self.select_view("skos")
        if hasattr(self, "skos_editor_frame"):
            c = skos_client.get_concept_by_label(term)
            if c:
                self.skos_editor_frame.load_concept_into_form(c)
                self.skos_editor_frame.entry_search.delete(0, "end")
                self.skos_editor_frame.entry_search.insert(0, term)
                self.skos_editor_frame._refresh_concepts_table()
            else:
                self.skos_editor_frame._clear_form()
                self.skos_editor_frame.ent_label_de.insert(0, term)
        self.show_toast(f"SKOS-Editor geöffnet für: '{term}'", "info")

    def _on_select_motif(self, choice: str):
        pass

    def _jump_to_skos_thesaurus(self, term: Optional[str] = None):
        self.open_skos_editor_with_term(term or "Delphinreiter")

    def _save_training_entry(self):
        self._save_all_verified_motifs()

    def _export_train_zip(self):
        if not self.training_dataset:
            self.show_toast("Trainings-Pool ist leer. Bitte speichere zuerst Annotationen.", "warning")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".zip", filetypes=[("ZIP Archive", "*.zip")])
        if save_path:
            zip_bytes = DatasetExporter.export_training_zip(self.training_dataset)
            with open(save_path, "wb") as f:
                f.write(zip_bytes)
            self.show_toast(f"Trainings-Paket exportiert: {os.path.basename(save_path)}", "success")

    def _export_train_coco(self):
        if not self.training_dataset:
            self.show_toast("Trainings-Pool ist leer.", "warning")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if save_path:
            coco_dict = DatasetExporter.build_coco_dataset(self.training_dataset)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(coco_dict, f, indent=2, ensure_ascii=False)
            self.show_toast(f"COCO JSON exportiert: {os.path.basename(save_path)}", "success")

    def _clear_train_pool(self):
        self.training_dataset = []
        if hasattr(self, "lbl_pool_counter"):
            self.lbl_pool_counter.configure(text="📦 Pool: 0")
        self.show_toast("Der Trainings-Pool wurde zurückgesetzt.", "info")
        self.training_dataset = []
        self.show_toast("Der Trainings-Pool wurde zurückgesetzt.", "info")

    # ---------------------------------------------------------
    # VIEW 3: 2D-Einzelbild & Szenen-Explorer
    # ---------------------------------------------------------
    def _build_single_view(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)

        # Header Bar
        top_bar = ctk.CTkFrame(frame, height=50, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        top_bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

        lbl = ctk.CTkLabel(top_bar, text="🔍 2D-Einzelbild-Klassifikator & Szenen-Detail-Explorer", font=ctk.CTkFont(size=14, weight="bold"), text_color=("#0f172a", "#f8fafc"))
        lbl.pack(side="left", padx=15, pady=8)

        btn_open = ctk.CTkButton(top_bar, text="📂 Bild laden", command=self._open_single_image, width=130, fg_color=("#0284c7", "#334155"), hover_color=("#0369a1", "#475569"))
        btn_open.pack(side="right", padx=10, pady=8)

        self.combo_pics_single = ctk.CTkComboBox(top_bar, values=self._get_pic_list(), command=self._on_select_pic_single, width=220)
        self.combo_pics_single.pack(side="right", padx=5, pady=8)

        # Left Column: Image Canvas
        col_left = ctk.CTkFrame(frame, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        col_left.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        col_left.grid_rowconfigure(1, weight=1)
        col_left.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(col_left, text="🖼️ Gesamtes Vasenbild", font=ctk.CTkFont(size=13, weight="bold"), text_color=("#0284c7", "#38bdf8")).grid(row=0, column=0, padx=10, pady=8, sticky="w")

        self.canvas_single = ctk.CTkLabel(col_left, text="Lade Bild...", fg_color=("#f1f5f9", "#0f172a"), text_color=("#0f172a", "#f8fafc"), corner_radius=8)
        self.canvas_single.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Right Column: Classification & SKOS Cards
        col_right = ctk.CTkFrame(frame, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        col_right.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        col_right.grid_rowconfigure(2, weight=1)
        col_right.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(col_right, text="🎯 Globale Vorhersage & SKOS-Erschließung", font=ctk.CTkFont(size=13, weight="bold"), text_color=("#d97706", "#f59e0b")).grid(row=0, column=0, padx=10, pady=8, sticky="w")

        self.lbl_single_results = ctk.CTkLabel(
            col_right, text="Analysiere Bild...", font=ctk.CTkFont(size=12),
            justify="left", fg_color=("#f1f5f9", "#0f172a"), text_color=("#0f172a", "#f8fafc"), corner_radius=8, padx=12, pady=10
        )
        self.lbl_single_results.grid(row=1, column=0, padx=10, pady=5, sticky="ew")

        # Scene detail frame
        scene_frame = ctk.CTkFrame(col_right, fg_color=("#f1f5f9", "#0f172a"), corner_radius=8)
        scene_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nsew")
        scene_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(scene_frame, text="🎭 Szenen- & Figuren-Detailanalyse", font=ctk.CTkFont(size=12, weight="bold"), text_color=("#d97706", "#fbbf24")).pack(padx=10, pady=8, anchor="w")

        self.lbl_scene_detail = ctk.CTkLabel(
            scene_frame, text="Muster-Szenenerkennung...", font=ctk.CTkFont(size=12), justify="left", text_color=("#0f172a", "#f8fafc")
        )
        self.lbl_scene_detail.pack(padx=10, pady=5, anchor="w")

        # YOLO Figure Detections Label
        self.lbl_yolo_detections = ctk.CTkLabel(
            scene_frame, text="", font=ctk.CTkFont(size=11), justify="left", text_color=("#0f172a", "#f8fafc")
        )
        self.lbl_yolo_detections.pack(padx=10, pady=(0, 6), anchor="w")

        # Action Buttons frame
        actions_frame = ctk.CTkFrame(col_right, fg_color="transparent")
        actions_frame.grid(row=3, column=0, padx=10, pady=8, sticky="ew")
        actions_frame.grid_columnconfigure(0, weight=1)
        actions_frame.grid_columnconfigure(1, weight=1)

        btn_detect_yolo = ctk.CTkButton(
            actions_frame, text="🎯 Gottheiten lokalisieren (YOLOv8)",
            fg_color="#d97706", hover_color="#b45309", text_color="#ffffff", font=ctk.CTkFont(size=12, weight="bold"), height=36,
            command=self._run_figure_detection
        )
        btn_detect_yolo.grid(row=0, column=0, padx=(0, 4), pady=(0, 4), sticky="ew")

        btn_send_sam = ctk.CTkButton(
            actions_frame, text="✨ In SAM-Studio öffnen",
            fg_color="#7c3aed", hover_color="#6d28d9", text_color="#ffffff", font=ctk.CTkFont(size=12, weight="bold"), height=36,
            command=self._send_single_to_sam
        )
        btn_send_sam.grid(row=0, column=1, padx=(4, 0), pady=(0, 4), sticky="ew")

        btn_skos_single = ctk.CTkButton(
            actions_frame, text="🌳 Im HECTOR SKOS-Explorer nachschlagen",
            fg_color=THEME["secondary_btn"], hover_color=THEME["secondary_btn_hover"], text_color=THEME["text_primary"],
            font=ctk.CTkFont(size=11, weight="bold"), height=30,
            command=self._jump_to_skos_from_single
        )
        btn_skos_single.grid(row=1, column=0, columnspan=2, pady=(4, 0), sticky="ew")

        return frame

    def _jump_to_skos_from_single(self):
        """Jump from 2D single view to SKOS explorer with predicted shape or scene."""
        term = ""
        if hasattr(self, "lbl_single_results"):
            t = self.lbl_single_results.cget("text")
            if "Gefäßform:" in t:
                term = t.split("Gefäßform:")[1].split("\n")[0].split("(")[0].strip()
        if not term and hasattr(self, "lbl_scene_detail"):
            t = self.lbl_scene_detail.cget("text")
            if "Motiv / Szene:" in t:
                term = t.split("Motiv / Szene:")[1].split("\n")[0].split("(")[0].strip()
        if term:
            self.select_view("skos")
            if hasattr(self, "entry_skos"):
                self.entry_skos.delete(0, "end")
                self.entry_skos.insert(0, term)
                self._search_skos()
                self.show_toast(f"HECTOR Thesaurus geöffnet: '{term}'", "info")
        else:
            self.show_toast("Keine Vorhersage für den Thesaurus vorhanden.", "warning")

    def _run_figure_detection(self):
        if self.single_pil is None:
            self.show_toast("Bitte laden Sie zuerst ein 2D-Vasenbild.", "warning")
            return

        self.lbl_yolo_detections.configure(text="⏳ Führe YOLOv8 Figuren-Detektion auf GPU aus...")

        def _task():
            detections = figure_detector.detect(self.single_pil, conf_threshold=0.18)
            if not detections:
                self.after(0, lambda: self.lbl_yolo_detections.configure(
                    text="ℹ️ Keine Einzelfiguren über dem Schwellenwert (18%) erkannt."
                ))
                return

            annotated_img = figure_detector.draw_detections(self.single_pil, detections)
            
            lines = ["🎯 Erkannte Gottheiten auf der Vase (YOLOv8):"]
            for d in detections:
                lines.append(f"  • {d['spatial_description']}: {d['confidence']*100:.1f}% Konfidenz")
                lines.append(f"    Box: {d['bbox_xyxy']} | URI: {d['skos_uri']}")

            det_text = "\n".join(lines)
            
            def _update_ui():
                self.lbl_yolo_detections.configure(text=det_text)
                self._display_image_on_label(annotated_img, self.canvas_single, (500, 440))
                self.show_toast(f"YOLOv8: {len(detections)} Gottheiten lokalisiert!", "success")

            self.after(0, _update_ui)

        threading.Thread(target=_task, daemon=True).start()

    def _send_single_to_sam(self):
        if self.single_pil is None:
            self.show_toast("Bitte laden Sie zuerst ein 2D-Bild.", "warning")
            return

        path = getattr(self, "single_current_path", None)
        if not path or not os.path.exists(path):
            out_dir = os.path.join(BASE_DIR, "pics")
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, "2d_single_export.png")
            self.single_pil.save(path)

        norm_path = os.path.normpath(path)
        self.select_view("sam")

        existing_norms = [os.path.normpath(p) for p in self.sam_image_paths]
        if norm_path not in existing_norms:
            self.sam_image_paths.insert(0, norm_path)
            self.sam_current_idx = 0
        else:
            self.sam_current_idx = existing_norms.index(norm_path)

        self._refresh_sam_image_combo()
        self._load_sam_image_path(norm_path)
        self.show_toast("Bild an SAM-Studio übertragen!", "success")

    def _load_default_single_image(self):
        pics = self._get_pic_list()
        if pics:
            self._load_single_image_path(os.path.join(PICS_DIR, pics[0]))

    def _on_select_pic_single(self, choice: str):
        self._load_single_image_path(os.path.join(PICS_DIR, choice))

    def _open_single_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        if path:
            self._load_single_image_path(path)

    def _load_single_image_path(self, path: str):
        self.single_current_path = path
        self.single_pil = Image.open(path).convert("RGB")
        self._display_image_on_label(self.single_pil, self.canvas_single, (500, 440))
        self.lbl_yolo_detections.configure(text="")

        def _task():
            pred = classifier.predict(self.single_pil)
            shape = pred["predicted_shape"]
            s_conf = pred["shape_confidence"]
            ware = pred["predicted_ware"]
            w_conf = pred["ware_confidence"]

            res = skos_client.resolve_prediction(shape, ware)
            sc = res["shape"]
            wc = res["ware"]

            # Scene prediction
            sc_pred = classifier.predict_scene(self.single_pil)
            scene_lbl = sc_pred["predicted_scene"]
            scene_conf = sc_pred["scene_confidence"]
            scene_skos = skos_client.get_concept_by_label(scene_lbl)

            myth_info = ""
            if pred.get("predicted_mythology"):
                m_name = pred["predicted_mythology"]
                m_conf = pred["mythology_confidence"]
                top_m = ", ".join([f"{m['label']} ({m['confidence']*100:.0f}%)" for m in pred.get("top_mythologies", [])[:4]])
                myth_info = f"\n\n⚡ Erkannte Gottheit: {m_name} ({m_conf*100:.1f}%)\n   Rangfolge: {top_m}"

            text_glob = (
                f"🏺 Gefäßform: {shape} ({s_conf*100:.1f}% Konfidenz)\n"
                f"🔗 SKOS Shape URI: {sc.uri if sc else 'N/A'}\n\n"
                f"🎨 Ware / Stil: {ware} ({w_conf*100:.1f}% Konfidenz)\n"
                f"🔗 SKOS Ware URI: {wc.uri if wc else 'N/A'}"
                f"{myth_info}"
            )
            self.lbl_single_results.configure(text=text_glob)

            text_scene = (
                f"🎭 Dominantes Motiv / Szene: {scene_lbl} ({scene_conf*100:.1f}% Konfidenz)\n"
                f"🏛️ SKOS Thesaurus: {scene_skos.pref_label if scene_skos else scene_lbl}\n"
                f"🔗 URI: {scene_skos.uri if scene_skos else 'N/A'}\n"
                f"📖 Kategorie: {scene_skos.category if scene_skos else 'Ikonographie'}"
            )
            self.lbl_scene_detail.configure(text=text_scene)

        threading.Thread(target=_task, daemon=True).start()

    # ---------------------------------------------------------
    # VIEW 4: Batch-Scanner & Gesamtkatalog
    # ---------------------------------------------------------
    def _build_batch_view(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        # Header Bar
        top_bar = ctk.CTkFrame(frame, height=50, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        top_bar.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        lbl = ctk.CTkLabel(top_bar, text="📂 Batch-Scanner & Gesamtkatalog für Bildersammlungen", font=ctk.CTkFont(size=14, weight="bold"), text_color=("#0f172a", "#f8fafc"))
        lbl.pack(side="left", padx=15, pady=8)

        btn_scan = ctk.CTkButton(top_bar, text="🚀 'pics/' Ordner scannen", command=self._run_batch_scan, width=190, fg_color="#0284c7", hover_color="#0369a1", text_color="#ffffff")
        btn_scan.pack(side="right", padx=10, pady=8)

        # Main Table Area
        tbl_frame = ctk.CTkFrame(frame, fg_color=("#ffffff", "#1e293b"), corner_radius=8)
        tbl_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        tbl_frame.grid_rowconfigure(1, weight=1)
        tbl_frame.grid_columnconfigure(0, weight=1)

        self.progress_batch = ctk.CTkProgressBar(tbl_frame)
        self.progress_batch.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
        self.progress_batch.set(0)

        # Treeview Table
        tree_scroll = ttk.Scrollbar(tbl_frame)
        tree_scroll.grid(row=1, column=1, sticky="ns")

        self.tree_catalog = ttk.Treeview(
            tbl_frame,
            columns=("ID", "Form", "Ware", "Szene", "SKOS Shape URI"),
            show="headings",
            yscrollcommand=tree_scroll.set
        )
        tree_scroll.config(command=self.tree_catalog.yview)

        for col, title in [("ID", "Objekt-ID"), ("Form", "Gefäßform"), ("Ware", "Warenart"), ("Szene", "Erkannte Szene"), ("SKOS Shape URI", "HECTOR SKOS URI")]:
            self.tree_catalog.heading(col, text=title)
            self.tree_catalog.column(col, width=180)

        self.tree_catalog.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Export Buttons
        btn_box = ctk.CTkFrame(tbl_frame, fg_color="transparent")
        btn_box.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        ctk.CTkButton(btn_box, text="📥 JSON-LD Export", fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._export_batch("jsonld")).pack(side="left", padx=5)
        ctk.CTkButton(btn_box, text="📥 RDF Turtle (.ttl) Export", fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._export_batch("turtle")).pack(side="left", padx=5)
        ctk.CTkButton(btn_box, text="📥 CSV Export", fg_color=("#e2e8f0", "#334155"), hover_color=("#cbd5e1", "#475569"), text_color=("#0f172a", "#f8fafc"), command=lambda: self._export_batch("csv")).pack(side="left", padx=5)

        return frame

    def _run_batch_scan(self):
        def _task():
            records = load_directory_metadata(PICS_DIR)
            catalog = []
            for item in self.tree_catalog.get_children():
                self.tree_catalog.delete(item)

            for i, meta in enumerate(records):
                self.progress_batch.set((i + 1) / len(records))
                shape = meta.shape
                ware = meta.ware
                scene_lbl = "N/A"

                if meta.image_path and os.path.exists(meta.image_path):
                    try:
                        p = classifier.predict(meta.image_path)
                        if not shape: shape = p["predicted_shape"]
                        if not ware: ware = p["predicted_ware"]
                        sc_p = classifier.predict_scene(meta.image_path)
                        scene_lbl = sc_p["predicted_scene"]
                    except Exception:
                        pass

                res = skos_client.resolve_prediction(shape, ware)
                sc, wc = res["shape"], res["ware"]

                row = meta.to_dict()
                row["shape_label"] = sc.pref_label if sc else shape
                row["shape_uri"] = sc.uri if sc else ""
                row["ware_label"] = wc.pref_label if wc else ware
                row["ware_uri"] = wc.uri if wc else ""
                row["scenes"] = [{"label": scene_lbl, "uri": skos_client.get_concept_by_label(scene_lbl).uri if skos_client.get_concept_by_label(scene_lbl) else ""}]
                catalog.append(row)

                self.tree_catalog.insert("", "end", values=(
                    row.get("id", ""),
                    row.get("shape_label", ""),
                    row.get("ware_label", ""),
                    scene_lbl,
                    row.get("shape_uri", "")
                ))

            self.catalog_items = catalog
            self.show_toast(f"Scan abgeschlossen: {len(catalog)} Vasen klassifiziert!", "success")

        threading.Thread(target=_task, daemon=True).start()

    def _export_batch(self, fmt: str):
        if not self.catalog_items:
            self.show_toast("Katalog ist leer. Bitte führe zuerst einen Ordner-Scan durch.", "warning")
            return

        ext = f".{fmt}" if fmt != "turtle" else ".ttl"
        save_path = filedialog.asksaveasfilename(defaultextension=ext)
        if save_path:
            if fmt == "jsonld":
                data = LODExporter.to_jsonld(self.catalog_items)
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif fmt == "turtle":
                ttl = LODExporter.to_rdf_turtle(self.catalog_items)
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(ttl)
            elif fmt == "csv":
                LODExporter.export_csv(self.catalog_items, save_path)

            self.show_toast(f"Gesamtkatalog exportiert: {os.path.basename(save_path)}", "success")

    # ---------------------------------------------------------
    # VIEW 5: HECTOR SKOS-Editor & Thesaurus
    # ---------------------------------------------------------
    def _build_skos_view(self, parent) -> ctk.CTkFrame:
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self.skos_editor_frame = SKOSEditorFrame(frame, app=self, is_popup=False)
        self.skos_editor_frame.grid(row=0, column=0, sticky="nsew")

        return frame

    def _search_skos(self):
        pass


if __name__ == "__main__":
    app = LODVasesApp()
    app.mainloop()
