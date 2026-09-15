"""
Training Dataset Exporter: Generates COCO, VLM JSONL, and PNG cutout packages
from SAM-segmented and human-verified ancient vase annotations.
"""

import os
import io
import json
import zipfile
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional

class DatasetExporter:
    """
    Exports verified segmentations and labels into training dataset archives.
    """

    @staticmethod
    def build_coco_dataset(annotated_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Builds a COCO format dictionary for instance segmentation training.
        """
        categories_map = {}
        categories = []
        images = []
        annotations = []
        
        ann_id = 1
        img_id_map = {}
        
        for item in annotated_items:
            img_path = item.get("image_path", "vase.jpg")
            img_filename = os.path.basename(img_path)
            
            if img_filename not in img_id_map:
                current_img_id = len(images) + 1
                img_id_map[img_filename] = current_img_id
                
                img_w = item.get("img_w", 1024)
                img_h = item.get("img_h", 1024)
                
                images.append({
                    "id": current_img_id,
                    "file_name": img_filename,
                    "width": img_w,
                    "height": img_h
                })
            else:
                current_img_id = img_id_map[img_filename]

            # Categories
            label = item.get("label", "Unknown")
            skos_uri = item.get("skos_uri", "")
            
            if label not in categories_map:
                cat_id = len(categories) + 1
                categories_map[label] = cat_id
                categories.append({
                    "id": cat_id,
                    "name": label,
                    "supercategory": item.get("category", "Ikonographie"),
                    "skos_uri": skos_uri
                })
            else:
                cat_id = categories_map[label]

            # Segmentation polygon
            polygon = item.get("polygon", [])
            flat_seg = []
            if polygon:
                flat_seg = [float(coord) for pt in polygon for coord in pt]

            bbox = item.get("bbox", [0, 0, 100, 100]) # [x, y, w, h]
            area = float(item.get("area", bbox[2] * bbox[3]))

            annotations.append({
                "id": ann_id,
                "image_id": current_img_id,
                "category_id": cat_id,
                "segmentation": [flat_seg] if flat_seg else [],
                "area": area,
                "bbox": [float(b) for b in bbox],
                "iscrowd": 0,
                "confidence": float(item.get("confidence", 1.0)),
                "skos_uri": skos_uri
            })
            ann_id += 1

        return {
            "info": {
                "description": "BCDH LODVases Ancient Greek Pottery Segmented Iconography Dataset",
                "version": "1.0",
                "year": 2026,
                "contributor": "BCDH Universität Bonn",
                "url": "https://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/de/"
            },
            "licenses": [{"id": 1, "name": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/"}],
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

    @staticmethod
    def export_training_zip(verified_annotations: List[Dict[str, Any]]) -> bytes:
        """
        Packages COCO JSON, transparent cutouts, and VLM JSONL into a downloadable ZIP archive.
        """
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            # 1. COCO JSON
            coco_dict = DatasetExporter.build_coco_dataset(verified_annotations)
            coco_json_str = json.dumps(coco_dict, indent=2, ensure_ascii=False)
            zf.writestr("coco/annotations_instances.json", coco_json_str)

            # 2. VLM / Vision-Language Fine-Tuning JSONL (LLaVA format)
            vlm_lines = []
            categories_unique = sorted(list(set(item.get("label", "Unknown") for item in verified_annotations)))
            cat_to_idx = {c: i for i, c in enumerate(categories_unique)}

            for idx, item in enumerate(verified_annotations):
                cutout_img = item.get("cutout")
                label = item.get("label", "Figur")
                skos = item.get("skos_uri", "")
                cutout_name = f"cutouts/motif_{idx+1:04d}_{label.replace('/', '_').replace(' ', '_')}.png"

                # Save cutout PNG
                if cutout_img is not None:
                    c_buf = io.BytesIO()
                    cutout_img.save(c_buf, format="PNG")
                    zf.writestr(cutout_name, c_buf.getvalue())

                # VLM conversation format
                vlm_entry = {
                    "id": f"lodvases_motif_{idx+1}",
                    "image": cutout_name,
                    "conversations": [
                        {
                            "from": "human",
                            "value": "<image>\nWelches antike Motiv / Figur ist in diesem freigestellten Ausschnitt der Vase dargestellt?"
                        },
                        {
                            "from": "gpt",
                            "value": f"In diesem Ausschnitt ist **{label}** dargestellt (SKOS-Thesaurus Konzept: {skos})."
                        }
                    ]
                }
                vlm_lines.append(json.dumps(vlm_entry, ensure_ascii=False))

            zf.writestr("vlm/dataset_train.jsonl", "\n".join(vlm_lines))

            # Readme & Documentation
            readme_text = f"""# BCDH LODVases Training Dataset Archive
Erstellt mit SAM (Segment Anything) & BCDH SKOS Thesaurus.

## Inhalt des Datensatzes:
- `coco/annotations_instances.json`: COCO Instance Segmentation Annotationen (Masken & Bounding Boxes).
- `vlm/dataset_train.jsonl`: LLaVA / VLM Multimodal Fine-Tuning Format.
- `cutouts/*.png`: Freigestellte Figuren mit Alphakanal-Transparenz.

## Enthaltene Klassen:
{chr(10).join([f"- {c} ({cat_to_idx[c]})" for c in categories_unique])}
"""
            zf.writestr("README.md", readme_text)

        return zip_buffer.getvalue()
