"""
Metadata Parser for Ancient Vase records.
Parses key-value text files associated with vase images.
"""

import os
import glob
import re
from typing import Dict, Any, List, Optional


def _clean_brackets(val: str) -> str:
    """Removes bracketed annotations like [SKOS: ...] or [Kerameikos: ...] from metadata values."""
    if not val:
        return ""
    return re.sub(r"\s*\[.*?\]", "", val).strip()


class VaseMetadata:
    def __init__(
        self,
        file_path: str,
        image_path: Optional[str] = None,
        raw_fields: Optional[Dict[str, str]] = None
    ):
        self.file_path = file_path
        self.image_path = image_path or self._infer_image_path(file_path)
        self.fields = raw_fields or {}
        
    @property
    def id(self) -> str:
        base = os.path.basename(self.file_path)
        return os.path.splitext(base)[0]
    
    @property
    def shape(self) -> str:
        val = self.fields.get("Gefäßform", self.fields.get("Gefässform", ""))
        return _clean_brackets(val)

    @property
    def ware(self) -> str:
        # Check explicit ware or construct from region + description
        # often indicated by "Produktionsgebiet" + style (e.g. Attisch, rotfigurig)
        ware_val = self.fields.get("Ware", self.fields.get("Ware/Stil", ""))
        ware_val = _clean_brackets(ware_val)
        if not ware_val:
            prod = _clean_brackets(self.fields.get("Produktionsgebiet", ""))
            name = self.fields.get("Name/Bezeichnung", "")
            desc = self.fields.get("Beschreibung", "")
            # Heuristic detection from name or desc
            if "rotfigurig" in name.lower() or "rotfigurig" in desc.lower():
                ware_val = f"{prod} Rotfigurig" if prod else "Rotfigurig"
            elif "schwarzfigurig" in name.lower() or "schwarzfigurig" in desc.lower():
                ware_val = f"{prod} Schwarzfigurig" if prod else "Schwarzfigurig"
            elif "weißgrundig" in name.lower() or "weissgrundig" in desc.lower():
                ware_val = f"{prod} Weißgrundig" if prod else "Weißgrundig"
            elif prod:
                ware_val = prod
        return ware_val

    @property
    def production_region(self) -> str:
        return _clean_brackets(self.fields.get("Produktionsgebiet", ""))

    @property
    def material(self) -> str:
        return _clean_brackets(self.fields.get("Material", "Ton"))

    @property
    def artist(self) -> str:
        val = self.fields.get("Künstler/Werkstatt", self.fields.get("Kuenstler/Werkstatt", ""))
        return _clean_brackets(val)

    @property
    def dating(self) -> str:
        return self.fields.get("Datierung", "")

    @property
    def name(self) -> str:
        return self.fields.get("Name/Bezeichnung", "")

    @property
    def description(self) -> str:
        return self.fields.get("Beschreibung", "")

    @property
    def beazley_number(self) -> str:
        return self.fields.get("Beazley-Nummer", "")

    @property
    def location(self) -> str:
        return self.fields.get("Standort", "")

    @property
    def inventory_number(self) -> str:
        return self.fields.get("Inventar-Nr.", "")

    @property
    def findspot(self) -> str:
        return self.fields.get("Herkunftsort", "")

    @property
    def dimensions(self) -> str:
        return self.fields.get("Maße", self.fields.get("Masse", ""))

    @property
    def image_credit(self) -> str:
        return self.fields.get("Abbildungsnachweis", "")

    def _infer_image_path(self, txt_path: str) -> Optional[str]:
        base, _ = os.path.splitext(txt_path)
        for ext in [".jpg", ".jpeg", ".png", ".webp"]:
            candidate = base + ext
            if os.path.exists(candidate):
                return candidate
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "image_path": self.image_path,
            "shape": self.shape,
            "ware": self.ware,
            "production_region": self.production_region,
            "material": self.material,
            "artist": self.artist,
            "dating": self.dating,
            "name": self.name,
            "description": self.description,
            "beazley_number": self.beazley_number,
            "location": self.location,
            "inventory_number": self.inventory_number,
            "findspot": self.findspot,
            "dimensions": self.dimensions,
            "image_credit": self.image_credit,
            "raw_fields": self.fields
        }

    @classmethod
    def from_file(cls, txt_path: str) -> "VaseMetadata":
        fields = {}
        with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if ":" in line:
                    k, v = line.split(":", 1)
                    fields[k.strip()] = v.strip()
        return cls(file_path=txt_path, raw_fields=fields)


def load_directory_metadata(dir_path: str) -> List[VaseMetadata]:
    """
    Loads all .txt metadata files and associates them with corresponding images.
    Also handles images that might not have a .txt file yet.
    """
    results = []
    
    # 1. Process all existing .txt files
    txt_files = glob.glob(os.path.join(dir_path, "*.txt"))
    handled_stems = set()
    
    for txt_file in txt_files:
        stem = os.path.splitext(os.path.basename(txt_file))[0]
        handled_stems.add(stem)
        try:
            meta = VaseMetadata.from_file(txt_file)
            results.append(meta)
        except Exception as e:
            print(f"Error reading {txt_file}: {e}")
            
    # 2. Check for images without .txt files
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        for img_file in glob.glob(os.path.join(dir_path, ext)):
            stem = os.path.splitext(os.path.basename(img_file))[0]
            if stem not in handled_stems:
                handled_stems.add(stem)
                meta = VaseMetadata(
                    file_path=os.path.splitext(img_file)[0] + ".txt",
                    image_path=img_file,
                    raw_fields={}
                )
                results.append(meta)
                
    return results
