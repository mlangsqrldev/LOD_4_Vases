"""
Corpus Builder: Assembles balanced, multi-class mythological vase datasets
from BAPD (CARC Oxford) and maps them directly into the BCDH HECTOR SKOS vocabulary.
"""

import os
import re
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from .bapd_client import BAPDClient, BAPDRecord
from .skos_client import skos_client
from .lod_exporter import LODExporter
from .metadata_parser import VaseMetadata, load_directory_metadata
from .kerameikos_client import kerameikos_client

# Target mythological deities and their canonical German / BCDH SKOS representations
MYTHOLOGY_TARGETS: Dict[str, Dict[str, str]] = {
    "ATHENA": {
        "label": "Athena",
        "skos_uri": "https://hector.bcdh.uni-bonn.de/athena",
        "description": "Göttin der Weisheit, Kriegskunst und Stadtgöttin Athens"
    },
    "DIONYSOS": {
        "label": "Dionysos",
        "skos_uri": "https://hector.bcdh.uni-bonn.de/dionysus",
        "description": "Gott des Weines, der Trauben, der Freude und der Ekstase"
    },
    "HERAKLES": {
        "label": "Herakles",
        "skos_uri": "https://hector.bcdh.uni-bonn.de/heracles",
        "description": "Griechischer Nationalheld und Sohn des Zeus mit Keule und Löwenfell"
    },
    "APOLLO": {
        "label": "Apollon",
        "skos_uri": "https://hector.bcdh.uni-bonn.de/apollo",
        "description": "Gott des Lichts, der Musik, der Dichtung und der Weissagung"
    }
}

# Mapping English BAPD shapes to standard German archaeological terminology
SHAPE_MAPPING = {
    "CUP": "Kylix",
    "KYLIX": "Kylix",
    "AMPHORA": "Amphora",
    "AMPHORA, NECK": "Halsamphora",
    "AMPHORA, BELLY": "Bauchamphora",
    "PANATHENAIC AMPHORA": "Panathenäische Preisamphore",
    "HYDRIA": "Hydria",
    "LEKYTHOS": "Lekythos",
    "KRATER": "Krater",
    "BELL KRATER": "Glockenkrater",
    "CALYX KRATER": "Kelchkrater",
    "VOLUTE KRATER": "Volutenkrater",
    "COLUMN KRATER": "Kolonettenkrater",
    "PELIKE": "Pelike",
    "PSYKTER": "Psykter",
    "STAMNOS": "Stamnos",
    "SKYPHOS": "Skyphos",
    "KANTHAROS": "Kantharos",
    "OINOCHOE": "Oinochoe",
    "PYXIS": "Pyxis",
    "ARYBALLOS": "Aryballos",
    "ALABASTRON": "Alabastron",
    "DINOS": "Dinos",
    "LEBES": "Dinos",
    "RHYTON": "Rhyton",
    "PLATE": "Teller",
}

# Mapping Technique & Fabric to German ware terms
TECHNIQUE_MAPPING = {
    "RED-FIGURE": "Rotfigurig",
    "BLACK-FIGURE": "Schwarzfigurig",
    "WHITE-GROUND": "Weißgrundig",
    "SIX'S TECHNIQUE": "Six-Technik",
    "CORINTHIAN": "Korinthisch",
    "GEOMETRIC": "Geometrisch"
}


def normalize_shape(raw_shape: str) -> str:
    """Translates BAPD shape names to canonical German names."""
    if not raw_shape:
        return "Gefäß"
    upper = raw_shape.upper().strip()
    # Match longest, most specific terms first (e.g. 'AMPHORA, NECK' before 'AMPHORA')
    for bapd_term, de_term in sorted(SHAPE_MAPPING.items(), key=lambda x: len(x[0]), reverse=True):
        if bapd_term in upper:
            return de_term
    return raw_shape.capitalize()


def normalize_ware(fabric: str, technique: str) -> Tuple[str, str]:
    """
    Constructs canonical (Produktionsgebiet, Ware/Stil) tuple from BAPD fields.
    """
    fab_upper = (fabric or "").upper()
    tech_upper = (technique or "").upper()

    region = "Attisch" if "ATHENIAN" in fab_upper or "ATTIC" in fab_upper else (fabric.capitalize() if fabric else "Attisch")
    
    style = "Rotfigurig"
    for bapd_tech, de_tech in TECHNIQUE_MAPPING.items():
        if bapd_tech in tech_upper:
            style = de_tech
            break

    if "CORINTH" in fab_upper:
        region = "Korinthisch"
    elif "APULI" in fab_upper or "SOUTH ITALIAN" in fab_upper:
        region = "Unteritalisch / Apulisch"
    elif "CHALCID" in fab_upper:
        region = "Chalkidisch"

    ware = f"{region} {style}".strip()
    return region, ware


def format_date_range(raw_date: str) -> str:
    """Formats BAPD dates like '-550 TO -500' to German 'ca. 550–500 v. Chr.'"""
    if not raw_date:
        return ""
    m = re.match(r"(-\d+)\s+TO\s+(-\d+)", raw_date.strip())
    if m:
        start_yr = abs(int(m.group(1)))
        end_yr = abs(int(m.group(2)))
        return f"ca. {start_yr}–{end_yr} v. Chr."
    return raw_date


def split_collection_and_inv(raw_coll: str) -> Tuple[str, str]:
    """Splits 'Paris, Musée du Louvre: F386' into location and inventory number."""
    if not raw_coll:
        return "", ""
    # Strip Previous Collections or trailing metadata blocks if appended
    clean = raw_coll.split("Publication Record:")[0]
    clean = clean.split("Previous Collections:")[0].strip()
    if ":" in clean:
        loc, inv = clean.split(":", 1)
        return loc.strip(), inv.strip()
    return clean, ""


def is_drawing_file(path: str) -> bool:
    """Checks if an existing image on disk is a line/profile drawing."""
    if not os.path.exists(path):
        return False
    try:
        from PIL import Image
        import numpy as np
        im = Image.open(path).convert("L")
        gray = np.array(im)
        white_frac = float(np.mean(gray > 235))
        dark_frac = float(np.mean(gray < 50))
        mean_val = float(np.mean(gray))
        return (white_frac > 0.58 and dark_frac < 0.08) or mean_val > 218
    except Exception:
        return False


class CorpusBuilder:
    """
    Builds structured training corpora from the Beazley Archive.
    """
    def __init__(self, output_dir: str = "data/bapd_corpus"):
        self.output_dir = output_dir
        self.client = BAPDClient()

    def build_corpus(
        self,
        figures: Optional[List[str]] = None,
        limit_per_figure: int = 25,
        max_views_per_vase: int = 1,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Harvests and compiles the training dataset for specified mythological figures.
        If max_views_per_vase > 1, saves multiple photographic perspectives (Side A, B, etc.).
        """
        if not figures:
            figures = ["ATHENA", "DIONYSOS", "HERAKLES", "APOLLO"]

        os.makedirs(self.output_dir, exist_ok=True)
        self.client.ensure_verified()

        stats = {fig: {"found": 0, "downloaded": 0, "records": []} for fig in figures}
        total_targets = len(figures) * limit_per_figure
        processed_total = 0

        print(f"\n=======================================================")
        print(f"🚀 Starting BAPD Training Corpus Harvest")
        print(f"Target Figures: {', '.join(figures)}")
        print(f"Limit per Figure: {limit_per_figure} (Total: {total_targets})")
        print(f"Max Views per Vase: {max_views_per_vase}")
        print(f"Output Directory: {self.output_dir}")
        print(f"=======================================================\n")

        for fig_query in figures:
            fig_key = fig_query.upper().strip()
            fig_meta = MYTHOLOGY_TARGETS.get(fig_key, {
                "label": fig_key.capitalize(),
                "skos_uri": f"https://hector.bcdh.uni-bonn.de/{fig_key.lower()}"
            })
            fig_label = fig_meta["label"]
            skos_uri = fig_meta["skos_uri"]

            print(f"🔍 Searching BAPD for figure: {fig_label} ({fig_query})...")
            search_results = self.client.search_vases(
                query=fig_query,
                with_images=True,
                limit=limit_per_figure * 2 # Fetch extra candidates in case some lack downloadable images
            )
            stats[fig_key]["found"] = len(search_results)
            print(f"   Found {len(search_results)} candidates with images for {fig_label}.")

            downloaded_for_fig = 0
            for guid, summary in search_results:
                if downloaded_for_fig >= limit_per_figure:
                    break

                rec = self.client.get_vase_record(guid)
                if not rec or not rec.image_urls:
                    continue

                vase_nr = rec.vase_number or guid.strip("{}")
                stem = f"bapd_{vase_nr}"

                # Prepare common metadata
                shape_de = normalize_shape(rec.shape)
                prod_reg, ware_de = normalize_ware(rec.fabric, rec.technique)
                dating_de = format_date_range(rec.date_range)
                location, inv_nr = split_collection_and_inv(rec.collection)
                
                scene_name = f"{fig_label} auf antiker {shape_de}"
                if rec.artist:
                    clean_artist = rec.artist.split("by")[0].strip()
                    scene_name += f" ({clean_artist})"

                txt_template = (
                    f"Abbildungsnachweis: Beazley Archive Pottery Database (BAPD) / CARC Oxford\n"
                    f"Beazley-Nummer: BAPD {vase_nr}\n"
                    f"Beschreibung: {rec.decoration or summary}\n"
                    f"Datierung: {dating_de}\n"
                    f"Gattung: Gefäß\n"
                    f"Gefäßform: {shape_de}\n"
                    f"Herkunftsort: {rec.provenance}\n"
                    f"Inventar-Nr.: {inv_nr}\n"
                    f"Künstler/Werkstatt: {rec.artist}\n"
                    f"Material: Ton\n"
                    f"Name/Bezeichnung: {scene_name}\n"
                    f"Produktionsgebiet: {prod_reg}\n"
                    f"Standort: {location}\n"
                    f"Ware: {ware_de}\n"
                    f"Hauptmotiv: {fig_label}\n"
                    f"SKOS-URI: {skos_uri}\n"
                )

                # Download images (filtering out 2D line drawings)
                saved_stems = []
                if max_views_per_vase == 1:
                    img_path = os.path.join(self.output_dir, f"{stem}.jpg")
                    txt_path = os.path.join(self.output_dir, f"{stem}.txt")
                    
                    success = os.path.exists(img_path) and not is_drawing_file(img_path)
                    if not success:
                        for img_url in rec.image_urls:
                            if self.client.download_image(img_url, img_path, min_size=200, reject_drawings=True):
                                success = True
                                break
                    if success:
                        with open(txt_path, "w", encoding="utf-8") as f:
                            f.write(txt_template)
                        saved_stems.append(stem)
                else:
                    # Multi-view mode (max_views_per_vase == 0 means ALL available photographic views)
                    limit_views = max_views_per_vase if max_views_per_vase > 0 else 999
                    view_num = 1
                    for img_url in rec.image_urls:
                        if view_num > limit_views:
                            break
                        v_stem = f"{stem}_{view_num}"
                        v_img = os.path.join(self.output_dir, f"{v_stem}.jpg")
                        v_txt = os.path.join(self.output_dir, f"{v_stem}.txt")

                        success = os.path.exists(v_img) and not is_drawing_file(v_img)
                        if not success:
                            if self.client.download_image(img_url, v_img, min_size=200, reject_drawings=True):
                                success = True

                        if success:
                            with open(v_txt, "w", encoding="utf-8") as f:
                                f.write(txt_template)
                            saved_stems.append(v_stem)
                            view_num += 1

                if not saved_stems:
                    continue

                downloaded_for_fig += 1
                processed_total += 1
                stats[fig_key]["downloaded"] = downloaded_for_fig
                stats[fig_key]["records"].append({
                    "vase_number": vase_nr,
                    "shape": shape_de,
                    "ware": ware_de,
                    "images": [f"{s}.jpg" for s in saved_stems]
                })

                if progress_callback:
                    progress_callback(fig_label, processed_total, total_targets)

                print(f"   [{downloaded_for_fig}/{limit_per_figure}] Saved: {', '.join(saved_stems)} ({shape_de}, {ware_de})")

        # Compile Linked Open Data catalog for the harvested corpus
        print("\n📦 Generating Linked Open Data catalogs (JSON-LD, Turtle, CSV)...")
        catalog_files = self._generate_corpus_catalogs()

        print(f"\n✨ Harvest Complete! Successfully created dataset in: {self.output_dir}")
        return {
            "output_dir": self.output_dir,
            "stats": stats,
            "catalogs": catalog_files
        }

    def _generate_corpus_catalogs(self) -> Dict[str, str]:
        """Generates JSON-LD, TTL, and CSV catalogs for the corpus."""
        vases = load_directory_metadata(self.output_dir)
        if not vases:
            return {}

        all_items = []
        for meta in vases:
            item = meta.to_dict()
            resolved = skos_client.resolve_prediction(meta.shape, meta.ware)
            shape_concept = resolved.get("shape")
            ware_concept = resolved.get("ware")

            item["shape_label"] = shape_concept.pref_label if shape_concept else meta.shape
            item["shape_uri"] = shape_concept.uri if shape_concept else ""
            item["ware_label"] = ware_concept.pref_label if ware_concept else meta.ware
            item["ware_uri"] = ware_concept.uri if ware_concept else ""

            # Check if there is a main figure SKOS URI
            fig_skos = meta.fields.get("SKOS-URI", "")
            if fig_skos:
                item["iconography_concept"] = {
                    "label": meta.fields.get("Hauptmotiv", ""),
                    "uri": fig_skos
                }

            # Kerameikos Concept Alignment
            k_shape = kerameikos_client.resolve_shape(item["shape_label"])
            if k_shape:
                item["kerameikos_shape_uri"] = k_shape.uri
                item["kerameikos_shape_label"] = k_shape.pref_label_en
                item["kerameikos_shape_matches"] = k_shape.exact_matches

            k_tech = kerameikos_client.resolve_technique(item["ware_label"])
            if k_tech:
                item["kerameikos_technique_uri"] = k_tech.uri
                item["kerameikos_technique_label"] = k_tech.pref_label_en
                item["kerameikos_technique_matches"] = k_tech.exact_matches

            k_place = kerameikos_client.resolve_production_place(item.get("production_region", ""))
            if k_place:
                item["kerameikos_place_uri"] = k_place.uri
                item["kerameikos_place_label"] = k_place.pref_label_en

            k_art = kerameikos_client.resolve_artist(item.get("artist", ""))
            if k_art:
                item["kerameikos_artist_uri"] = k_art.uri
                item["kerameikos_artist_label"] = k_art.pref_label_en
                item["kerameikos_artist_type"] = k_art.concept_type

            all_items.append(item)

        jsonld_path = os.path.join(self.output_dir, "corpus_catalog.jsonld")
        ttl_path = os.path.join(self.output_dir, "corpus_catalog.ttl")
        csv_path = os.path.join(self.output_dir, "corpus_catalog.csv")

        # 1. JSON-LD
        with open(jsonld_path, "w", encoding="utf-8") as f:
            json.dump(LODExporter.to_jsonld(all_items), f, indent=2, ensure_ascii=False)

        # 2. Turtle
        with open(ttl_path, "w", encoding="utf-8") as f:
            f.write(LODExporter.to_rdf_turtle(all_items))

        # 3. CSV
        df = LODExporter.to_dataframe(all_items)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        return {
            "jsonld": jsonld_path,
            "turtle": ttl_path,
            "csv": csv_path
        }
