"""
CLI for LODVases: Ancient Vase Shape & Ware Classifier with SKOS / LOD Indexing.
"""

import os
import sys
import argparse
import json
from typing import Optional

# Ensure UTF-8 stdout for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from lodvases.classifier import classifier
from lodvases.skos_client import skos_client
from lodvases.lod_exporter import LODExporter
from lodvases.metadata_parser import load_directory_metadata, VaseMetadata

def print_header():
    print("=" * 65)
    print("  LODVases - Ancient Ceramics AI Classifier & SKOS / LOD Indexer")
    print("  BCDH Universität Bonn | HECTOR Heritage Assets Vocabulary")
    print("=" * 65)

def predict_single(image_path: str, format_out: str = "text"):
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)

    print(f"\n[1/3] Analyzing image: {image_path}...")
    pred = classifier.predict(image_path)
    
    shape_label = pred["predicted_shape"]
    ware_label = pred["predicted_ware"]
    shape_conf = pred["shape_confidence"]
    ware_conf = pred["ware_confidence"]

    print(f"[2/3] Resolving SKOS concepts via BCDH HECTOR Skosmos...")
    resolved = skos_client.resolve_prediction(shape_label, ware_label)
    shape_concept = resolved["shape"]
    ware_concept = resolved["ware"]

    print(f"[3/3] Analysis complete!\n")

    item_data = {
        "id": os.path.splitext(os.path.basename(image_path))[0],
        "image_path": image_path,
        "shape_label": shape_concept.pref_label if shape_concept else shape_label,
        "shape_uri": shape_concept.uri if shape_concept else "",
        "confidence_shape": shape_conf,
        "ware_label": ware_concept.pref_label if ware_concept else ware_label,
        "ware_uri": ware_concept.uri if ware_concept else "",
        "confidence_ware": ware_conf
    }

    if format_out == "jsonld":
        print(json.dumps(LODExporter.to_jsonld([item_data]), indent=2, ensure_ascii=False))
    elif format_out == "turtle":
        print(LODExporter.to_rdf_turtle([item_data]))
    elif format_out == "json":
        print(json.dumps(item_data, indent=2, ensure_ascii=False))
    else:
        print("-----------------------------------------------------------------")
        print(f"[Form] Erkannte Gefaessform : {shape_label} ({shape_conf * 100:.1f}%)")
        if shape_concept:
            print(f"   -> SKOS PrefLabel        : {shape_concept.pref_label}")
            print(f"   -> HECTOR URI            : {shape_concept.uri}")
            print(f"   -> Skosmos Web           : {shape_concept.web_url}")
            if shape_concept.broader_label:
                print(f"   -> Oberbegriff           : {shape_concept.broader_label}")

        print(f"\n[Ware] Erkannte Ware/Stil    : {ware_label} ({ware_conf * 100:.1f}%)")
        if ware_concept:
            print(f"   -> SKOS PrefLabel        : {ware_concept.pref_label}")
            print(f"   -> HECTOR URI            : {ware_concept.uri}")
            print(f"   -> Skosmos Web           : {ware_concept.web_url}")

        print(f"\n[Color] Dominante Farben     : " + ", ".join([f"{c['hex']} ({c['percentage']}%)" for c in pred["color_palette"]]))
        if pred.get("predicted_mythology"):
            print(f"\n[Mythos] Erkannte Gottheit   : {pred['predicted_mythology']} ({pred['mythology_confidence'] * 100:.1f}%)")
            print("Top-4 Mythologische Figuren:")
            for m in pred.get("top_mythologies", []):
                print(f"  * {m['label']:16}: {m['confidence']*100:.1f}%")

        print("\nTop-3 Gefaessformen:")
        for s in pred["top_shapes"][:3]:
            print(f"  * {s['label']:16}: {s['confidence']*100:.1f}%")
        print("\nTop-3 Waren / Stile:")
        for w in pred["top_wares"][:3]:
            print(f"  * {w['label']:24}: {w['confidence']*100:.1f}%")
        print("-----------------------------------------------------------------")


def scan_directory(dir_path: str, output_path: Optional[str] = None, export_format: str = "jsonld"):
    if not os.path.exists(dir_path):
        print(f"Error: Directory '{dir_path}' not found.")
        sys.exit(1)

    print(f"\nScanning directory: {dir_path}")
    meta_list = load_directory_metadata(dir_path)
    print(f"Found {len(meta_list)} vase records.")

    all_items = []
    for i, meta in enumerate(meta_list, 1):
        img_path = meta.image_path
        print(f"[{i}/{len(meta_list)}] Processing {meta.id}...", end=" ", flush=True)

        shape_label = meta.shape
        ware_label = meta.ware
        shape_conf = None
        ware_conf = None

        # Run AI prediction if image exists
        if img_path and os.path.exists(img_path):
            try:
                pred = classifier.predict(img_path)
                ai_shape = pred["predicted_shape"]
                ai_ware = pred["predicted_ware"]
                shape_conf = pred["shape_confidence"]
                ware_conf = pred["ware_confidence"]
                if not shape_label:
                    shape_label = ai_shape
                if not ware_label:
                    ware_label = ai_ware
            except Exception as e:
                print(f"(Vision warning: {e})", end=" ")

        # Resolve SKOS concepts
        resolved = skos_client.resolve_prediction(shape_label, ware_label)
        shape_concept = resolved["shape"]
        ware_concept = resolved["ware"]

        item = meta.to_dict()
        item["shape_label"] = shape_concept.pref_label if shape_concept else shape_label
        item["shape_uri"] = shape_concept.uri if shape_concept else ""
        item["confidence_shape"] = shape_conf
        item["ware_label"] = ware_concept.pref_label if ware_concept else ware_label
        item["ware_uri"] = ware_concept.uri if ware_concept else ""
        item["confidence_ware"] = ware_conf

        all_items.append(item)
        print(f"-> {item['shape_label']} | {item['ware_label']}")

    print(f"\nSuccessfully indexed {len(all_items)} vases.")

    # Export
    if not output_path:
        ext = ".jsonld" if export_format == "jsonld" else (".ttl" if export_format == "turtle" else ".csv")
        output_path = os.path.join(dir_path, f"lod_catalog{ext}")

    print(f"Exporting to {output_path} (Format: {export_format})...")
    if export_format == "jsonld":
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(LODExporter.to_jsonld(all_items), f, indent=2, ensure_ascii=False)
    elif export_format == "turtle":
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(LODExporter.to_rdf_turtle(all_items))
    elif export_format == "csv":
        df = LODExporter.to_dataframe(all_items)
        df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("Done!")


def skos_search(query: str):
    print(f"\nSearching BCDH HECTOR Vocabulary for: '{query}'...")
    results = skos_client.search_live(query)
    if not results:
        # Check cache
        c = skos_client.get_concept_by_label(query)
        if c:
            results = [c]

    if not results:
        print("No matching concepts found.")
        return

    print(f"Found {len(results)} concepts:\n")
    for r in results:
        print(f"  • {r.pref_label}")
        print(f"    URI: {r.uri}")
        print(f"    Web: {r.web_url}")
        if r.broader_label:
            print(f"    Oberbegriff: {r.broader_label}")
        print()


def main():
    parser = argparse.ArgumentParser(description="LODVases - Ancient Vase AI Classifier & SKOS Indexer")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Predict single
    p_pred = subparsers.add_parser("predict", help="Predict shape and ware for a single image")
    p_pred.add_argument("image", help="Path to vase image (.jpg/.png)")
    p_pred.add_argument("--format", choices=["text", "jsonld", "turtle", "json"], default="text", help="Output format")

    # Scan directory
    p_scan = subparsers.add_parser("scan", help="Batch scan directory and export LOD catalog")
    p_scan.add_argument("dir", help="Directory containing vase images and metadata")
    p_scan.add_argument("--output", "-o", help="Output file path")
    p_scan.add_argument("--format", choices=["jsonld", "turtle", "csv"], default="jsonld", help="Export format")

    # SKOS search
    p_search = subparsers.add_parser("skos-search", help="Search BCDH HECTOR vocabulary")
    p_search.add_argument("query", help="Search query")

    # Kerameikos lookup
    p_kera = subparsers.add_parser("kerameikos-lookup", help="Look up a term in Kerameikos.org Linked Open Data ontology")
    p_kera.add_argument("term", help="Term to look up (e.g. Kylix, Rotfigurig, Attisch, Euphronios)")
    p_kera.add_argument("--type", choices=["auto", "shape", "technique", "place", "artist"], default="auto", help="Concept type filter (default: auto)")

    # Enrich Corpus
    p_enrich = subparsers.add_parser("enrich-corpus", help="Regenerate JSON-LD, Turtle, and CSV catalogs for a corpus with Kerameikos & BCDH LOD")
    p_enrich.add_argument("--dir", "-d", default="data/bapd_corpus", help="Path to corpus directory (default: data/bapd_corpus)")

    # Harvest BAPD
    p_harvest = subparsers.add_parser("harvest-bapd", help="Harvest vase dataset from Beazley Archive Pottery Database (CARC)")
    p_harvest.add_argument("--figures", default="ATHENA,DIONYSOS,HERAKLES,APOLLO", help="Comma-separated mythological figures (default: ATHENA,DIONYSOS,HERAKLES,APOLLO)")
    p_harvest.add_argument("--limit", type=int, default=25, help="Number of vases per figure (default: 25)")
    p_harvest.add_argument("--max-views", type=int, default=0, help="Max photographic views per vase (0 = all available views, 1 = main scene only, default: 0)")
    p_harvest.add_argument("--output", "-o", default="data/bapd_corpus", help="Output directory for images and metadata (default: data/bapd_corpus)")

    # Train Corpus
    p_train = subparsers.add_parser("train-corpus", help="Train AI classifier on vase corpus with GPU acceleration")
    p_train.add_argument("--corpus", "-c", default="data/bapd_corpus", help="Directory containing vase dataset (default: data/bapd_corpus)")
    p_train.add_argument("--epochs", "-e", type=int, default=40, help="Number of training epochs (default: 40)")
    p_train.add_argument("--batch-size", "-b", type=int, default=32, help="Batch size (default: 32)")
    p_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    p_train.add_argument("--output", "-o", default="models/mythology_classifier.pt", help="Path to save trained model (default: models/mythology_classifier.pt)")

    # YOLO: Build Dataset
    p_yolo_ds = subparsers.add_parser("build-yolo-dataset", help="Generate YOLOv8 object detection dataset from BAPD corpus")
    p_yolo_ds.add_argument("--corpus", "-c", default="data/bapd_corpus", help="Path to corpus directory (default: data/bapd_corpus)")
    p_yolo_ds.add_argument("--output", "-o", default="data/yolo_deities", help="Output directory for YOLO dataset (default: data/yolo_deities)")
    p_yolo_ds.add_argument("--train-ratio", type=float, default=0.80, help="Ratio for training split (default: 0.80)")

    # YOLO: Train Model
    p_yolo_tr = subparsers.add_parser("train-yolo", help="Fine-tune YOLOv8 object detection model on GPU")
    p_yolo_tr.add_argument("--data", default="data/yolo_deities/data.yaml", help="Path to data.yaml (default: data/yolo_deities/data.yaml)")
    p_yolo_tr.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs (default: 30)")
    p_yolo_tr.add_argument("--batch-size", "-b", type=int, default=16, help="Batch size (default: 16)")
    p_yolo_tr.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    p_yolo_tr.add_argument("--output", "-o", default="models/yolo_deities_best.pt", help="Output weights path (default: models/yolo_deities_best.pt)")

    # YOLO: Detect Figures
    p_yolo_det = subparsers.add_parser("detect-figures", help="Detect deities and figure bounding boxes on ancient vase image")
    p_yolo_det.add_argument("image", help="Path to vase image")
    p_yolo_det.add_argument("--conf", type=float, default=0.18, help="Confidence threshold (default: 0.18)")
    p_yolo_det.add_argument("--output-image", "-o", help="Optional path to save annotated visualization image")

    # GUI Launcher
    p_gui = subparsers.add_parser("gui", help="Launch LODVases Desktop Application (CustomTkinter)")

    args = parser.parse_args()

    if args.command == "predict":
        print_header()
        predict_single(args.image, args.format)
    elif args.command == "harvest-bapd":
        print_header()
        from lodvases.corpus_builder import CorpusBuilder
        figures = [f.strip() for f in args.figures.split(",") if f.strip()]
        builder = CorpusBuilder(output_dir=args.output)
        builder.build_corpus(figures=figures, limit_per_figure=args.limit, max_views_per_vase=args.max_views)
    elif args.command == "kerameikos-lookup":
        print_header()
        from lodvases.kerameikos_client import kerameikos_client
        term = args.term
        ctype = args.type
        concept = None
        if ctype == "shape" or ctype == "auto":
            concept = kerameikos_client.resolve_shape(term)
        if not concept and (ctype == "technique" or ctype == "auto"):
            concept = kerameikos_client.resolve_technique(term)
        if not concept and (ctype == "place" or ctype == "auto"):
            concept = kerameikos_client.resolve_production_place(term)
        if not concept and (ctype == "artist" or ctype == "auto"):
            concept = kerameikos_client.resolve_artist(term)

        if concept:
            print(f"\n[Kerameikos Match Found]")
            print(f"  URI          : {concept.uri}")
            print(f"  Concept Type : {concept.concept_type}")
            print(f"  Label (EN)   : {concept.pref_label_en}")
            print(f"  Label (DE)   : {concept.pref_label_de}")
            if concept.broader:
                print(f"  Broader      : {concept.broader}")
            if concept.exact_matches:
                print(f"  Exact Matches: ")
                for m in concept.exact_matches:
                    print(f"    * {m}")
        else:
            print(f"\nNo Kerameikos concept found for '{term}' (type: {ctype}).")
    elif args.command == "enrich-corpus":
        print_header()
        from lodvases.corpus_builder import CorpusBuilder
        print(f"Recompiling catalogs with Kerameikos & BCDH LOD for: {args.dir}...")
        builder = CorpusBuilder(output_dir=args.dir)
        cats = builder._generate_corpus_catalogs()
        print(f"Successfully updated catalogs:")
        for k, v in cats.items():
            print(f"  - {k.upper()}: {v}")
    elif args.command == "scan":
        print_header()
        scan_directory(args.dir, args.output, args.format)
    elif args.command == "skos-search":
        print_header()
        skos_search(args.query)
    elif args.command == "train-corpus":
        print_header()
        from lodvases.trainer import CorpusTrainer
        trainer = CorpusTrainer(corpus_dir=args.corpus, output_model_path=args.output)
        trainer.train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    elif args.command == "build-yolo-dataset":
        print_header()
        from lodvases.yolo_dataset_builder import yolo_dataset_builder
        yolo_dataset_builder.build_dataset(
            corpus_dir=args.corpus,
            output_dir=args.output,
            train_ratio=args.train_ratio
        )
    elif args.command == "train-yolo":
        print_header()
        from lodvases.train_yolo import train_yolo
        train_yolo(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            output_weights=args.output
        )
    elif args.command == "detect-figures":
        print_header()
        from lodvases.detector import figure_detector
        print(f"Detecting figures and deities in: {args.image} (conf >= {args.conf:.2f})...\n")
        detections = figure_detector.detect(args.image, conf_threshold=args.conf)
        if not detections:
            print("Keine Figuren über dem Konfidenz-Schwellenwert erkannt.")
        else:
            print(f"Erkannte Figuren ({len(detections)}):")
            print(f"{'-'*75}")
            print(f"{'Gottheit':12s} | {'Position':10s} | {'Konfidenz':10s} | {'Bounding Box [x1,y1,x2,y2]':26s} | {'SKOS-URI'}")
            print(f"{'-'*75}")
            for d in detections:
                box_str = str(d['bbox_xyxy'])
                print(f"{d['label']:12s} | {d['spatial_position']:10s} | {d['confidence']*100:6.1f}%    | {box_str:26s} | {d['skos_uri']}")
            print(f"{'-'*75}\n")

        if args.output_image and detections:
            annotated = figure_detector.draw_detections(args.image, detections)
            annotated.save(args.output_image)
            print(f"[OK] Visualisierung gespeichert unter: {args.output_image}")
    elif args.command == "gui":
        print("Launching LODVases Desktop App...")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        gui_path = os.path.join(base_dir, "gui.py")
        os.system(f"python \"{gui_path}\"")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
