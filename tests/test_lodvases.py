"""
Unit Tests for LODVases including Shape, Ware, Scene Classification, SAM Segmentation,
3D GLB Processing, and Geometry Profile Analysis.
"""

import os
import sys
import json
import io
from PIL import Image
import rdflib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from lodvases.metadata_parser import VaseMetadata, load_directory_metadata
from lodvases.skos_client import SKOSClient, skos_client
from lodvases.lod_exporter import LODExporter
from lodvases.classifier import classifier
from lodvases.sam_segmenter import sam_segmenter
from lodvases.dataset_exporter import DatasetExporter
from lodvases.glb_processor import glb_processor
from lodvases.profile_analyzer import profile_analyzer

PICS_DIR = os.path.join(BASE_DIR, "pics")


def test_metadata_parser():
    """Test loading and parsing .txt metadata files."""
    assert os.path.exists(PICS_DIR), "pics directory must exist"
    records = load_directory_metadata(PICS_DIR)
    assert len(records) >= 20, f"Expected at least 20 records, got {len(records)}"


def test_skos_client_curated_lookup():
    """Test lookup of curated HECTOR concepts."""
    c_lekythos = skos_client.get_concept_by_label("Lekythos")
    assert c_lekythos is not None
    assert c_lekythos.pref_label == "Lekythos"

    c_delphin = skos_client.get_concept_by_label("Delphinreiter")
    assert c_delphin is not None
    assert "dolphin_group" in c_delphin.uri


def test_lod_exporter_3d_dimensions():
    """Test JSON-LD export with 3D model and dimensions."""
    sample_item = {
        "id": "3d_lekythos_01",
        "name": "3D Lekythos",
        "shape_label": "Lekythos",
        "shape_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_5b62f4af",
        "model_3d": "lekythos.glb",
        "dimensions_3d": {
            "height": 30.0,
            "max_diameter": 9.0,
            "rim_diameter": 4.0,
            "base_diameter": 5.0,
            "estimated_volume_liters": 0.85
        }
    }

    jsonld_doc = LODExporter.to_jsonld([sample_item])
    node = jsonld_doc["@graph"][0]
    assert "model3D" in node
    assert "hasDimension" in node
    assert node["hasDimension"]["height"] == "30.0 cm"


def test_classifier_and_sam():
    """Test vision classifier and SAM segmenter."""
    sample_img_path = os.path.join(PICS_DIR, "0470d55f-92f3-4d6a-8180-4d2d891a9277.jpg")
    img = Image.open(sample_img_path).convert("RGB")
    
    pred = classifier.predict(img, top_k=3)
    assert "predicted_shape" in pred
    
    scene_pred = classifier.predict_scene(img, top_k=3)
    assert "predicted_scene" in scene_pred


def test_3d_glb_processing_and_profile_analysis():
    """Test 3D GLB mesh processing and archaeological profile section drawing."""
    # 1. Create sample synthetic 3D Lekythos
    glb_bytes = glb_processor.create_sample_3d_vase("Lekythos")
    assert len(glb_bytes) > 0, "GLB bytes should not be empty"

    # 2. Load GLB
    glb_info = glb_processor.load_glb(glb_bytes)
    assert glb_info["vertices_count"] > 0
    assert glb_info["faces_count"] > 0
    mesh = glb_info["mesh"]

    # 3. Analyze Geometry Profile
    profile_data = profile_analyzer.analyze_mesh(mesh)
    assert profile_data["height"] > 0
    assert profile_data["max_diameter"] > 0
    assert profile_data["suggested_shape"] == "Lekythos"
    assert profile_data["slenderness_ratio"] > 1.8

    # 4. Generate Section Drawing
    drawing_buf = profile_analyzer.generate_section_drawing(profile_data)
    assert len(drawing_buf.getvalue()) > 0, "Section drawing PNG should not be empty"


def test_style_informed_segmentation():
    """Test style-informed structure mask extraction and constrained SAM segmentation."""
    # 1. Red-figure test (Herakles Psykter)
    rf_img_path = os.path.join(PICS_DIR, "c444f0e4-61ae-4e19-93e1-779305f0773b.jpg")
    rf_img = Image.open(rf_img_path).convert("RGB")
    rf_mask = sam_segmenter.extract_structure_mask(rf_img, style="Attisch Rotfigurig")
    assert rf_mask.shape == (rf_img.height, rf_img.width)
    assert rf_mask.sum() > 0, "Red-figure structure mask should contain terracotta figure areas"

    # 2. Black-figure test (Amphora unroll)
    bf_img_path = os.path.join(PICS_DIR, "3d_abrollung_Amphora_38_gameready.glb.png")
    if os.path.exists(bf_img_path):
        bf_img = Image.open(bf_img_path).convert("RGB")
        bf_mask = sam_segmenter.extract_structure_mask(bf_img, style="Attisch Schwarzfigurig")
        assert bf_mask.shape == (bf_img.height, bf_img.width)
        assert bf_mask.sum() > 0, "Black-figure structure mask should contain dark glaze silhouettes"

    # 3. Constrained segment_all test
    motifs = sam_segmenter.segment_all(rf_img, style="Attisch Rotfigurig", use_style_constraint=True, min_area_pct=0.5)
    assert len(motifs) > 0, "Should extract constrained red-figure motifs"
    for m in motifs:
        assert m["cutout"] is not None
        assert "mask" in m
        assert "bbox" in m


if __name__ == "__main__":
    test_metadata_parser()
    print("test_metadata_parser: OK")
    test_skos_client_curated_lookup()
    print("test_skos_client_curated_lookup: OK")
    test_lod_exporter_3d_dimensions()
    print("test_lod_exporter_3d_dimensions: OK")
    test_classifier_and_sam()
    print("test_classifier_and_sam: OK")
    test_3d_glb_processing_and_profile_analysis()
    print("test_3d_glb_processing_and_profile_analysis: OK")
    test_style_informed_segmentation()
    print("test_style_informed_segmentation: OK")
    print("\nAll unit tests (including 3D GLB, Profile Analysis & Style-Informed SAM) passed successfully!")
