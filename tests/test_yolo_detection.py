"""
Unit tests for YOLOv8 Object Detection and Deity Localization Pipeline.
"""

import os
import unittest
import yaml
from PIL import Image
from lodvases.yolo_dataset_builder import YOLODatasetBuilder, DEITY_KEYWORDS, CLASS_NAMES
from lodvases.detector import VaseFigureDetector, figure_detector


class TestYOLODetection(unittest.TestCase):

    def test_base_id_grouping(self):
        """Verifies that multi-view filenames map to the same vessel ID for leak prevention."""
        self.assertEqual(YOLODatasetBuilder.extract_base_id("data/bapd_corpus/bapd_596_1.jpg"), "bapd_596")
        self.assertEqual(YOLODatasetBuilder.extract_base_id("data/bapd_corpus/bapd_596_2.jpg"), "bapd_596")
        self.assertEqual(YOLODatasetBuilder.extract_base_id("bapd_36.jpg"), "bapd_36")
        self.assertEqual(YOLODatasetBuilder.extract_base_id("bapd_1000_4.jpg"), "bapd_1000")

    def test_spatial_position_calculation(self):
        """Verifies spatial position mapping: links, Mitte, rechts."""
        img_w = 1000
        # Left (center at 200/1000 = 0.20 < 0.38)
        self.assertEqual(VaseFigureDetector.calculate_spatial_position(100, 300, img_w), "links")
        # Center (center at 500/1000 = 0.50)
        self.assertEqual(VaseFigureDetector.calculate_spatial_position(400, 600, img_w), "Mitte")
        # Right (center at 800/1000 = 0.80 > 0.62)
        self.assertEqual(VaseFigureDetector.calculate_spatial_position(700, 900, img_w), "rechts")

    def test_yolo_dataset_yaml(self):
        """Verifies that data/yolo_deities/data.yaml exists and has the expected format."""
        yaml_path = "data/yolo_deities/data.yaml"
        if os.path.exists(yaml_path):
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            self.assertIn("names", data)
            self.assertIn("train", data)
            self.assertIn("val", data)
            self.assertEqual(len(data["names"]), 4)
            self.assertEqual(data["names"][0], "Athena")
            self.assertEqual(data["names"][1], "Dionysos")
            self.assertEqual(data["names"][2], "Herakles")
            self.assertEqual(data["names"][3], "Apollon")

    def test_detector_inference(self):
        """Tests end-to-end detection on a sample corpus image."""
        self.assertTrue(figure_detector.is_ready, "Model weights should be loaded")
        test_img = "data/bapd_corpus/bapd_100_1.jpg"
        if os.path.exists(test_img):
            dets = figure_detector.detect(test_img, conf_threshold=0.15)
            self.assertIsInstance(dets, list)
            if dets:
                first = dets[0]
                self.assertIn("label", first)
                self.assertIn("spatial_position", first)
                self.assertIn("bbox_xyxy", first)
                self.assertIn("confidence", first)
                self.assertIn("skos_uri", first)
                self.assertIn(first["spatial_position"], ["links", "Mitte", "rechts"])
                self.assertIn(first["label"], CLASS_NAMES)

    def test_detector_draw_overlay(self):
        """Tests that drawing bounding box overlays produces a valid PIL image."""
        test_img = "data/bapd_corpus/bapd_100_1.jpg"
        if os.path.exists(test_img):
            dets = figure_detector.detect(test_img, conf_threshold=0.15)
            annotated = figure_detector.draw_detections(test_img, dets)
            self.assertIsInstance(annotated, Image.Image)
            self.assertEqual(annotated.mode, "RGB")


if __name__ == "__main__":
    unittest.main()
