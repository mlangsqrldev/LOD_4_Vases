"""
Unit tests for BAPD Client and Corpus Builder.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lodvases.bapd_client import BAPDClient
from lodvases.corpus_builder import (
    normalize_shape,
    normalize_ware,
    format_date_range,
    CorpusBuilder,
    MYTHOLOGY_TARGETS
)
from lodvases.metadata_parser import VaseMetadata


class TestBAPDCorpus(unittest.TestCase):
    def test_normalization_rules(self):
        # Shape normalization
        self.assertEqual(normalize_shape("AMPHORA, NECK"), "Halsamphora")
        self.assertEqual(normalize_shape("CUP"), "Kylix")
        self.assertEqual(normalize_shape("BELL KRATER"), "Glockenkrater")
        self.assertEqual(normalize_shape("LEKYTHOS"), "Lekythos")
        self.assertEqual(normalize_shape("UNKNOWN SHAPE"), "Unknown shape")

        # Ware normalization
        reg, ware = normalize_ware("ATHENIAN", "RED-FIGURE")
        self.assertEqual(reg, "Attisch")
        self.assertEqual(ware, "Attisch Rotfigurig")

        reg2, ware2 = normalize_ware("ATHENIAN", "BLACK-FIGURE")
        self.assertEqual(reg2, "Attisch")
        self.assertEqual(ware2, "Attisch Schwarzfigurig")

        # Date formatting
        self.assertEqual(format_date_range("-525 TO -475"), "ca. 525–475 v. Chr.")
        self.assertEqual(format_date_range("-550 TO -500"), "ca. 550–500 v. Chr.")

    def test_mythology_targets(self):
        for fig in ["ATHENA", "DIONYSOS", "HERAKLES", "APOLLO"]:
            self.assertIn(fig, MYTHOLOGY_TARGETS)
            self.assertTrue(MYTHOLOGY_TARGETS[fig]["skos_uri"].startswith("https://hector.bcdh.uni-bonn.de/"))

    def test_bapd_client_live_search(self):
        client = BAPDClient()
        self.assertTrue(client.ensure_verified())
        
        # Test search for Dionysos
        results = client.search_vases("DIONYSOS", with_images=True, limit=3)
        self.assertGreaterEqual(len(results), 1)
        
        guid, summary = results[0]
        self.assertTrue(guid.startswith("{"))
        self.assertTrue(guid.endswith("}"))
        
        # Test record retrieval
        record = client.get_vase_record(guid)
        self.assertIsNotNone(record)
        self.assertTrue(bool(record.vase_number))


if __name__ == "__main__":
    unittest.main()
