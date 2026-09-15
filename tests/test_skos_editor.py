"""
Test suite for SKOSClient local Turtle (.ttl) loading, concept CRUD, search, and live synchronization.
"""

import os
import tempfile
import unittest
from lodvases.skos_client import skos_client, SKOSConcept


class TestSKOSEditorAndTTL(unittest.TestCase):

    def test_local_ttl_loaded(self):
        self.assertGreater(len(skos_client.concepts_by_uri), 3000, "Should load > 3000 concepts from heritage_assets.ttl")
        labels = skos_client.get_all_labels_for_selection()
        self.assertGreater(len(labels), 3000, "Should have > 3000 unique labels for UI dropdowns")

    def test_search_concepts(self):
        # Search by term
        results = skos_client.search_concepts("Amphora", limit=20)
        self.assertGreater(len(results), 0, "Should find Amphora concepts")
        self.assertTrue(any("amphora" in c.pref_label.lower() for c in results))

    def test_add_edit_and_save_concept(self):
        listener_fired = []

        def on_change():
            listener_fired.append(True)

        skos_client.register_on_change_listener(on_change)

        # 1. Add new concept
        test_label = "Krieger mit Phrygischer Mütze"
        concept = skos_client.add_or_update_concept(
            pref_label_de=test_label,
            pref_label_en="Warrior with Phrygian Cap",
            alt_labels=["Phrygischer Krieger", "Amazonenkrieger"],
            category="Mythologie / Figur",
            definition="Kriegergestalt aus der kleinasiatischen und attischen Vasenikonographie."
        )

        self.assertEqual(concept.pref_label, test_label)
        self.assertTrue(len(listener_fired) > 0, "Listener should be notified on concept add")

        # 2. Lookup newly added concept
        resolved = skos_client.get_concept_by_label(test_label)
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.pref_label, test_label)
        self.assertEqual(resolved.definition, "Kriegergestalt aus der kleinasiatischen und attischen Vasenikonographie.")

        # 3. Save to temporary TTL and re-parse
        with tempfile.NamedTemporaryFile(suffix=".ttl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            skos_client.save_to_ttl(tmp_path)
            self.assertTrue(os.path.exists(tmp_path))
            self.assertGreater(os.path.getsize(tmp_path), 500000)

            # Re-load
            count = skos_client.load_local_ttl(tmp_path)
            self.assertGreater(count, 3000)
            re_resolved = skos_client.get_concept_by_label(test_label)
            self.assertIsNotNone(re_resolved)
            self.assertEqual(re_resolved.pref_label, test_label)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        # 4. Clean up test concept
        skos_client.delete_concept(concept.uri)


if __name__ == "__main__":
    unittest.main()
