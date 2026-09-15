"""
Unit Tests for Kerameikos.org Linked Open Data Client, Concept Resolver,
and Semantically Enriched LOD Exporter.
"""

import os
import sys
import unittest
import json
import rdflib
from rdflib import Graph, URIRef, RDF
from rdflib.namespace import SKOS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from lodvases.kerameikos_client import kerameikos_client, KerameikosClient
from lodvases.lod_exporter import LODExporter, KON, KERAMEIKOS, CRM


class TestKerameikos(unittest.TestCase):

    def setUp(self):
        self.client = kerameikos_client

    def test_shape_resolution(self):
        """Test resolving canonical Greek pottery shapes to Kerameikos.org URIs."""
        test_cases = [
            ("Kylix", "https://kerameikos.org/id/kylix"),
            ("Halsamphora", "https://kerameikos.org/id/neck_amphora"),
            ("Bauchamphora", "https://kerameikos.org/id/belly_amphora"),
            ("Panathenäische Preisamphore", "https://kerameikos.org/id/panathenaic_amphora"),
            ("Hydria", "https://kerameikos.org/id/hydria"),
            ("Lekythos", "https://kerameikos.org/id/lekythos"),
            ("Bauchlekythos", "https://kerameikos.org/id/squat_lekythos"),
            ("Glockenkrater", "https://kerameikos.org/id/bell_krater"),
            ("Kelchkrater", "https://kerameikos.org/id/calyx_krater"),
            ("Volutenkrater", "https://kerameikos.org/id/volute_krater"),
            ("Kolonettenkrater", "https://kerameikos.org/id/column_krater"),
            ("Pelike", "https://kerameikos.org/id/pelike"),
            ("Psykter", "https://kerameikos.org/id/psykter"),
            ("Pyxis", "https://kerameikos.org/id/pyxis"),
            ("Aryballos", "https://kerameikos.org/id/aryballos"),
            ("Alabastron", "https://kerameikos.org/id/alabastron"),
            ("Skyphos", "https://kerameikos.org/id/skyphos"),
            ("Kantharos", "https://kerameikos.org/id/kantharos"),
            ("Oinochoe", "https://kerameikos.org/id/oinochoe"),
            ("Teller", "https://kerameikos.org/id/plate"),
        ]
        for shape_name, expected_uri in test_cases:
            concept = self.client.resolve_shape(shape_name)
            self.assertIsNotNone(concept, f"Failed to resolve shape: {shape_name}")
            self.assertEqual(concept.uri, expected_uri, f"Mismatch for {shape_name}")
            self.assertEqual(concept.concept_type, "Shape")
            if concept.exact_matches:
                self.assertTrue(any("aat" in m or "wikidata" in m for m in concept.exact_matches))

    def test_technique_resolution(self):
        """Test resolving decoration techniques to Kerameikos.org URIs."""
        test_cases = [
            ("Rotfigurig", "https://kerameikos.org/id/red_figure"),
            ("Attisch Rotfigurig", "https://kerameikos.org/id/red_figure"),
            ("Schwarzfigurig", "https://kerameikos.org/id/black_figure"),
            ("Attisch Schwarzfigurig", "https://kerameikos.org/id/black_figure"),
            ("Weißgrundig", "https://kerameikos.org/id/white_ground"),
            ("Six-Technik", "https://kerameikos.org/id/six"),
            ("Korallenrot", "https://kerameikos.org/id/coral_red"),
        ]
        for tech_name, expected_uri in test_cases:
            concept = self.client.resolve_technique(tech_name)
            self.assertIsNotNone(concept, f"Failed to resolve technique: {tech_name}")
            self.assertEqual(concept.uri, expected_uri)
            self.assertEqual(concept.concept_type, "Technique")

    def test_place_resolution(self):
        """Test resolving production places/regions to Kerameikos.org URIs."""
        test_cases = [
            ("Attisch", "https://kerameikos.org/id/athens"),
            ("Athen", "https://kerameikos.org/id/athens"),
            ("Attika", "https://kerameikos.org/id/attica"),
            ("Korinthisch", "https://kerameikos.org/id/corinth"),
            ("Apulisch", "https://kerameikos.org/id/apulia"),
            ("Unteritalisch", "https://kerameikos.org/id/italy"),
            ("Kreta", "https://kerameikos.org/id/crete"),
        ]
        for place_name, expected_uri in test_cases:
            concept = self.client.resolve_production_place(place_name)
            self.assertIsNotNone(concept, f"Failed to resolve place: {place_name}")
            self.assertEqual(concept.uri, expected_uri)
            self.assertEqual(concept.concept_type, "ProductionPlace")

    def test_artist_resolution(self):
        """Test resolving painters, potters, and workshops to Kerameikos.org URIs."""
        test_cases = [
            ("Achilles Painter", "https://kerameikos.org/id/achilles_painter", "Person"),
            ("Manner of ANTIMENES P by PAUL, E.", "https://kerameikos.org/id/antimenes_painter", "Person"),
            ("EUPHRONIOS by SIGNATURE", "https://kerameikos.org/id/euphronios", "Person"),
            ("Near LYDOS by UNKNOWN", "https://kerameikos.org/id/lydos", "Person"),
            ("GROUP E by UNKNOWN", "https://kerameikos.org/id/group_e", "Group"),
            ("LEAGROS GROUP", "https://kerameikos.org/id/leagros_group", "Group"),
            ("HAIMON GROUP", "https://kerameikos.org/id/haimon_group", "Group"),
        ]
        for raw_str, expected_uri, expected_type in test_cases:
            concept = self.client.resolve_artist(raw_str)
            self.assertIsNotNone(concept, f"Failed to resolve artist: {raw_str}")
            self.assertEqual(concept.uri, expected_uri)
            self.assertEqual(concept.concept_type, expected_type)

    def test_jsonld_export_with_kerameikos(self):
        """Test JSON-LD export contains Kerameikos shape, production, technique and place links."""
        sample = {
            "id": "bapd_test_01",
            "name": "BAPD Test Kylix",
            "shape_label": "Kylix",
            "shape_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_kylix",
            "ware_label": "Attisch Rotfigurig",
            "ware_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_rf",
            "production_region": "Attisch",
            "artist": "Euphronios",
            "dating": "ca. 515–500 v. Chr."
        }

        doc = LODExporter.to_jsonld([sample])
        self.assertIn("@context", doc)
        self.assertIn("kon", doc["@context"])
        self.assertIn("kerameikos", doc["@context"])

        node = doc["@graph"][0]
        # Check kon:hasShape
        self.assertIn("hasShape", node)
        self.assertEqual(node["hasShape"]["@id"], "https://kerameikos.org/id/kylix")
        self.assertEqual(node["consistsOf"], "https://kerameikos.org/id/terracotta")

        # Check exactMatch on BCDH concept
        self.assertEqual(node["shapeConcept"]["exactMatch"], "https://kerameikos.org/id/kylix")
        self.assertEqual(node["wareConcept"]["exactMatch"], "https://kerameikos.org/id/red_figure")

        # Check Production Event
        self.assertIn("wasProducedBy", node)
        prod = node["wasProducedBy"]
        self.assertEqual(prod["@type"], "crm:E12_Production")
        self.assertEqual(prod["tookPlaceAt"]["@id"], "https://kerameikos.org/id/athens")
        self.assertEqual(prod["carriedOutBy"]["@id"], "https://kerameikos.org/id/euphronios")
        self.assertEqual(prod["usedTechnique"]["@id"], "https://kerameikos.org/id/red_figure")

    def test_turtle_export_with_kerameikos(self):
        """Test RDF Turtle serialization parses cleanly and produces valid Kerameikos/CIDOC triples."""
        sample = {
            "id": "bapd_test_02",
            "name": "BAPD Test Amphora",
            "shape_label": "Halsamphora",
            "shape_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_neck_amphora",
            "ware_label": "Attisch Schwarzfigurig",
            "ware_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_bf",
            "production_region": "Attisch",
            "artist": "Exekias",
            "dating": "ca. 540–530 v. Chr."
        }

        ttl_str = LODExporter.to_rdf_turtle([sample])
        g = Graph()
        g.parse(data=ttl_str, format="turtle")

        obj_uri = URIRef("http://data.lodvases.bcdh.uni-bonn.de/object/bapd_test_02")
        prod_uri = URIRef("http://data.lodvases.bcdh.uni-bonn.de/object/bapd_test_02#production")
        k_shape = URIRef("https://kerameikos.org/id/neck_amphora")
        k_tech = URIRef("https://kerameikos.org/id/black_figure")
        k_art = URIRef("https://kerameikos.org/id/exekias")
        k_place = URIRef("https://kerameikos.org/id/athens")

        # Object hasShape and consists_of
        self.assertIn((obj_uri, KON.hasShape, k_shape), g)
        self.assertIn((obj_uri, CRM.P45_consists_of, KERAMEIKOS["terracotta"]), g)

        # Production event
        self.assertIn((obj_uri, CRM.P108i_was_produced_by, prod_uri), g)
        self.assertIn((prod_uri, CRM.P7_took_place_at, k_place), g)
        self.assertIn((prod_uri, CRM.P14_carried_out_by, k_art), g)
        self.assertIn((prod_uri, CRM.P32_used_general_technique, k_tech), g)

        # Exact match linking BCDH SKOS to Kerameikos
        hector_shape = URIRef(sample["shape_uri"])
        self.assertIn((hector_shape, SKOS.exactMatch, k_shape), g)

    def test_dataframe_and_txt_export(self):
        """Test DataFrame and enriched text export contain Kerameikos URIs."""
        sample = {
            "id": "bapd_test_03",
            "name": "BAPD Test Lekythos",
            "shape_label": "Lekythos",
            "ware_label": "Attisch Rotfigurig",
            "production_region": "Attisch",
            "artist": "Achilles Painter",
            "dating": "ca. 450 v. Chr."
        }

        df = LODExporter.to_dataframe([sample])
        self.assertIn("Kerameikos Form-URI", df.columns)
        self.assertEqual(df["Kerameikos Form-URI"].iloc[0], "https://kerameikos.org/id/lekythos")
        self.assertEqual(df["Kerameikos Technik-URI"].iloc[0], "https://kerameikos.org/id/red_figure")
        self.assertEqual(df["Kerameikos Herkunft-URI"].iloc[0], "https://kerameikos.org/id/athens")
        self.assertEqual(df["Kerameikos Künstler-URI"].iloc[0], "https://kerameikos.org/id/achilles_painter")

        txt = LODExporter.to_enriched_txt(sample)
        self.assertIn("[Kerameikos: https://kerameikos.org/id/lekythos]", txt)
        self.assertIn("[Kerameikos: https://kerameikos.org/id/red_figure]", txt)
        self.assertIn("[Kerameikos: https://kerameikos.org/id/athens]", txt)
        self.assertIn("[Kerameikos: https://kerameikos.org/id/achilles_painter]", txt)


if __name__ == "__main__":
    unittest.main()
