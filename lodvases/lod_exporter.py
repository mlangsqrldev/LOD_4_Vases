"""
LOD Exporter: Exports vase annotations and scene/iconography detections to JSON-LD, RDF Turtle (.ttl), CSV, and enriched text.
Complies with W3C SKOS, W3C Web Annotation, CIDOC CRM, Kerameikos.org, and Schema.org standards.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
import rdflib
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import SKOS, DCTERMS, RDFS, XSD

from .kerameikos_client import kerameikos_client

SCHEMA = Namespace("http://schema.org/")
CRM = Namespace("http://www.cidoc-crm.org/cidoc-crm/")
HECTOR = Namespace("http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/")
BASE_OBJECT = Namespace("http://data.lodvases.bcdh.uni-bonn.de/object/")
OA = Namespace("http://www.w3.org/ns/oa#")
KON = Namespace("https://kerameikos.org/ontology#")
KERAMEIKOS = Namespace("https://kerameikos.org/id/")


def _enrich_with_kerameikos(item: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to ensure Kerameikos LOD concepts are resolved for any item."""
    enriched = dict(item)
    
    # Shape
    if not enriched.get("kerameikos_shape_uri"):
        shape_term = enriched.get("shape_label") or enriched.get("shape") or ""
        kc = kerameikos_client.resolve_shape(shape_term)
        if kc:
            enriched["kerameikos_shape_uri"] = kc.uri
            enriched["kerameikos_shape_label"] = kc.pref_label_en
            enriched["kerameikos_shape_matches"] = kc.exact_matches
            
    # Technique / Ware
    if not enriched.get("kerameikos_technique_uri"):
        ware_term = enriched.get("ware_label") or enriched.get("ware") or ""
        kc = kerameikos_client.resolve_technique(ware_term)
        if kc:
            enriched["kerameikos_technique_uri"] = kc.uri
            enriched["kerameikos_technique_label"] = kc.pref_label_en
            enriched["kerameikos_technique_matches"] = kc.exact_matches
            
    # Production Place / Fabric
    if not enriched.get("kerameikos_place_uri"):
        place_term = enriched.get("production_region") or enriched.get("findspot") or ""
        kc = kerameikos_client.resolve_production_place(place_term)
        if kc:
            enriched["kerameikos_place_uri"] = kc.uri
            enriched["kerameikos_place_label"] = kc.pref_label_en
            
    # Artist / Workshop
    if not enriched.get("kerameikos_artist_uri"):
        artist_term = enriched.get("artist") or ""
        kc = kerameikos_client.resolve_artist(artist_term)
        if kc:
            enriched["kerameikos_artist_uri"] = kc.uri
            enriched["kerameikos_artist_label"] = kc.pref_label_en
            enriched["kerameikos_artist_type"] = kc.concept_type

    return enriched


class LODExporter:
    """
    Exports vase metadata, classification results, and clicked scene annotations to semantic Linked Open Data formats.
    """

    @staticmethod
    def to_jsonld(vase_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Converts a list of vase records with SKOS concepts and scene annotations into a JSON-LD graph.
        Includes BCDH HECTOR, Kerameikos.org, and CIDOC CRM alignments.
        """
        context = {
            "@vocab": "http://schema.org/",
            "crm": "http://www.cidoc-crm.org/cidoc-crm/",
            "skos": "http://www.w3.org/2004/02/skos/core#",
            "dcterms": "http://purl.org/dc/terms/",
            "oa": "http://www.w3.org/ns/oa#",
            "hector": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/",
            "kon": "https://kerameikos.org/ontology#",
            "kerameikos": "https://kerameikos.org/id/",
            "prefLabel": "skos:prefLabel",
            "exactMatch": {"@id": "skos:exactMatch", "@type": "@id"},
            "shapeConcept": {"@id": "crm:P2_has_type", "@type": "@id"},
            "wareConcept": {"@id": "crm:P32_used_general_technique", "@type": "@id"},
            "hasShape": {"@id": "https://kerameikos.org/ontology#hasShape", "@type": "@id"},
            "wasProducedBy": {"@id": "crm:P108i_was_produced_by"},
            "consistsOf": {"@id": "crm:P45_consists_of", "@type": "@id"},
            "tookPlaceAt": {"@id": "crm:P7_took_place_at", "@type": "@id"},
            "carriedOutBy": {"@id": "crm:P14_carried_out_by", "@type": "@id"},
            "usedTechnique": {"@id": "crm:P32_used_general_technique", "@type": "@id"},
            "represents": {"@id": "crm:P138_represents", "@type": "@id"},
            "confidenceShape": "http://data.lodvases.bcdh.uni-bonn.de/vocab/confidenceShape",
            "confidenceWare": "http://data.lodvases.bcdh.uni-bonn.de/vocab/confidenceWare"
        }

        graph = []
        for raw_item in vase_items:
            item = _enrich_with_kerameikos(raw_item)
            obj_id = item.get("id", "unknown")
            shape_uri = item.get("shape_uri", "")
            ware_uri = item.get("ware_uri", "")
            
            node = {
                "@id": f"http://data.lodvases.bcdh.uni-bonn.de/object/{obj_id}",
                "@type": ["VisualArtwork", "crm:E22_Human-Made_Object"],
                "name": item.get("name") or f"Ancient Vase ({item.get('shape_label', 'Unknown shape')})",
                "description": item.get("description", ""),
                "artform": item.get("shape_label", ""),
                "material": item.get("material", "Ton"),
                "consistsOf": "https://kerameikos.org/id/terracotta",
                "creator": item.get("artist", "") if item.get("artist") else None,
                "dateCreated": item.get("dating", "") if item.get("dating") else None,
                "locationCreated": item.get("production_region", "") if item.get("production_region") else None,
                "contentLocation": item.get("location", "") if item.get("location") else None,
                "identifier": item.get("inventory_number") or item.get("beazley_number") or obj_id,
                "image": item.get("image_path", ""),
                "creditText": item.get("image_credit", "")
            }

            # SKOS Shape Link (BCDH HECTOR)
            if shape_uri:
                s_node = {
                    "@id": shape_uri,
                    "@type": "skos:Concept",
                    "prefLabel": item.get("shape_label", "")
                }
                if item.get("kerameikos_shape_uri"):
                    s_node["exactMatch"] = item["kerameikos_shape_uri"]
                node["shapeConcept"] = s_node

            # Kerameikos Shape Link (kon:hasShape)
            if item.get("kerameikos_shape_uri"):
                k_shape = {
                    "@id": item["kerameikos_shape_uri"],
                    "@type": ["skos:Concept", "kon:Shape"],
                    "prefLabel": item.get("kerameikos_shape_label", item.get("shape_label", ""))
                }
                if item.get("kerameikos_shape_matches"):
                    k_shape["exactMatch"] = item["kerameikos_shape_matches"]
                node["hasShape"] = k_shape

            # SKOS Ware Link (BCDH HECTOR)
            if ware_uri:
                w_node = {
                    "@id": ware_uri,
                    "@type": "skos:Concept",
                    "prefLabel": item.get("ware_label", "")
                }
                if item.get("kerameikos_technique_uri"):
                    w_node["exactMatch"] = item["kerameikos_technique_uri"]
                node["wareConcept"] = w_node

            # CIDOC CRM Production Event (crm:P108i_was_produced_by)
            has_prod = bool(
                item.get("dating") or
                item.get("kerameikos_place_uri") or
                item.get("production_region") or
                item.get("kerameikos_artist_uri") or
                item.get("artist") or
                item.get("kerameikos_technique_uri")
            )
            if has_prod:
                prod: Dict[str, Any] = {
                    "@id": f"http://data.lodvases.bcdh.uni-bonn.de/object/{obj_id}#production",
                    "@type": "crm:E12_Production"
                }
                if item.get("dating"):
                    prod["crm:P4_has_time-span"] = item["dating"]
                if item.get("kerameikos_place_uri") or item.get("production_region"):
                    p_uri = item.get("kerameikos_place_uri") or f"http://data.lodvases.bcdh.uni-bonn.de/place/{obj_id}"
                    prod["tookPlaceAt"] = {
                        "@id": p_uri,
                        "@type": "crm:E53_Place",
                        "prefLabel": item.get("kerameikos_place_label", item.get("production_region", ""))
                    }
                if item.get("kerameikos_artist_uri") or item.get("artist"):
                    a_uri = item.get("kerameikos_artist_uri") or f"http://data.lodvases.bcdh.uni-bonn.de/artist/{obj_id}"
                    is_grp = item.get("kerameikos_artist_type") == "Group"
                    prod["carriedOutBy"] = {
                        "@id": a_uri,
                        "@type": "crm:E74_Group" if is_grp else "crm:E21_Person",
                        "prefLabel": item.get("kerameikos_artist_label", item.get("artist", ""))
                    }
                if item.get("kerameikos_technique_uri"):
                    prod["usedTechnique"] = {
                        "@id": item["kerameikos_technique_uri"],
                        "@type": ["skos:Concept", "kon:Technique"],
                        "prefLabel": item.get("kerameikos_technique_label", "")
                    }
                node["wasProducedBy"] = prod

            # Scene / Iconography Annotations
            scenes = item.get("scenes", [])
            if scenes:
                node["represents"] = []
                for s in scenes:
                    s_node = {
                        "@id": s.get("uri", f"http://data.lodvases.bcdh.uni-bonn.de/concept/{s.get('label', 'scene')}"),
                        "@type": "skos:Concept",
                        "prefLabel": s.get("label", ""),
                        "category": s.get("category", "Ikonographie")
                    }
                    if s.get("confidence"):
                        s_node["confidence"] = round(float(s["confidence"]), 4)
                    if s.get("fragment"):
                        s_node["targetSelector"] = {
                            "@type": "oa:FragmentSelector",
                            "value": s["fragment"],
                            "conformsTo": "http://www.w3.org/TR/media-frags/"
                        }
                    node["represents"].append(s_node)

            # 3D Model & Geometric Dimensions
            if item.get("model_3d"):
                node["model3D"] = {
                    "@type": "3DModel",
                    "encodingFormat": "model/gltf-binary",
                    "contentUrl": item["model_3d"]
                }
            if item.get("dimensions_3d"):
                dims = item["dimensions_3d"]
                node["hasDimension"] = {
                    "@type": "crm:E54_Dimension",
                    "height": f"{dims.get('height')} cm",
                    "maxDiameter": f"{dims.get('max_diameter')} cm",
                    "rimDiameter": f"{dims.get('rim_diameter')} cm",
                    "baseDiameter": f"{dims.get('base_diameter')} cm",
                    "estimatedVolume": f"{dims.get('estimated_volume_liters')} L"
                }

            if item.get("confidence_shape"):
                node["confidenceShape"] = round(float(item["confidence_shape"]), 4)
            if item.get("confidence_ware"):
                node["confidenceWare"] = round(float(item["confidence_ware"]), 4)

            # Clean None values
            node = {k: v for k, v in node.items() if v is not None and v != ""}
            graph.append(node)

        return {
            "@context": context,
            "@graph": graph
        }

    @staticmethod
    def to_rdf_turtle(vase_items: List[Dict[str, Any]]) -> str:
        """
        Generates standard RDF Turtle (.ttl) string using RDFLib.
        Binds BCDH HECTOR, Kerameikos.org, and CIDOC CRM namespaces.
        """
        g = Graph()
        g.bind("skos", SKOS)
        g.bind("dcterms", DCTERMS)
        g.bind("schema", SCHEMA)
        g.bind("crm", CRM)
        g.bind("hector", HECTOR)
        g.bind("oa", OA)
        g.bind("kon", KON)
        g.bind("kerameikos", KERAMEIKOS)
        g.bind("obj", BASE_OBJECT)

        for raw_item in vase_items:
            item = _enrich_with_kerameikos(raw_item)
            obj_id = item.get("id", "unknown")
            obj_uri = BASE_OBJECT[obj_id]

            g.add((obj_uri, RDF.type, SCHEMA.VisualArtwork))
            g.add((obj_uri, RDF.type, CRM.E22_Human_Made_Object))

            name = item.get("name") or f"Vase {obj_id}"
            g.add((obj_uri, DCTERMS.title, Literal(name)))
            g.add((obj_uri, SCHEMA.name, Literal(name)))

            if item.get("description"):
                g.add((obj_uri, DCTERMS.description, Literal(item["description"])))
            if item.get("dating"):
                g.add((obj_uri, DCTERMS.date, Literal(item["dating"])))
            if item.get("artist"):
                g.add((obj_uri, DCTERMS.creator, Literal(item["artist"])))
            if item.get("production_region"):
                g.add((obj_uri, SCHEMA.locationCreated, Literal(item["production_region"])))
            if item.get("material"):
                g.add((obj_uri, SCHEMA.material, Literal(item["material"])))
            g.add((obj_uri, CRM.P45_consists_of, KERAMEIKOS["terracotta"]))
            if item.get("location"):
                g.add((obj_uri, SCHEMA.contentLocation, Literal(item["location"])))

            # SKOS Shape (BCDH HECTOR)
            if item.get("shape_uri"):
                s_uri = URIRef(item["shape_uri"])
                g.add((obj_uri, CRM.P2_has_type, s_uri))
                g.add((s_uri, RDF.type, SKOS.Concept))
                if item.get("shape_label"):
                    g.add((s_uri, SKOS.prefLabel, Literal(item["shape_label"], lang="de")))

            # Kerameikos Shape Link (kon:hasShape)
            if item.get("kerameikos_shape_uri"):
                k_shape = URIRef(item["kerameikos_shape_uri"])
                g.add((obj_uri, KON.hasShape, k_shape))
                g.add((k_shape, RDF.type, KON.Shape))
                g.add((k_shape, RDF.type, SKOS.Concept))
                if item.get("kerameikos_shape_label"):
                    g.add((k_shape, SKOS.prefLabel, Literal(item["kerameikos_shape_label"], lang="en")))
                if item.get("shape_uri"):
                    g.add((URIRef(item["shape_uri"]), SKOS.exactMatch, k_shape))
                for match_uri in item.get("kerameikos_shape_matches", []):
                    g.add((k_shape, SKOS.exactMatch, URIRef(match_uri)))

            # SKOS Ware (BCDH HECTOR)
            if item.get("ware_uri"):
                w_uri = URIRef(item["ware_uri"])
                g.add((obj_uri, CRM.P32_used_general_technique, w_uri))
                g.add((w_uri, RDF.type, SKOS.Concept))
                if item.get("ware_label"):
                    g.add((w_uri, SKOS.prefLabel, Literal(item["ware_label"], lang="de")))

            # Kerameikos Technique Link
            if item.get("kerameikos_technique_uri"):
                k_tech = URIRef(item["kerameikos_technique_uri"])
                g.add((k_tech, RDF.type, KON.Technique))
                g.add((k_tech, RDF.type, SKOS.Concept))
                if item.get("kerameikos_technique_label"):
                    g.add((k_tech, SKOS.prefLabel, Literal(item["kerameikos_technique_label"], lang="en")))
                if item.get("ware_uri"):
                    g.add((URIRef(item["ware_uri"]), SKOS.exactMatch, k_tech))
                for match_uri in item.get("kerameikos_technique_matches", []):
                    g.add((k_tech, SKOS.exactMatch, URIRef(match_uri)))

            # CIDOC CRM Production Event
            has_prod = bool(
                item.get("dating") or
                item.get("kerameikos_place_uri") or
                item.get("production_region") or
                item.get("kerameikos_artist_uri") or
                item.get("artist") or
                item.get("kerameikos_technique_uri")
            )
            if has_prod:
                prod_uri = URIRef(f"http://data.lodvases.bcdh.uni-bonn.de/object/{obj_id}#production")
                g.add((obj_uri, CRM.P108i_was_produced_by, prod_uri))
                g.add((prod_uri, RDF.type, CRM.E12_Production))
                if item.get("dating"):
                    g.add((prod_uri, DCTERMS.date, Literal(item["dating"])))
                if item.get("kerameikos_place_uri"):
                    k_place = URIRef(item["kerameikos_place_uri"])
                    g.add((prod_uri, CRM.P7_took_place_at, k_place))
                    g.add((k_place, RDF.type, CRM.E53_Place))
                    if item.get("kerameikos_place_label"):
                        g.add((k_place, SKOS.prefLabel, Literal(item["kerameikos_place_label"], lang="en")))
                if item.get("kerameikos_artist_uri"):
                    k_art = URIRef(item["kerameikos_artist_uri"])
                    g.add((prod_uri, CRM.P14_carried_out_by, k_art))
                    art_cls = CRM.E74_Group if item.get("kerameikos_artist_type") == "Group" else CRM.E21_Person
                    g.add((k_art, RDF.type, art_cls))
                    if item.get("kerameikos_artist_label"):
                        g.add((k_art, SKOS.prefLabel, Literal(item["kerameikos_artist_label"], lang="en")))
                if item.get("kerameikos_technique_uri"):
                    g.add((prod_uri, CRM.P32_used_general_technique, URIRef(item["kerameikos_technique_uri"])))

            # Scenes / Iconography
            for sc in item.get("scenes", []):
                if sc.get("uri"):
                    sc_uri = URIRef(sc["uri"])
                    g.add((obj_uri, CRM.P138_represents, sc_uri))
                    g.add((sc_uri, RDF.type, SKOS.Concept))
                    if sc.get("label"):
                        g.add((sc_uri, SKOS.prefLabel, Literal(sc["label"], lang="de")))

        return g.serialize(format="turtle")

    @staticmethod
    def to_dataframe(vase_items: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Creates a structured pandas DataFrame for tabular presentation and CSV export.
        Includes Kerameikos.org LOD URIs alongside BCDH HECTOR links.
        """
        rows = []
        for raw_item in vase_items:
            item = _enrich_with_kerameikos(raw_item)
            shape_uri = item.get("shape_uri", "")
            ware_uri = item.get("ware_uri", "")
            
            scenes = item.get("scenes", [])
            scenes_summary = ", ".join([f"{s.get('label')} ({s.get('uri', '')})" for s in scenes]) if scenes else "-"

            rows.append({
                "ID": item.get("id", ""),
                "Bilddatei": os.path.basename(item.get("image_path", "")) if item.get("image_path") else "",
                "Name / Bezeichnung": item.get("name", ""),
                "Erkannte Gefäßform": item.get("shape_label", ""),
                "Form SKOS-URI": shape_uri,
                "Form Konfidenz (%)": f"{item.get('confidence_shape', 0) * 100:.1f}%" if item.get("confidence_shape") is not None else "-",
                "Erkannte Ware/Stil": item.get("ware_label", ""),
                "Ware SKOS-URI": ware_uri,
                "Ware Konfidenz (%)": f"{item.get('confidence_ware', 0) * 100:.1f}%" if item.get("confidence_ware") is not None else "-",
                "Erkannte Szenen / Motive": scenes_summary,
                "Produktionsgebiet": item.get("production_region", ""),
                "Künstler / Werkstatt": item.get("artist", ""),
                "Datierung": item.get("dating", ""),
                "Material": item.get("material", "Ton"),
                "Standort": item.get("location", ""),
                "Inventar-Nr.": item.get("inventory_number", ""),
                "Beazley-Nr.": item.get("beazley_number", ""),
                "Kerameikos Form-URI": item.get("kerameikos_shape_uri", ""),
                "Kerameikos Technik-URI": item.get("kerameikos_technique_uri", ""),
                "Kerameikos Herkunft-URI": item.get("kerameikos_place_uri", ""),
                "Kerameikos Künstler-URI": item.get("kerameikos_artist_uri", ""),
                "Skosmos Link (Form)": f"https://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/de/page/{shape_uri.split('/')[-1]}" if "concept_" in shape_uri else shape_uri,
                "Skosmos Link (Ware)": f"https://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/de/page/{ware_uri.split('/')[-1]}" if "concept_" in ware_uri else ware_uri
            })
        return pd.DataFrame(rows)

    @staticmethod
    def to_enriched_txt(raw_item: Dict[str, Any]) -> str:
        """
        Generates enriched text metadata format compatible with pics/*.txt and bapd_corpus/*.txt.
        Includes BCDH SKOS and Kerameikos URIs.
        """
        item = _enrich_with_kerameikos(raw_item)
        lines = []
        if item.get("image_credit"):
            lines.append(f"Abbildungsnachweis: {item['image_credit']}")
        if item.get("beazley_number"):
            lines.append(f"Beazley-Nummer: {item['beazley_number']}")
        if item.get("description"):
            lines.append(f"Beschreibung: {item['description']}")
        if item.get("dating"):
            lines.append(f"Datierung: {item['dating']}")
        if item.get("findspot"):
            lines.append(f"Fundkontext: {item['findspot']}")
        lines.append("Gattung: Gefäß")
        
        shape_str = item.get("shape_label", "")
        if item.get("shape_uri"):
            shape_str += f" [SKOS: {item['shape_uri']}]"
        if item.get("kerameikos_shape_uri"):
            shape_str += f" [Kerameikos: {item['kerameikos_shape_uri']}]"
        lines.append(f"Gefäßform: {shape_str}")

        if item.get("findspot"):
            lines.append(f"Herkunftsort: {item['findspot']}")
        if item.get("inventory_number"):
            lines.append(f"Inventar-Nr.: {item['inventory_number']}")
            
        artist_str = item.get("artist", "")
        if item.get("kerameikos_artist_uri"):
            artist_str += f" [Kerameikos: {item['kerameikos_artist_uri']}]"
        if artist_str:
            lines.append(f"Künstler/Werkstatt: {artist_str}")
            
        if item.get("material"):
            lines.append(f"Material: {item['material']}")
        if item.get("name"):
            lines.append(f"Name/Bezeichnung: {item['name']}")
            
        region_str = item.get("production_region", "")
        if item.get("kerameikos_place_uri"):
            region_str += f" [Kerameikos: {item['kerameikos_place_uri']}]"
        if region_str:
            lines.append(f"Produktionsgebiet: {region_str}")
            
        ware_str = item.get("ware_label", "")
        if item.get("ware_uri"):
            ware_str += f" [SKOS: {item['ware_uri']}]"
        if item.get("kerameikos_technique_uri"):
            ware_str += f" [Kerameikos: {item['kerameikos_technique_uri']}]"
        if ware_str:
            lines.append(f"Ware/Stil: {ware_str}")

        # Scenes
        scenes = item.get("scenes", [])
        if scenes:
            scene_strs = [f"{s.get('label')} [SKOS: {s.get('uri')}]" for s in scenes if s.get('label')]
            lines.append(f"Ikonographie/Szenen: {'; '.join(scene_strs)}")
            
        if item.get("location"):
            lines.append(f"Standort: {item['location']}")
            
        return "\n".join(lines)
