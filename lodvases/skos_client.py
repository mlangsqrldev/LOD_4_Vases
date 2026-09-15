"""
SKOS Client for BCDH HECTOR Vocabulary.
Interacts with the Skosmos REST API (https://vocabs.bcdh.uni-bonn.de/rest/v1/)
and provides an offline/fallback knowledge base for ancient Greek vase concepts.
"""

import json
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field

SKOSMOS_API_BASE = "https://vocabs.bcdh.uni-bonn.de/rest/v1"
DEFAULT_VOCAB = "hector_heritage_assets"
VOCAB_WEB_BASE = "https://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/de/page"

@dataclass
class SKOSConcept:
    uri: str
    pref_label: str
    vocab: str = DEFAULT_VOCAB
    alt_labels: List[str] = field(default_factory=list)
    broader_uri: Optional[str] = None
    broader_label: Optional[str] = None
    narrower_concepts: List[Dict[str, str]] = field(default_factory=list)
    definition: Optional[str] = None
    notation: Optional[str] = None
    category: Optional[str] = None # e.g. "Gefäßform", "Ware/Stil", "Material", "Epoche"

    @property
    def local_id(self) -> str:
        return self.uri.rstrip("/").split("/")[-1]

    @property
    def web_url(self) -> str:
        if "hector_heritage_assets" in self.uri:
            return f"https://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/de/page/{self.local_id}"
        elif "hector_epochs" in self.uri:
            return f"https://vocabs.bcdh.uni-bonn.de/hector_epochs/de/page/{self.local_id}"
        return self.uri

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["local_id"] = self.local_id
        d["web_url"] = self.web_url
        return d


# Comprehensive built-in curated knowledge base of HECTOR concepts for instant lookup & offline fallback
CURATED_HECTOR_CONCEPTS: Dict[str, Dict[str, Any]] = {
    # --- Gefäßformen (Vessel Shapes) ---
    "Lekythos": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_5b62f4af",
        "pref_label": "Lekythos",
        "category": "Gefäßform",
        "broader_label": "Salbgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_12149e99",
        "alt_labels": ["Lekythoi", "Ölfläschchen", "Schulterlekythos", "Bauchlekythos"],
        "definition": "Schlankes antikes griechisches Gefäß zur Aufbewahrung von Salböl, oft als Grabbeigabe verwendet."
    },
    "Schulterlekythos": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_4894cb93",
        "pref_label": "Schulterlekythos",
        "category": "Gefäßform",
        "broader_label": "Lekythos",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_5b62f4af",
        "alt_labels": ["Shoulder lekythos"]
    },
    "Bauchlekythos": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_7a3be9d5",
        "pref_label": "Bauchlekythos",
        "category": "Gefäßform",
        "broader_label": "Lekythos",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_5b62f4af",
        "alt_labels": ["Squat lekythos"]
    },
    "Psykter": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_c73fec47",
        "pref_label": "Psykter",
        "category": "Gefäßform",
        "broader_label": "Kühlgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_14a0274a",
        "alt_labels": ["Weinkühler", "Cooler"],
        "definition": "Pilzförmiges griechisches Gefäß zum Kühlen von Wein im Krater."
    },
    "Trinkschale": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b1d6ebaf",
        "pref_label": "Trinkschale",
        "category": "Gefäßform",
        "broader_label": "Trinkgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_cd3210b2",
        "alt_labels": ["Kylix", "Schale", "Kylikes", "Drinking cup"],
        "definition": "Flache Trinkschale mit zwei Henkeln und meist hohem Fuß, zentrales Gefäß des Symposions."
    },
    "Kylix": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b1d6ebaf",
        "pref_label": "Trinkschale",
        "category": "Gefäßform",
        "broader_label": "Trinkgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_cd3210b2",
        "alt_labels": ["Kylix", "Schale Typ A", "Schale Typ B", "Schale Typ C"],
        "definition": "Flache Trinkschale mit zwei Henkeln (Kylix)."
    },
    "Schale Typ A": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b7a46c4b",
        "pref_label": "Schale Typ A",
        "category": "Gefäßform",
        "broader_label": "Trinkschale",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b1d6ebaf",
        "alt_labels": ["Type A cup"]
    },
    "Schale Typ B": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_574f86fb",
        "pref_label": "Schale Typ B",
        "category": "Gefäßform",
        "broader_label": "Trinkschale",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b1d6ebaf",
        "alt_labels": ["Type B cup"]
    },
    "Schale Typ C": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_4ed9cd1f",
        "pref_label": "Schale Typ C",
        "category": "Gefäßform",
        "broader_label": "Trinkschale",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b1d6ebaf",
        "alt_labels": ["Type C cup"]
    },
    "Augenschale": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b1f225cc",
        "pref_label": "Augenschale",
        "category": "Gefäßform",
        "broader_label": "Trinkschale",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b1d6ebaf",
        "alt_labels": ["Eye cup"]
    },
    "Stamnos": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_f17f817a",
        "pref_label": "Stamnos",
        "category": "Gefäßform",
        "broader_label": "Gefäßtypen",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_89023780",
        "alt_labels": ["Stamnoi", "Vorrats- und Mischgefäß"],
        "definition": "Bauchiges Gefäß mit zwei horizontalen Henkeln und flachem Hals zum Mischen und Aufbewahren von Wein."
    },
    "Krater": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_4ec3b0ea",
        "pref_label": "Krater",
        "category": "Gefäßform",
        "broader_label": "Mischgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_63b8b6a3",
        "alt_labels": ["Mischgefäß", "Mixing bowl"],
        "definition": "Großes antikes Gefäß zum Mischen von Wein und Wasser beim Gastmahl."
    },
    "Glockenkrater": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b7477658",
        "pref_label": "Glockenkrater",
        "category": "Gefäßform",
        "broader_label": "Krater",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_4ec3b0ea",
        "alt_labels": ["Bell krater"]
    },
    "Kelchkrater": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_33af97ce",
        "pref_label": "Kelchkrater",
        "category": "Gefäßform",
        "broader_label": "Krater",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_4ec3b0ea",
        "alt_labels": ["Calyx krater"]
    },
    "Volutenkrater": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_1d7db261",
        "pref_label": "Volutenkrater",
        "category": "Gefäßform",
        "broader_label": "Krater",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_4ec3b0ea",
        "alt_labels": ["Volute krater"]
    },
    "Kolonettenkrater": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_088f05e0",
        "pref_label": "Kolonettenkrater",
        "category": "Gefäßform",
        "broader_label": "Krater",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_4ec3b0ea",
        "alt_labels": ["Column krater"]
    },
    "Transportamphora": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_65909227",
        "pref_label": "Transportamphora",
        "category": "Gefäßform",
        "broader_label": "Transportgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_22973848",
        "alt_labels": ["Amphora", "Amphore", "Bauchamphora", "Halsamphora"],
        "definition": "Zweihenkliges Vorrats- und Transportgefäß mit ovalem oder eiförmigem Bauch."
    },
    "Amphora": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_65909227",
        "pref_label": "Transportamphora",
        "category": "Gefäßform",
        "broader_label": "Transportgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_22973848",
        "alt_labels": ["Amphora", "Bauchamphora", "Halsamphora", "Nikosthenische Amphora"]
    },
    "Hydria": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_bf070035",
        "pref_label": "Hydria",
        "category": "Gefäßform",
        "broader_label": "Gefäßtypen",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_89023780",
        "alt_labels": ["Wassergefäß", "Kalpis"],
        "definition": "Dreihenkliges griechisches Wassergefäß mit zwei horizontalen Tragehenkeln und einem vertikalen Schöpfhenkel."
    },
    "Skyphos": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_9961a74e",
        "pref_label": "Skyphos",
        "category": "Gefäßform",
        "broader_label": "Trinkgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_cd3210b2",
        "alt_labels": ["Trinkbecher", "Skyphoi"],
        "definition": "Tiefer zweihenkliger Trinkbecher."
    },
    "Kantharos": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b20b6954",
        "pref_label": "Kantharos",
        "category": "Gefäßform",
        "broader_label": "Trinkgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_cd3210b2",
        "alt_labels": ["Dionysosbecher", "High-handled cup"],
        "definition": "Trinkgefäß mit zwei hoch über den Rand gezogenen Schlaufenhenkeln, Attribut des Gottes Dionysos."
    },
    "Pyxis": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_f41eb9e9",
        "pref_label": "Pyxis",
        "category": "Gefäßform",
        "broader_label": "Aufbewahrungsgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_4aff40b4",
        "alt_labels": ["Büchse", "Schmuckdose"],
        "definition": "Runde Deckelbüchse zur Aufbewahrung von Schmuck oder Kosmetika."
    },
    "Dinos": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_2b10bc00",
        "pref_label": "Dinos",
        "category": "Gefäßform",
        "broader_label": "Mischgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_63b8b6a3",
        "alt_labels": ["Lebes", "Kessel"]
    },
    "Aryballos": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_8041bbd3",
        "pref_label": "Arybalos",
        "category": "Gefäßform",
        "broader_label": "Salbgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_12149e99",
        "alt_labels": ["Aryballos", "Kugelaryballos"]
    },
    "Alabastron": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_7643ba12",
        "pref_label": "Alabastron",
        "category": "Gefäßform",
        "broader_label": "Salbgefäß",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_12149e99",
        "alt_labels": ["Salbfläschchen"]
    },
    "Pelike": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_89023780", # under Gefäßtypen
        "pref_label": "Pelike",
        "category": "Gefäßform",
        "broader_label": "Gefäßtypen",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_89023780",
        "alt_labels": ["Tropfenförmige Amphora"]
    },
    "Oinochoe": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_fed2da29",
        "pref_label": "Oinochoe",
        "category": "Gefäßform",
        "broader_label": "Kanne",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_853e9944",
        "alt_labels": ["Weinkanne", "Olpe"]
    },

    # --- Waren, Stile & Keramikgattungen (Wares & Styles) ---
    "Attisch Rotfigurig": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_69d3cf4c",
        "pref_label": "Attisch Rotfigurig",
        "category": "Ware/Stil",
        "broader_label": "Rotfiguriger Stil",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_307f1d16",
        "alt_labels": ["Attic red-figure", "rotfigurig", "Attisch-rotfigurig", "rotfigurige Vasenmalerei"],
        "definition": "Vasenmalerei-Stil aus Attika, bei dem die Figuren im tongrundigen Rot ausgespart und der Hintergrund schwarz glänzend gefirnisst wird."
    },
    "Attisch Schwarzfigurig": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_fd4cbaed",
        "pref_label": "Attisch schwarzfigurig",
        "category": "Ware/Stil",
        "broader_label": "Schwarzfiguriger Stil",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_8c223ea8",
        "alt_labels": ["Attic black-figure", "schwarzfigurig", "Attisch-schwarzfigurig"],
        "definition": "Vasenmalerei-Stil aus Attika, bei dem die Figuren mit schwarzem Glanzton gemalt und Konturen/Details eingeritzt werden."
    },
    "Attisch Weißgrundig": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_45093154",
        "pref_label": "Attisch Weißgrundig",
        "category": "Ware/Stil",
        "broader_label": "Weißgrundige Vase",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_6b02e968",
        "alt_labels": ["Attic white-ground", "weißgrundig", "White ground"],
        "definition": "Vasenmalerei-Technik mit weißem Kreide- oder Tonschlickerüberzug, typisch für attische Grab-Lekythen."
    },
    "Attische Keramik": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_2ad94740",
        "pref_label": "Attische Keramik",
        "category": "Ware/Stil",
        "broader_label": "Griechische Keramik Landschaft",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b9ae0dc3",
        "alt_labels": ["Attika", "Attisch", "Attic pottery"]
    },
    "Attika": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_2ad94740",
        "pref_label": "Attische Keramik",
        "category": "Ware/Stil",
        "broader_label": "Griechische Keramik Landschaft",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b9ae0dc3",
        "alt_labels": ["Attika", "Attisch"]
    },
    "Apulisch": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b5d694d6",
        "pref_label": "Unteritalisch Rotfigurig",
        "category": "Ware/Stil",
        "broader_label": "Rotfiguriger Stil",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_307f1d16",
        "alt_labels": ["Apulisch", "Apulisch Rotfigurig"]
    },
    "Unteritalisch Rotfigurig": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_b5d694d6",
        "pref_label": "Unteritalisch Rotfigurig",
        "category": "Ware/Stil",
        "broader_label": "Rotfiguriger Stil",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_307f1d16",
        "alt_labels": ["Apulisch Rotfigurig", "Lukanisch Rotfigurig", "Kampanisch Rotfigurig", "South Italian red-figure"]
    },
    "Attisch geometrisch": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_1d64ee12",
        "pref_label": "Attisch geometrisch",
        "category": "Ware/Stil",
        "broader_label": "Geometrische Keramik",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_5e25cead",
        "alt_labels": ["Attic geometric", "Geometrisch"]
    },
    "Korinthisch Schwarzfigurig": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_8bf08032",
        "pref_label": "Korinthisch Schwarzfigurig",
        "category": "Ware/Stil",
        "broader_label": "Schwarzfiguriger Stil",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_8c223ea8",
        "alt_labels": ["Corinthian black-figure"]
    },
    "Rotfigurig": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_307f1d16",
        "pref_label": "Rotfiguriger Stil",
        "category": "Ware/Stil",
        "broader_label": "Feinkeramik",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_881e389c",
        "alt_labels": ["Rotfigurig", "Red-figure"]
    },
    "Schwarzfigurig": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_8c223ea8",
        "pref_label": "Schwarzfiguriger Stil",
        "category": "Ware/Stil",
        "broader_label": "Feinkeramik",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_881e389c",
        "alt_labels": ["Schwarzfigurig", "Black-figure"]
    },

    # --- Oberkategorien (Broad Vessel Classes) ---
    "Gefäß": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_43d07f2a",
        "pref_label": "Gefäß",
        "category": "Gattung",
        "broader_label": "Kulturgut",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_aeb4b812"
    },
    "Keramik": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_6cf5b92a",
        "pref_label": "Keramik",
        "category": "Gattung/Material",
        "broader_label": "Kulturgut",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_aeb4b812"
    },
    "Trinkgefäß": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_cd3210b2",
        "pref_label": "Trinkgefäß",
        "category": "Funktion",
        "broader_label": "Gefäßtypen",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_89023780"
    },
    "Mischgefäß": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_63b8b6a3",
        "pref_label": "Mischgefäß",
        "category": "Funktion",
        "broader_label": "Gefäßtypen",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_89023780"
    },
    "Salbgefäß": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_12149e99",
        "pref_label": "Salbgefäß",
        "category": "Funktion",
        "broader_label": "Gefäßtypen",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_89023780"
    },
    "Kühlgefäß": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_14a0274a",
        "pref_label": "Kühlgefäß",
        "category": "Funktion",
        "broader_label": "Gefäßtypen",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_89023780"
    },
    "Aufbewahrungsgefäß": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_4aff40b4",
        "pref_label": "Aufbewahrungsgefäß",
        "category": "Funktion",
        "broader_label": "Gefäßtypen",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_89023780"
    },

    # --- Ikonographie, Figuren & Szenen (Iconography & Scenes) ---
    "Herakles": {
        "uri": "https://hector.bcdh.uni-bonn.de/heracles",
        "pref_label": "Herakles",
        "category": "Mythologie/Figur",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Herakles", "Heracles", "Herkules", "Herakles-Sage"],
        "definition": "Griechischer Nationalheld und Sohn des Zeus, berühmt für die 12 Taten und dargestellt mit Löwenfell und Keule."
    },
    "Dionysos": {
        "uri": "https://hector.bcdh.uni-bonn.de/dionysus",
        "pref_label": "Dionysos",
        "category": "Mythologie/Gott",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Dionysus", "Bacchus", "Weingott"],
        "definition": "Gott des Weines, der Freude, der Trauben und der Ekstase, oft mit Kantharos und Thyrsosstab abgebildet."
    },
    "Satyr": {
        "uri": "https://hector.bcdh.uni-bonn.de/satyr",
        "pref_label": "Satyr",
        "category": "Mythologie/Wesen",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Satyroi", "Silen", "Satyros", "Satyriskion"],
        "definition": "Dämonisches Mischwesen aus dem Gefolge des Dionysos mit Pferdeschwanz und spitzen Ohren."
    },
    "Mänade": {
        "uri": "https://hector.bcdh.uni-bonn.de/maenads",
        "pref_label": "Mänaden",
        "category": "Mythologie/Wesen",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Mänade", "Maenad", "Bakchantin", "Thyiade"],
        "definition": "Rasende weibliche Begleiterinnen des Dionysos, in orgiastischem Tanz mit Thyrsos und Pantherfell."
    },
    "Eros": {
        "uri": "https://hector.bcdh.uni-bonn.de/eros",
        "pref_label": "Eros",
        "category": "Mythologie/Gott",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Eroten", "Amor", "Flügelknabe"],
        "definition": "Gott der begehrlichen Liebe, dargestellt als geflügelter Knabe oder Jüngling."
    },
    "Athena": {
        "uri": "https://hector.bcdh.uni-bonn.de/athena",
        "pref_label": "Athena",
        "category": "Mythologie/Göttin",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Athene", "Minerva", "Pallas Athena"],
        "definition": "Göttin der Weisheit, der Kriegskunst und Stadtgöttin Athens, mit Helm, Lanze und Aigis."
    },
    "Apollon": {
        "uri": "https://hector.bcdh.uni-bonn.de/apollo",
        "pref_label": "Apollon",
        "category": "Mythologie/Gott",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Apollo", "Phoibos Apollon"],
        "definition": "Gott des Lichts, der Musik, der Dichtkunst und der Weissagung, mit Kithara oder Bogen."
    },
    "Krieger / Hoplit": {
        "uri": "https://hector.bcdh.uni-bonn.de/warrior_group",
        "pref_label": "Krieger",
        "category": "Ikonographie/Figur",
        "broader_label": "Figur",
        "alt_labels": ["Krieger", "Hoplit", "Soldat", "Krieger-Gruppe", "Kriegerfries"],
        "definition": "Schwerbewaffneter griechischer Fußsoldat mit korinthischem Helm, Rundschild (Aspis) und Lanze."
    },
    "Reiter / Delphinreiter": {
        "uri": "https://hector.bcdh.uni-bonn.de/dolphin_group",
        "pref_label": "Delphinreiter / Reiter",
        "category": "Ikonographie/Szene",
        "broader_label": "Figur",
        "alt_labels": ["Delphinreiter", "Reiter", "Delphin", "Pferdereiter", "Reiter-Gruppe"],
        "definition": "Bewaffneter Reiter auf einem Reittier oder Delphin, bekannt aus der archaischen attischen Vasenmalerei."
    },
    "Athlet": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/LVR_Fundansprachen/figur_visuelles_werk",
        "pref_label": "Athlet",
        "category": "Ikonographie/Figur",
        "broader_label": "Figur",
        "alt_labels": ["Athlet", "Ringer", "Läufer", "Diskuswerfer", "Palästra"],
        "definition": "Nackter antiker Sportler beim Wettkampf oder Training im Gymnasion."
    },
    "Symposion / Zecher": {
        "uri": "https://hector.bcdh.uni-bonn.de/symposium",
        "pref_label": "Symposion",
        "category": "Ikonographie/Szene",
        "broader_label": "Kulturgut",
        "alt_labels": ["Symposion", "Symposium", "Gastmahl", "Zecher", "Kottabos"],
        "definition": "Männliches Trinkgelage und zentrales gesellschaftliches Ereignis der griechischen Antike."
    },
    "Komast / Tänzer": {
        "uri": "https://hector.bcdh.uni-bonn.de/komast_group",
        "pref_label": "Komast",
        "category": "Ikonographie/Figur",
        "broader_label": "Figur",
        "alt_labels": ["Komast", "Komasten", "Tänzer", "Komos"],
        "definition": "Trunkener Zecher und Tänzer bei einem festlichen Umzug (Komos)."
    },
    "Amazonen": {
        "uri": "https://hector.bcdh.uni-bonn.de/amazons",
        "pref_label": "Amazonen",
        "category": "Mythologie/Wesen",
        "broader_label": "Mythologie",
        "alt_labels": ["Amazone", "Amazonomachie", "Amazonenkrieg"],
        "definition": "Mythologisches Volk kriegerischer Frauen."
    },
    "Tier / Löwe / Stier": {
        "uri": "https://hector.bcdh.uni-bonn.de/animals",
        "pref_label": "Tiere",
        "category": "Ikonographie/Motiv",
        "broader_label": "Kulturgut",
        "alt_labels": ["Tier", "Löwe", "Stier", "Pferd", "Tierfries"],
        "definition": "Wilde oder gezähmte Tiere in Fabel- und Tierfriesdarstellungen."
    },
    "Delphinreiter": {
        "uri": "https://hector.bcdh.uni-bonn.de/dolphin_group",
        "pref_label": "Delphinreiter",
        "category": "Ikonographie/Szene",
        "broader_label": "Figur",
        "alt_labels": ["Delphinreiter", "Reiter", "Delphin", "Pferdereiter", "Reiter-Gruppe", "Krieger auf Delphinen"],
        "definition": "Bewaffneter Reiter auf einem Reittier oder Delphin, bekannt aus der archaischen attischen Vasenmalerei."
    },
    "Satyr / Silen": {
        "uri": "https://hector.bcdh.uni-bonn.de/satyr",
        "pref_label": "Satyr / Silen",
        "category": "Mythologie/Wesen",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Satyr", "Satyroi", "Silen", "Silene", "Satyros", "Satyriskion"],
        "definition": "Dämonisches Mischwesen aus dem Gefolge des Dionysos mit Pferdeschwanz und spitzen Ohren."
    },
    "Herakles & Acheloos": {
        "uri": "https://hector.bcdh.uni-bonn.de/heracles",
        "pref_label": "Herakles & Acheloos",
        "category": "Mythologie/Szene",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Herakles und Acheloos", "Ringkampf Herakles Acheloos", "Acheloos"],
        "definition": "Der Kampf des Herakles gegen den Flussgott Acheloos um Deïaneira."
    },
    "Herakles (Löwenfell & Keule)": {
        "uri": "https://hector.bcdh.uni-bonn.de/heracles",
        "pref_label": "Herakles",
        "category": "Mythologie/Figur",
        "broader_label": "Mythologie",
        "broader_uri": "https://hector.bcdh.uni-bonn.de/mythology",
        "alt_labels": ["Herakles", "Heracles", "Herkules", "Nemeischer Löwe"],
        "definition": "Griechischer Nationalheld mit Attributen Löwenfell, Keule, Bogen oder Kantharos."
    },
    "Theseus & Amazone": {
        "uri": "https://hector.bcdh.uni-bonn.de/amazons",
        "pref_label": "Theseus & Antiope (Amazonomachie)",
        "category": "Mythologie/Szene",
        "broader_label": "Mythologie",
        "alt_labels": ["Theseus und Antiope", "Amazone", "Amazonomachie", "Amazonenkampf"],
        "definition": "Theseus entführt die Amazonenkönigin Antiope oder kämpft gegen Amazonen."
    },
    "Achill & Hektor": {
        "uri": "https://hector.bcdh.uni-bonn.de/homer",
        "pref_label": "Achill, Hektor & Priamos",
        "category": "Ilias/Szene",
        "broader_label": "Mythologie",
        "alt_labels": ["Achill und Hektor", "Priamos erbittet Leiche des Hektors von Achill", "Achilleus"],
        "definition": "Homerische Szene: König Priamos erbittet vom lagernden Achill den Leichnam seines Sohnes Hektor."
    },
    "Zeus & Götterversammlung": {
        "uri": "https://hector.bcdh.uni-bonn.de/olympians",
        "pref_label": "Zeus & Götterversammlung (Quadriga)",
        "category": "Mythologie/Szene",
        "broader_label": "Mythologie",
        "alt_labels": ["Zeus Quadriga", "Abfahrt des Zeus", "Götterversammlung", "Viergespann"],
        "definition": "Abfahrt des Zeus vom Olymp im Viergespann in Anwesenheit der olympischen Götter."
    },
    "Hermes": {
        "uri": "https://hector.bcdh.uni-bonn.de/hermes",
        "pref_label": "Hermes",
        "category": "Mythologie/Gott",
        "broader_label": "Mythologie",
        "alt_labels": ["Hermes", "Kerykeion", "Flügelschuhe", "Götterbote"],
        "definition": "Götterbote Hermes mit Flügelschuhen, Reisehut (Petasos) und Heroldstab (Kerykeion)."
    },
    "Aphrodite & Ares": {
        "uri": "https://hector.bcdh.uni-bonn.de/aphrodite",
        "pref_label": "Aphrodite & Ares",
        "category": "Mythologie/Szene",
        "broader_label": "Mythologie",
        "alt_labels": ["Aphrodite und Ares", "Liebesgöttin Aphrodite"],
        "definition": "Göttin der Liebe Aphrodite zusammen mit dem Kriegsgott Ares."
    },
    "Nereus & Nereide": {
        "uri": "https://hector.bcdh.uni-bonn.de/sea_deities",
        "pref_label": "Nereus & Nereide",
        "category": "Mythologie/Wesen",
        "broader_label": "Mythologie",
        "alt_labels": ["Nereus", "Nereide", "Meeresgott", "Meeresnymphe"],
        "definition": "Der greise Meeresgott Nereus mit Fischleib und Meeresnymphen (Nereiden)."
    },
    "Kentaur Chiron": {
        "uri": "https://hector.bcdh.uni-bonn.de/centaur",
        "pref_label": "Kentaur Chiron & Achill",
        "category": "Mythologie/Wesen",
        "broader_label": "Mythologie",
        "alt_labels": ["Chiron", "Kentaur", "Centaur", "Pferdemensch"],
        "definition": "Der weise Zentaur Chiron unterrichtet den jungen Helden Achill."
    },
    "Jüngling mit Ball / Pferd": {
        "uri": "https://hector.bcdh.uni-bonn.de/youth",
        "pref_label": "Jüngling mit Ball / Reiter",
        "category": "Ikonographie/Figur",
        "broader_label": "Figur",
        "alt_labels": ["Jüngling mit Ball", "Krieger mit Pferd", "Geranomachie"],
        "definition": "Griechischer Jüngling im Spiel oder Krieger zu Pferde."
    },
    "Palmette & Mäander": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_25df5832",
        "pref_label": "Palmette & Mäander",
        "category": "Ornament",
        "broader_label": "Ornament",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_25df5832",
        "alt_labels": ["Palmette", "Mäander", "Ornamentband", "Mäanderfries"],
        "definition": "Klassische antike Zierbänder mit Palmettenfächern und rechtwinkligen Mäandern."
    },
    "Ornament": {
        "uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_25df5832",
        "pref_label": "Ornament",
        "category": "Ornament",
        "broader_label": "Kulturgut",
        "broader_uri": "http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_aeb4b812",
        "alt_labels": ["Verzierung", "Bordüre"]
    }
}


class SKOSClient:
    """
    Client for querying and caching SKOS concepts from the BCDH Skosmos API.
    """
    def __init__(self, vocab: str = DEFAULT_VOCAB, use_live_api: bool = True):
        self.vocab = vocab
        self.use_live_api = use_live_api
        self.cache: Dict[str, SKOSConcept] = {}
        self._load_curated_concepts()

    def _load_curated_concepts(self):
        for key, data in CURATED_HECTOR_CONCEPTS.items():
            concept = SKOSConcept(
                uri=data["uri"],
                pref_label=data.get("pref_label", key),
                vocab=self.vocab,
                alt_labels=data.get("alt_labels", []),
                broader_uri=data.get("broader_uri"),
                broader_label=data.get("broader_label"),
                definition=data.get("definition"),
                category=data.get("category")
            )
            self.cache[key.lower()] = concept
            self.cache[data["uri"]] = concept
            for alt in data.get("alt_labels", []):
                self.cache[alt.lower()] = concept

    def search_live(self, query: str, lang: str = "de") -> List[SKOSConcept]:
        """
        Executes a live search against the Skosmos REST API.
        """
        if not self.use_live_api or not query.strip():
            return []
        
        encoded_q = urllib.parse.quote(query.strip())
        url = f"{SKOSMOS_API_BASE}/{self.vocab}/search?query={encoded_q}*&lang={lang}"
        req = urllib.request.Request(url, headers={"User-Agent": "LODVases-AI/1.0", "Accept": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                results = []
                for item in data.get("results", []):
                    uri = item.get("uri", "")
                    pref_label = item.get("prefLabel", "")
                    c = SKOSConcept(
                        uri=uri,
                        pref_label=pref_label,
                        vocab=item.get("vocab", self.vocab),
                        alt_labels=[item.get("altLabel")] if item.get("altLabel") else []
                    )
                    results.append(c)
                    self.cache[uri] = c
                    self.cache[pref_label.lower()] = c
                return results
        except Exception as e:
            # Fallback smoothly to cached concepts
            return []

    def get_concept_by_label(self, label: str) -> Optional[SKOSConcept]:
        """
        Resolves a term (e.g. "Lekythos", "Psykter", "Attisch Rotfigurig") to a SKOS concept.
        Checks cache, curated thesaurus, and queries the live Skosmos API if not cached.
        """
        if not label:
            return None
        
        clean_label = label.strip()
        lower_label = clean_label.lower()

        # 1. Exact match in cache / curated list
        if lower_label in self.cache:
            return self.cache[lower_label]

        # 2. Fuzzy / partial match in curated list
        for key, concept in self.cache.items():
            if not key.startswith("http") and (key in lower_label or lower_label in key):
                return concept

        # 3. Live search query
        live_res = self.search_live(clean_label)
        if live_res:
            return live_res[0]

        # 4. Fallback: Generic Concept with simulated URI if not found
        return SKOSConcept(
            uri=f"http://vocabs.bcdh.uni-bonn.de/hector_heritage_assets/concept_custom_{urllib.parse.quote(lower_label)}",
            pref_label=clean_label,
            category="Erkannter Begriff"
        )

    def resolve_prediction(self, shape_name: str, ware_name: str) -> Dict[str, Optional[SKOSConcept]]:
        """
        Resolves both shape and ware predictions into SKOS concepts.
        """
        shape_concept = self.get_concept_by_label(shape_name)
        ware_concept = self.get_concept_by_label(ware_name)
        return {
            "shape": shape_concept,
            "ware": ware_concept
        }

    def get_all_concepts_list(self) -> List[SKOSConcept]:
        """
        Returns list of all unique curated concepts.
        """
        unique = {}
        for c in self.cache.values():
            if c.uri not in unique:
                unique[c.uri] = c
        return list(unique.values())


# Global singleton instance
skos_client = SKOSClient()
