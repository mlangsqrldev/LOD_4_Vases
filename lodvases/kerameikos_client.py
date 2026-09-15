"""
Kerameikos Client: Resolves Greek pottery shapes, techniques, production places, and artists/workshops
to canonical Linked Open Data concepts from Kerameikos.org (W3C SKOS, CIDOC-CRM, Getty AAT, Wikidata).
"""

import os
import re
import json
import urllib.request
import urllib.parse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


KERAMEIKOS_ID_BASE = "https://kerameikos.org/id/"
KERAMEIKOS_ONTOLOGY = "https://kerameikos.org/ontology#"
KERAMEIKOS_SPARQL = "https://kerameikos.org/query"


@dataclass
class KerameikosConcept:
    uri: str
    pref_label_en: str
    pref_label_de: str
    concept_type: str  # Shape, Technique, ProductionPlace, Person, Group
    definition: Optional[str] = None
    exact_matches: List[str] = field(default_factory=list)
    broader: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uri": self.uri,
            "pref_label_en": self.pref_label_en,
            "pref_label_de": self.pref_label_de,
            "concept_type": self.concept_type,
            "definition": self.definition,
            "exact_matches": self.exact_matches,
            "broader": self.broader
        }


# Curated offline mapping: Shape German/English -> Kerameikos Concept ID
KERAMEIKOS_SHAPES: Dict[str, Dict[str, Any]] = {
    "kylix": {
        "id": "kylix",
        "en": "Kylix",
        "de": "Kylix",
        "matches": [
            "http://vocab.getty.edu/aat/300198842",
            "http://www.wikidata.org/entity/Q668349",
            "http://d-nb.info/gnd/4715607-7",
            "https://www.britishmuseum.org/collection/term/x7595"
        ],
        "broader": "cup"
    },
    "trinkschale": {
        "id": "kylix",
        "en": "Kylix",
        "de": "Kylix",
        "matches": ["http://vocab.getty.edu/aat/300198842"],
        "broader": "cup"
    },
    "halsamphora": {
        "id": "neck_amphora",
        "en": "Neck Amphora",
        "de": "Halsamphora",
        "matches": [
            "http://vocab.getty.edu/aat/300148700",
            "http://www.wikidata.org/entity/Q1313645"
        ],
        "broader": "amphora"
    },
    "bauchamphora": {
        "id": "belly_amphora",
        "en": "Belly Amphora",
        "de": "Bauchamphora",
        "matches": [
            "http://vocab.getty.edu/aat/300148701",
            "http://www.wikidata.org/entity/Q1313646"
        ],
        "broader": "amphora"
    },
    "panathenäische preisamphore": {
        "id": "panathenaic_amphora",
        "en": "Panathenaic Amphora",
        "de": "Panathenäische Preisamphore",
        "matches": [
            "http://vocab.getty.edu/aat/300198858",
            "http://www.wikidata.org/entity/Q1262521"
        ],
        "broader": "amphora"
    },
    "panathenäische amphora": {
        "id": "panathenaic_amphora",
        "en": "Panathenaic Amphora",
        "de": "Panathenäische Preisamphore",
        "matches": ["http://vocab.getty.edu/aat/300198858"],
        "broader": "amphora"
    },
    "amphora": {
        "id": "amphora",
        "en": "Amphora",
        "de": "Amphora",
        "matches": [
            "http://vocab.getty.edu/aat/300148682",
            "http://www.wikidata.org/entity/Q178401",
            "http://d-nb.info/gnd/4001770-9"
        ]
    },
    "amphore": {
        "id": "amphora",
        "en": "Amphora",
        "de": "Amphora",
        "matches": ["http://vocab.getty.edu/aat/300148682"]
    },
    "hydria": {
        "id": "hydria",
        "en": "Hydria",
        "de": "Hydria",
        "matches": [
            "http://vocab.getty.edu/aat/300198837",
            "http://www.wikidata.org/entity/Q739008",
            "http://d-nb.info/gnd/4160913-1"
        ]
    },
    "kalpis": {
        "id": "kalpis",
        "en": "Kalpis",
        "de": "Kalpis",
        "matches": ["http://vocab.getty.edu/aat/300198839"],
        "broader": "hydria"
    },
    "lekythos": {
        "id": "lekythos",
        "en": "Lekythos",
        "de": "Lekythos",
        "matches": [
            "http://vocab.getty.edu/aat/300198845",
            "http://www.wikidata.org/entity/Q599292",
            "http://d-nb.info/gnd/4167261-0"
        ]
    },
    "bauchlekythos": {
        "id": "squat_lekythos",
        "en": "Squat Lekythos",
        "de": "Bauchlekythos",
        "matches": ["http://vocab.getty.edu/aat/300198846"],
        "broader": "lekythos"
    },
    "glockenkrater": {
        "id": "bell_krater",
        "en": "Bell Krater",
        "de": "Glockenkrater",
        "matches": [
            "http://vocab.getty.edu/aat/300198841",
            "http://www.wikidata.org/entity/Q792247"
        ],
        "broader": "krater"
    },
    "kelchkrater": {
        "id": "calyx_krater",
        "en": "Calyx Krater",
        "de": "Kelchkrater",
        "matches": [
            "http://vocab.getty.edu/aat/300198840",
            "http://www.wikidata.org/entity/Q792246"
        ],
        "broader": "krater"
    },
    "volutenkrater": {
        "id": "volute_krater",
        "en": "Volute Krater",
        "de": "Volutenkrater",
        "matches": [
            "http://vocab.getty.edu/aat/300198843",
            "http://www.wikidata.org/entity/Q792248"
        ],
        "broader": "krater"
    },
    "kolonettenkrater": {
        "id": "column_krater",
        "en": "Column Krater",
        "de": "Kolonettenkrater",
        "matches": [
            "http://vocab.getty.edu/aat/300198844",
            "http://www.wikidata.org/entity/Q792249"
        ],
        "broader": "krater"
    },
    "krater": {
        "id": "krater",
        "en": "Krater",
        "de": "Krater",
        "matches": [
            "http://vocab.getty.edu/aat/300198838",
            "http://www.wikidata.org/entity/Q484435",
            "http://d-nb.info/gnd/4165488-2"
        ]
    },
    "pelike": {
        "id": "pelike",
        "en": "Pelike",
        "de": "Pelike",
        "matches": [
            "http://vocab.getty.edu/aat/300198860",
            "http://www.wikidata.org/entity/Q574041"
        ],
        "broader": "amphora"
    },
    "psykter": {
        "id": "psykter",
        "en": "Psykter",
        "de": "Psykter",
        "matches": [
            "http://vocab.getty.edu/aat/300198863",
            "http://www.wikidata.org/entity/Q1071271"
        ]
    },
    "stamnos": {
        "id": "stamnos",
        "en": "Stamnos",
        "de": "Stamnos",
        "matches": [
            "http://vocab.getty.edu/aat/300198871",
            "http://www.wikidata.org/entity/Q1164998"
        ]
    },
    "skyphos": {
        "id": "skyphos",
        "en": "Skyphos",
        "de": "Skyphos",
        "matches": [
            "http://vocab.getty.edu/aat/300198868",
            "http://www.wikidata.org/entity/Q941427"
        ]
    },
    "kantharos": {
        "id": "kantharos",
        "en": "Kantharos",
        "de": "Kantharos",
        "matches": [
            "http://vocab.getty.edu/aat/300198836",
            "http://www.wikidata.org/entity/Q753765"
        ]
    },
    "oinochoe": {
        "id": "oinochoe",
        "en": "Oinochoe",
        "de": "Oinochoe",
        "matches": [
            "http://vocab.getty.edu/aat/300198854",
            "http://www.wikidata.org/entity/Q645858"
        ]
    },
    "chous": {
        "id": "chous",
        "en": "Chous",
        "de": "Chous",
        "matches": ["http://vocab.getty.edu/aat/300198855"],
        "broader": "oinochoe"
    },
    "olpe": {
        "id": "olpe",
        "en": "Olpe",
        "de": "Olpe",
        "matches": ["http://vocab.getty.edu/aat/300198856"],
        "broader": "oinochoe"
    },
    "pyxis": {
        "id": "pyxis",
        "en": "Pyxis",
        "de": "Pyxis",
        "matches": [
            "http://vocab.getty.edu/aat/300198864",
            "http://www.wikidata.org/entity/Q684803"
        ]
    },
    "aryballos": {
        "id": "aryballos",
        "en": "Aryballos",
        "de": "Aryballos",
        "matches": [
            "http://vocab.getty.edu/aat/300198833",
            "http://www.wikidata.org/entity/Q717871"
        ]
    },
    "alabastron": {
        "id": "alabastron",
        "en": "Alabastron",
        "de": "Alabastron",
        "matches": [
            "http://vocab.getty.edu/aat/300198831",
            "http://www.wikidata.org/entity/Q652410"
        ]
    },
    "dinos": {
        "id": "dinos",
        "en": "Dinos",
        "de": "Dinos",
        "matches": [
            "http://vocab.getty.edu/aat/300198835",
            "http://www.wikidata.org/entity/Q1226343"
        ]
    },
    "lebes": {
        "id": "lebes",
        "en": "Lebes",
        "de": "Lebes",
        "matches": ["http://vocab.getty.edu/aat/300198835"]
    },
    "rhyton": {
        "id": "rhyton",
        "en": "Rhyton",
        "de": "Rhyton",
        "matches": [
            "http://vocab.getty.edu/aat/300198866",
            "http://www.wikidata.org/entity/Q476883"
        ]
    },
    "teller": {
        "id": "plate",
        "en": "Plate",
        "de": "Teller",
        "matches": [
            "http://vocab.getty.edu/aat/300042991",
            "http://www.wikidata.org/entity/Q57216"
        ]
    },
    "fischplatte": {
        "id": "fish_plate",
        "en": "Fish Plate",
        "de": "Fischplatte",
        "matches": ["http://vocab.getty.edu/aat/300198847"],
        "broader": "plate"
    },
    "kyathos": {
        "id": "kyathos",
        "en": "Kyathos",
        "de": "Kyathos",
        "matches": [
            "http://vocab.getty.edu/aat/300198851",
            "http://www.wikidata.org/entity/Q1255866"
        ]
    },
    "phiale": {
        "id": "phiale",
        "en": "Phiale",
        "de": "Phiale",
        "matches": [
            "http://vocab.getty.edu/aat/300198861",
            "http://www.wikidata.org/entity/Q1194200"
        ]
    },
    "loutrophoros": {
        "id": "loutrophoros",
        "en": "Loutrophoros",
        "de": "Loutrophoros",
        "matches": [
            "http://vocab.getty.edu/aat/300198852",
            "http://www.wikidata.org/entity/Q1429815"
        ]
    },
    "lekanis": {
        "id": "lekanis",
        "en": "Lekanis",
        "de": "Lekanis",
        "matches": [
            "http://vocab.getty.edu/aat/300198848",
            "http://www.wikidata.org/entity/Q1816550"
        ]
    },
    "epinetron": {
        "id": "epinetron",
        "en": "Epinetron",
        "de": "Epinetron",
        "matches": [
            "http://vocab.getty.edu/aat/300263229",
            "http://www.wikidata.org/entity/Q1347076"
        ]
    },
    "askos": {
        "id": "askos",
        "en": "Askos",
        "de": "Askos",
        "matches": [
            "http://vocab.getty.edu/aat/300198834",
            "http://www.wikidata.org/entity/Q732446"
        ]
    },
    "kothon": {
        "id": "exaleiptron",
        "en": "Exaleiptron",
        "de": "Kothon / Exaleiptron",
        "matches": ["http://vocab.getty.edu/aat/300198849"]
    },
    "exaleiptron": {
        "id": "exaleiptron",
        "en": "Exaleiptron",
        "de": "Exaleiptron",
        "matches": ["http://vocab.getty.edu/aat/300198849"]
    }
}


# Curated offline mapping: Techniques
KERAMEIKOS_TECHNIQUES: Dict[str, Dict[str, Any]] = {
    "rotfigurig": {
        "id": "red_figure",
        "en": "Red-figure",
        "de": "Rotfigurig",
        "matches": [
            "http://vocab.getty.edu/aat/300020119",
            "http://www.wikidata.org/entity/Q645479"
        ]
    },
    "schwarzfigurig": {
        "id": "black_figure",
        "en": "Black-figure",
        "de": "Schwarzfigurig",
        "matches": [
            "http://vocab.getty.edu/aat/300020118",
            "http://www.wikidata.org/entity/Q500350"
        ]
    },
    "weißgrundig": {
        "id": "white_ground",
        "en": "White Ground",
        "de": "Weißgrundig",
        "matches": [
            "http://vocab.getty.edu/aat/300020120",
            "http://www.wikidata.org/entity/Q1317079"
        ]
    },
    "schwarzfigurig weißgrundig": {
        "id": "black_figure_white_ground",
        "en": "Black-figure White Ground",
        "de": "Schwarzfigurig Weißgrundig",
        "matches": []
    },
    "six-technik": {
        "id": "six",
        "en": "Six's Technique",
        "de": "Six-Technik",
        "matches": [
            "http://vocab.getty.edu/aat/300263238",
            "http://www.wikidata.org/entity/Q2290538"
        ]
    },
    "korallenrot": {
        "id": "coral_red",
        "en": "Coral Red",
        "de": "Korallenrot",
        "matches": ["http://vocab.getty.edu/aat/300263239"]
    },
    "schwarzfirnis": {
        "id": "black_glaze",
        "en": "Black Glaze",
        "de": "Schwarzfirnis",
        "matches": ["http://vocab.getty.edu/aat/300263240"]
    }
}


# Curated offline mapping: Production Places
KERAMEIKOS_PLACES: Dict[str, Dict[str, Any]] = {
    "attisch": {
        "id": "athens",
        "en": "Athens",
        "de": "Athen / Attika",
        "matches": [
            "http://vocab.getty.edu/tgn/7001393",
            "http://www.wikidata.org/entity/Q1524"
        ]
    },
    "athen": {
        "id": "athens",
        "en": "Athens",
        "de": "Athen",
        "matches": ["http://vocab.getty.edu/tgn/7001393"]
    },
    "attika": {
        "id": "attica",
        "en": "Attica",
        "de": "Attika",
        "matches": [
            "http://vocab.getty.edu/tgn/7002681",
            "http://www.wikidata.org/entity/Q170072"
        ]
    },
    "korinthisch": {
        "id": "corinth",
        "en": "Corinth",
        "de": "Korinth",
        "matches": [
            "http://vocab.getty.edu/tgn/7010731",
            "http://www.wikidata.org/entity/Q131367"
        ]
    },
    "korinth": {
        "id": "corinth",
        "en": "Corinth",
        "de": "Korinth",
        "matches": ["http://vocab.getty.edu/tgn/7010731"]
    },
    "apulisch": {
        "id": "apulia",
        "en": "Apulia",
        "de": "Apulien",
        "matches": [
            "http://vocab.getty.edu/tgn/7003004",
            "http://www.wikidata.org/entity/Q1447"
        ]
    },
    "unteritalisch": {
        "id": "italy",
        "en": "Italy",
        "de": "Unteritalien / Italien",
        "matches": [
            "http://vocab.getty.edu/tgn/1000080",
            "http://www.wikidata.org/entity/Q38"
        ]
    },
    "italien": {
        "id": "italy",
        "en": "Italy",
        "de": "Italien",
        "matches": ["http://vocab.getty.edu/tgn/1000080"]
    },
    "chalkidisch": {
        "id": "west_greece",
        "en": "West Greece",
        "de": "Westgriechenland / Chalkidike",
        "matches": []
    },
    "kreta": {
        "id": "crete",
        "en": "Crete",
        "de": "Kreta",
        "matches": [
            "http://vocab.getty.edu/tgn/7012056",
            "http://www.wikidata.org/entity/Q1261"
        ]
    }
}


# Curated offline mapping: Prominent Greek Vase Painters and Workshops
KERAMEIKOS_ARTISTS: Dict[str, Dict[str, Any]] = {
    "acheloos": {"id": "acheloos_painter", "en": "Acheloos Painter", "type": "Person"},
    "achilles": {"id": "achilles_painter", "en": "Achilles Painter", "type": "Person"},
    "amasis p": {"id": "amasis_painter", "en": "Amasis Painter", "type": "Person"},
    "amasis": {"id": "amasis", "en": "Amasis", "type": "Person"},
    "andokides p": {"id": "andokides_painter", "en": "Andokides Painter", "type": "Person"},
    "andokides": {"id": "andokides", "en": "Andokides", "type": "Person"},
    "antimenes": {"id": "antimenes_painter", "en": "Antimenes Painter", "type": "Person"},
    "athena p": {"id": "athena_painter", "en": "Athena Painter", "type": "Person"},
    "berlin p": {"id": "berlin_painter", "en": "Berlin Painter", "type": "Person"},
    "bowdoin": {"id": "bowdoin-eye_painter", "en": "Bowdoin-eye Painter", "type": "Person"},
    "brygos p": {"id": "brygos_painter", "en": "Brygos Painter", "type": "Person"},
    "brygos": {"id": "brygos", "en": "Brygos", "type": "Person"},
    "douris": {"id": "douris", "en": "Douris", "type": "Person"},
    "edinburgh": {"id": "edinburgh_painter", "en": "Edinburgh Painter", "type": "Person"},
    "euphronios": {"id": "euphronios", "en": "Euphronios", "type": "Person"},
    "euxitheos": {"id": "euxitheos", "en": "Euxitheos", "type": "Person"},
    "euthymides": {"id": "euthymides", "en": "Euthymides", "type": "Person"},
    "exekias": {"id": "exekias", "en": "Exekias", "type": "Person"},
    "gela": {"id": "gela_painter", "en": "Gela Painter", "type": "Person"},
    "harrow": {"id": "harrow_painter", "en": "Harrow Painter", "type": "Person"},
    "kleophrades": {"id": "kleophrades_painter", "en": "Kleophrades Painter", "type": "Person"},
    "lydos": {"id": "lydos", "en": "Lydos", "type": "Person"},
    "makron": {"id": "makron", "en": "Makron", "type": "Person"},
    "meidias": {"id": "meidias_painter", "en": "Meidias Painter", "type": "Person"},
    "myson": {"id": "myson", "en": "Myson", "type": "Person"},
    "nikosthenes": {"id": "nikosthenes", "en": "Nikosthenes", "type": "Person"},
    "pan p": {"id": "pan_painter", "en": "Pan Painter", "type": "Person"},
    "phiale": {"id": "phiale_painter", "en": "Phiale Painter", "type": "Person"},
    "phrynos": {"id": "phrynos_painter", "en": "Phrynos Painter", "type": "Person"},
    "sophilos": {"id": "sophilos", "en": "Sophilos", "type": "Person"},
    "group e": {"id": "group_e", "en": "Group E", "type": "Group"},
    "leagros group": {"id": "leagros_group", "en": "Leagros Group", "type": "Group"},
    "haimon group": {"id": "haimon_group", "en": "Haimon Group", "type": "Group"}
}


class KerameikosClient:
    """
    Client for resolving Greek vase attributes to Kerameikos.org Linked Open Data concepts.
    Provides offline-first curated lookups with an optional SPARQL fallback and local JSON caching.
    """

    def __init__(self, cache_file: str = "data/kerameikos_cache.json", online_lookup: bool = False):
        self.cache_file = cache_file
        self.online_lookup = online_lookup
        self.dynamic_cache: Dict[str, Any] = {}
        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.dynamic_cache = json.load(f)
            except Exception:
                self.dynamic_cache = {}

    def _save_cache(self):
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.dynamic_cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def resolve_shape(self, shape_label: str) -> Optional[KerameikosConcept]:
        """
        Resolves a shape term (German or English) to a Kerameikos Shape concept.
        Matches compound and specific forms first.
        """
        if not shape_label:
            return None
        norm = shape_label.lower().strip()

        # Check offline shapes
        # Longest match first to prioritize 'halsamphora' over 'amphora'
        for term, data in sorted(KERAMEIKOS_SHAPES.items(), key=lambda x: len(x[0]), reverse=True):
            if term in norm:
                uri = f"{KERAMEIKOS_ID_BASE}{data['id']}"
                return KerameikosConcept(
                    uri=uri,
                    pref_label_en=data["en"],
                    pref_label_de=data["de"],
                    concept_type="Shape",
                    exact_matches=data.get("matches", []),
                    broader=f"{KERAMEIKOS_ID_BASE}{data['broader']}" if data.get("broader") else None
                )

        # Dynamic cache or online SPARQL
        if norm in self.dynamic_cache.get("shape", {}):
            d = self.dynamic_cache["shape"][norm]
            return KerameikosConcept(**d)

        if self.online_lookup:
            res = self._sparql_lookup_concept(norm, "kon:Shape")
            if res:
                self.dynamic_cache.setdefault("shape", {})[norm] = res.to_dict()
                self._save_cache()
                return res

        return None

    def resolve_technique(self, technique_or_ware: str) -> Optional[KerameikosConcept]:
        """
        Resolves a decoration technique or ware style to a Kerameikos Technique concept.
        """
        if not technique_or_ware:
            return None
        norm = technique_or_ware.lower().strip()

        for term, data in sorted(KERAMEIKOS_TECHNIQUES.items(), key=lambda x: len(x[0]), reverse=True):
            if term in norm:
                uri = f"{KERAMEIKOS_ID_BASE}{data['id']}"
                return KerameikosConcept(
                    uri=uri,
                    pref_label_en=data["en"],
                    pref_label_de=data["de"],
                    concept_type="Technique",
                    exact_matches=data.get("matches", [])
                )

        if norm in self.dynamic_cache.get("technique", {}):
            return KerameikosConcept(**self.dynamic_cache["technique"][norm])

        if self.online_lookup:
            res = self._sparql_lookup_concept(norm, "kon:Technique")
            if res:
                self.dynamic_cache.setdefault("technique", {})[norm] = res.to_dict()
                self._save_cache()
                return res

        return None

    def resolve_production_place(self, region_label: str) -> Optional[KerameikosConcept]:
        """
        Resolves a production region/fabric to a Kerameikos Place concept.
        """
        if not region_label:
            return None
        norm = region_label.lower().strip()

        for term, data in sorted(KERAMEIKOS_PLACES.items(), key=lambda x: len(x[0]), reverse=True):
            if term in norm:
                uri = f"{KERAMEIKOS_ID_BASE}{data['id']}"
                return KerameikosConcept(
                    uri=uri,
                    pref_label_en=data["en"],
                    pref_label_de=data["de"],
                    concept_type="ProductionPlace",
                    exact_matches=data.get("matches", [])
                )

        if norm in self.dynamic_cache.get("place", {}):
            return KerameikosConcept(**self.dynamic_cache["place"][norm])

        if self.online_lookup:
            res = self._sparql_lookup_concept(norm, "crm:E53_Place")
            if res:
                self.dynamic_cache.setdefault("place", {})[norm] = res.to_dict()
                self._save_cache()
                return res

        return None

    def resolve_artist(self, artist_raw: str) -> Optional[KerameikosConcept]:
        """
        Resolves an artist, painter, or workshop attribution string to a Kerameikos Concept.
        Handles complex BAPD strings (e.g. 'Near LYDOS by UNKNOWN LYDOS by UNKNOWN').
        """
        if not artist_raw:
            return None
        norm = artist_raw.lower().strip()

        for key, data in sorted(KERAMEIKOS_ARTISTS.items(), key=lambda x: len(x[0]), reverse=True):
            if key in norm:
                uri = f"{KERAMEIKOS_ID_BASE}{data['id']}"
                return KerameikosConcept(
                    uri=uri,
                    pref_label_en=data["en"],
                    pref_label_de=data["en"],
                    concept_type=data.get("type", "Person")
                )

        if norm in self.dynamic_cache.get("artist", {}):
            return KerameikosConcept(**self.dynamic_cache["artist"][norm])

        return None

    def _sparql_lookup_concept(self, search_term: str, rdf_type: str) -> Optional[KerameikosConcept]:
        """Queries the live Kerameikos SPARQL endpoint."""
        query = f"""
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX kon: <https://kerameikos.org/ontology#>
        PREFIX crm: <http://www.cidoc-crm.org/cidoc-crm/>
        SELECT ?id ?prefLabel ?def ?match WHERE {{
          ?id a {rdf_type} .
          ?id skos:prefLabel ?prefLabel .
          FILTER(CONTAINS(LCASE(STR(?prefLabel)), "{search_term}"))
          OPTIONAL {{ ?id skos:definition ?def . FILTER(LANG(?def) = 'en' || LANG(?def) = 'de' || LANG(?def) = '') }}
          OPTIONAL {{ ?id skos:exactMatch ?match . }}
        }}
        LIMIT 5
        """
        try:
            url = KERAMEIKOS_SPARQL + "?query=" + urllib.parse.quote(query) + "&output=json"
            req = urllib.request.Request(url, headers={"User-Agent": "LODVases-Research/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                bindings = data.get("results", {}).get("bindings", [])
                if not bindings:
                    return None
                
                uri = bindings[0]["id"]["value"]
                lbl = bindings[0].get("prefLabel", {}).get("value", search_term)
                defn = bindings[0].get("def", {}).get("value")
                matches = list(set([b["match"]["value"] for b in bindings if "match" in b]))

                ctype = "Shape" if "Shape" in rdf_type else ("Technique" if "Technique" in rdf_type else "Concept")
                return KerameikosConcept(
                    uri=uri,
                    pref_label_en=lbl,
                    pref_label_de=lbl,
                    concept_type=ctype,
                    definition=defn,
                    exact_matches=matches
                )
        except Exception:
            return None


# Global singleton
kerameikos_client = KerameikosClient()
