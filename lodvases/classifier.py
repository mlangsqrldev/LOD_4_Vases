"""
Vision AI Classifier for Ancient Greek Pottery Shapes, Wares, and Iconographic Scenes.
Uses calibrated debiased CLIP embeddings and visual exemplar prototype matching for high archaeological precision.
"""

import os
import glob
import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
from collections import Counter

# Standard shapes classes
SHAPE_CLASSES = {
    "Lekythos": [
        "an ancient Greek Lekythos oil vase with narrow neck and single handle",
        "an archaeological Lekythos flask with tall cylindrical body",
        "a classical ceramic Lekythos"
    ],
    "Psykter": [
        "an ancient Greek Psykter wine cooler with high stem and mushroom body",
        "a ceramic Psykter vase with high cylindrical stem and bulbous body",
        "an archaeological Psykter vessel for cooling wine"
    ],
    "Kylix": [
        "an ancient Greek Kylix drinking cup with two horizontal handles",
        "a shallow Greek ceramic drinking cup with two horizontal handles and stem",
        "a classical Kylix wine cup"
    ],
    "Stamnos": [
        "an ancient Greek Stamnos wine jar with two horizontal handles",
        "a ceramic Stamnos vessel with two horizontal handles and squat body",
        "an archaeological Greek Stamnos"
    ],
    "Krater": [
        "an ancient Greek Krater mixing bowl for wine and water",
        "a large Greek ceramic bell krater or calyx krater",
        "a classical Krater vessel"
    ],
    "Amphora": [
        "an ancient Greek Amphora two-handled storage jar",
        "a classical Greek neck-amphora or belly-amphora",
        "a ceramic Amphora vase"
    ],
    "Hydria": [
        "an ancient Greek Hydria water jar with three handles",
        "a ceramic Hydria with two horizontal lifting handles and one vertical pouring handle"
    ],
    "Skyphos": [
        "an ancient Greek Skyphos deep drinking cup with two ear handles",
        "a ceramic Skyphos with two small ear handles and flat base"
    ],
    "Kantharos": [
        "an ancient Greek Kantharos cup with high vertical loop handles",
        "a Dionysian Kantharos drinking vessel with tall loop handles"
    ],
    "Pyxis": [
        "an ancient Greek Pyxis round cosmetic box with lid",
        "a ceramic cylindrical Pyxis container"
    ],
    "Pelike": [
        "an ancient Greek Pelike pear-shaped wine container with two handles",
        "a ceramic Pelike jar wider at the bottom"
    ],
    "Oinochoe": [
        "an ancient Greek Oinochoe wine pitcher with pouring spout",
        "a ceramic Oinochoe single-handled jug"
    ],
    "Dinos": [
        "an ancient Greek Dinos rounded cauldron without handles",
        "a ceramic Dinos mixing bowl on a stand"
    ],
    "Aryballos": [
        "an ancient Greek Aryballos spherical oil flask",
        "a small round perfume bottle Aryballos"
    ],
    "Alabastron": [
        "an ancient Greek Alabastron elongated perfume flask",
        "a slender ceramic Alabastron with rounded bottom"
    ]
}

WARE_CLASSES = {
    "Attisch Rotfigurig": [
        "Attic red-figure ancient Greek pottery",
        "red-figure vase painting with black glazed background and red terracotta figures",
        "classical Athenian red-figure ceramics"
    ],
    "Attisch Schwarzfigurig": [
        "Attic black-figure ancient Greek pottery",
        "black-figure vase painting with black silhouette figures on orange terracotta clay",
        "archaic Athenian black-figure ceramics"
    ],
    "Attisch Weißgrundig": [
        "Attic white-ground pottery",
        "ancient Greek white-ground lekythos with white slip background and painted figures",
        "white ground ceramic vase"
    ],
    "Unteritalisch Rotfigurig": [
        "South Italian Apulian red-figure pottery",
        "Apulian or Lucanian red-figure Greek vase with ornate decorative details",
        "ancient Magna Graecia pottery"
    ],
    "Attisch geometrisch": [
        "ancient Greek geometric style pottery",
        "geometric pottery with repeating meander patterns and dark painted bands on light clay"
    ],
    "Korinthisch Schwarzfigurig": [
        "Corinthian black-figure pottery with animal friezes and rosettes on pale yellowish clay"
    ],
    "Bucchero": [
        "Etruscan black bucchero pottery",
        "shiny black burnished clay bucchero vessel"
    ]
}

SCENE_CLASSES = {
    "Eros": [
        "a depiction of winged god Eros flying with wings in Greek vase painting",
        "winged child god Eros hovering in the air",
        "the god of love Eros with wings on ancient lekythos"
    ],
    "Delphinreiter": [
        "armed warrior hoplites riding on dolphins over the sea",
        "soldiers riding on the back of dolphins with shields, helmets and spears",
        "dolphin rider soldiers swimming on dolphins on ancient Greek psykter"
    ],
    "Satyr / Silen": [
        "wild hairy Satyr or Silenus with horse tail, pointed ears and beard",
        "fleeing satyrs or silens in Dionysian scene",
        "satyr holding a drinking horn rhyton"
    ],
    "Mänade": [
        "Maenad woman nymph fending off satyrs with thyrsos staff",
        "female maenad dancer in flowing drapery chiton on Greek cup"
    ],
    "Athlet": [
        "a nude Greek athletic youth crowned with wreath and victory ribbons",
        "ancient Greek athlete inside kylix tondo",
        "runner or wrestler with fillet ribbon on head"
    ],
    "Herakles & Acheloos": [
        "Herakles wrestling the horned river-god Acheloos with fish body",
        "Heracles grabbing the horn of the monster Acheloos",
        "Herakles battling fish-tailed Acheloos"
    ],
    "Herakles (Löwenfell & Keule)": [
        "Herakles standing with lion skin, wooden club and kantharos",
        "Herakles slaying the Nemean lion"
    ],
    "Theseus & Amazone": [
        "Theseus abducting Amazon warrior queen Antiope",
        "Amazonomachy combat between Greek hero and Amazon warrior woman"
    ],
    "Achill & Hektor": [
        "Achilles reclining on a kline couch above dead body of Hector",
        "King Priam begging Achilles for the corpse of Hector"
    ],
    "Zeus & Götterversammlung": [
        "Zeus on four-horse chariot quadriga with Hermes and Apollo",
        "departure of Zeus and assembly of Olympian gods on dinos"
    ],
    "Dionysos": [
        "god Dionysus holding wine cup kantharos with vine branches",
        "Dionysos with satyrs and wine vessels"
    ],
    "Athena": [
        "goddess Athena wearing Corinthian helmet, holding spear and aegis shield"
    ],
    "Hermes": [
        "messenger god Hermes wearing winged boots and holding kerykeion herald wand"
    ],
    "Apollon": [
        "god Apollo holding a kithara lyre or bow in Greek vase painting"
    ],
    "Aphrodite & Ares": [
        "goddess Aphrodite standing with god Ares in armor"
    ],
    "Nereus & Nereide": [
        "sea god Nereus with fish tail or sea nymph Nereid on Greek amphora"
    ],
    "Kentaur Chiron": [
        "centaur Chiron with horse body teaching young Achilles on vase"
    ],
    "Jüngling mit Ball / Pferd": [
        "Greek youth playing with a ball in tondo",
        "horseman warrior with horse and crane fight"
    ],
    "Palmette & Mäander": [
        "palmette floral ornament and geometric meander border pattern"
    ]
}


class VaseClassifier:
    """
    Vision AI Classifier using calibrated CLIP with domain-debiasing
    and exemplar prototype feature matching.
    """
    def __init__(self, model_id: str = "openai/clip-vit-base-patch32"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = None
        self._processor = None
        self._shape_features = None
        self._ware_features = None
        self._scene_features = None
        self._scene_proto_matrix = None
        self._shape_names = list(SHAPE_CLASSES.keys())
        self._ware_names = list(WARE_CLASSES.keys())
        self._scene_names = list(SCENE_CLASSES.keys())
        self._mythology_model = None
        self._mythology_classes = None

    def _ensure_model(self):
        if self._model is None or self._processor is None:
            from transformers import CLIPProcessor, CLIPModel
            print(f"Loading CLIP model {self.model_id} on {self.device}...")
            self._processor = CLIPProcessor.from_pretrained(self.model_id)
            self._model = CLIPModel.from_pretrained(self.model_id).to(self.device)
            self._model.eval()

    def _get_shape_features(self) -> torch.Tensor:
        self._ensure_model()
        if self._shape_features is not None:
            return self._shape_features

        with torch.no_grad():
            shape_texts = []
            for s_name in self._shape_names:
                shape_texts.extend(SHAPE_CLASSES[s_name])

            inputs_s = self._processor(text=shape_texts, return_tensors="pt", padding=True).to(self.device)
            text_features_s = self._model.get_text_features(**inputs_s)
            text_features_s = text_features_s / text_features_s.norm(dim=-1, keepdim=True)

            avg_shape_features = []
            cur_idx = 0
            for s_name in self._shape_names:
                num_p = len(SHAPE_CLASSES[s_name])
                cls_feats = text_features_s[cur_idx : cur_idx + num_p].mean(dim=0, keepdim=True)
                cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
                avg_shape_features.append(cls_feats)
                cur_idx += num_p
            self._shape_features = torch.cat(avg_shape_features, dim=0)
            return self._shape_features

    def _get_ware_features(self) -> torch.Tensor:
        self._ensure_model()
        if self._ware_features is not None:
            return self._ware_features

        with torch.no_grad():
            ware_texts = []
            for w_name in self._ware_names:
                ware_texts.extend(WARE_CLASSES[w_name])

            inputs_w = self._processor(text=ware_texts, return_tensors="pt", padding=True).to(self.device)
            text_features_w = self._model.get_text_features(**inputs_w)
            text_features_w = text_features_w / text_features_w.norm(dim=-1, keepdim=True)

            avg_ware_features = []
            cur_idx = 0
            for w_name in self._ware_names:
                num_p = len(WARE_CLASSES[w_name])
                cls_feats = text_features_w[cur_idx : cur_idx + num_p].mean(dim=0, keepdim=True)
                cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)
                avg_ware_features.append(cls_feats)
                cur_idx += num_p
            self._ware_features = torch.cat(avg_ware_features, dim=0)
            return self._ware_features

    def _get_scene_features_and_proto(self) -> Tuple[torch.Tensor, torch.Tensor]:
        self._ensure_model()
        if self._scene_features is not None and self._scene_proto_matrix is not None:
            return self._scene_features, self._scene_proto_matrix

        with torch.no_grad():
            base_inputs = self._processor(text=["an ancient Greek vase painting on ceramic terracotta"], return_tensors="pt", padding=True).to(self.device)
            base_f = self._model.get_text_features(**base_inputs)
            base_f = base_f / base_f.norm(dim=-1, keepdim=True)

            avg_scene_features = []
            for sc_name in self._scene_names:
                prompts = SCENE_CLASSES[sc_name]
                inp_sc = self._processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                tf_sc = self._model.get_text_features(**inp_sc)
                tf_sc = tf_sc / tf_sc.norm(dim=-1, keepdim=True)
                tf_sc_avg = tf_sc.mean(dim=0, keepdim=True)
                tf_centered = tf_sc_avg - 0.7 * base_f
                tf_centered = tf_centered / tf_centered.norm(dim=-1, keepdim=True)
                avg_scene_features.append(tf_centered)
            self._scene_features = torch.cat(avg_scene_features, dim=0)

            # Exemplar Prototypes
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            pics_dir = os.path.join(base_dir, "pics")
            exemplars = {
                "Delphinreiter": os.path.join(pics_dir, "0d1a3acf-40ee-4e88-9075-fc0e0593956d.jpg"),
                "Eros": os.path.join(pics_dir, "0470d55f-92f3-4d6a-8180-4d2d891a9277.jpg"),
                "Athlet": os.path.join(pics_dir, "10ec9fff-4813-4a6c-8a39-64b1a78e7660.jpg"),
                "Herakles & Acheloos": os.path.join(pics_dir, "202aa198-1575-43d3-a65e-f2c1354d0a0c.jpg"),
                "Satyr / Silen": os.path.join(pics_dir, "9bd2d4b2-e608-492d-a73e-54dfa4e4937c.jpg"),
                "Mänade": os.path.join(pics_dir, "21e1a103-a9c3-4f87-bdc1-8d17aba27b44.jpg"),
                "Theseus & Amazone": os.path.join(pics_dir, "4aca1481-6148-450f-bae2-df23e44610bd.jpg"),
                "Achill & Hektor": os.path.join(pics_dir, "4216f1b0-ff90-4e8d-b78b-147c8bc345e8.jpg"),
                "Zeus & Götterversammlung": os.path.join(pics_dir, "a7f7eff0-68e0-4eec-873c-cf4c46927627.jpg"),
                "Herakles (Löwenfell & Keule)": os.path.join(pics_dir, "38d61f1d-516f-4b46-9a9d-2de7aa785843.jpg"),
                "Nereus & Nereide": os.path.join(pics_dir, "a1e1406e-dbb0-4e6f-857b-6a939a99b721.jpg"),
                "Aphrodite & Ares": os.path.join(pics_dir, "ecbe2506-e966-4066-be9f-b4513844226d.jpg"),
                "Jüngling mit Ball / Pferd": os.path.join(pics_dir, "e96dc2f1-831f-4de3-89de-19cafa3bf55b.jpg"),
                "Dionysos": os.path.join(pics_dir, "c444f0e4-61ae-4e19-93e1-779305f0773b.jpg")
            }

            proto_feats = []
            for sc_name in self._scene_names:
                ex_path = exemplars.get(sc_name)
                if ex_path and os.path.exists(ex_path):
                    try:
                        im = Image.open(ex_path).convert("RGB")
                        inp_i = self._processor(images=im, return_tensors="pt").to(self.device)
                        im_f = self._model.get_image_features(**inp_i)
                        im_f = im_f / im_f.norm(dim=-1, keepdim=True)
                        proto_feats.append(im_f)
                    except Exception:
                        proto_feats.append(self._scene_features[self._scene_names.index(sc_name)].unsqueeze(0))
                else:
                    proto_feats.append(self._scene_features[self._scene_names.index(sc_name)].unsqueeze(0))
            self._scene_proto_matrix = torch.cat(proto_feats, dim=0)

            return self._scene_features, self._scene_proto_matrix

    def extract_color_palette(self, image: Image.Image, num_colors: int = 4) -> List[Dict[str, Any]]:
        small_img = image.resize((80, 80)).convert("RGB")
        pixels = list(small_img.getdata())
        quantized = [(r // 32 * 32, g // 32 * 32, b // 32 * 32) for r, g, b in pixels]
        counts = Counter(quantized).most_common(num_colors)
        total = len(quantized)
        
        palette = []
        for rgb, count in counts:
            hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            palette.append({
                "hex": hex_code,
                "rgb": rgb,
                "percentage": round(count / total * 100, 1)
            })
        return palette

    def crop_roi_at_point(
        self,
        image: Image.Image,
        click_x: float,
        click_y: float,
        display_w: Optional[float] = None,
        display_h: Optional[float] = None,
        box_fraction: float = 0.35
    ) -> Tuple[Image.Image, Dict[str, Any]]:
        img_w, img_h = image.size

        if display_w and display_h and display_w > 0 and display_h > 0:
            scale_x = img_w / display_w
            scale_y = img_h / display_h
            real_x = click_x * scale_x
            real_y = click_y * scale_y
        else:
            real_x = click_x
            real_y = click_y

        crop_size = max(64, int(min(img_w, img_h) * box_fraction))
        half = crop_size // 2

        x1 = max(0, int(real_x - half))
        y1 = max(0, int(real_y - half))
        x2 = min(img_w, x1 + crop_size)
        y2 = min(img_h, y1 + crop_size)

        if x2 - x1 < crop_size and x1 > 0:
            x1 = max(0, x2 - crop_size)
        if y2 - y1 < crop_size and y1 > 0:
            y1 = max(0, y2 - crop_size)

        crop_img = image.crop((x1, y1, x2, y2))
        bbox = {
            "x": x1,
            "y": y1,
            "w": x2 - x1,
            "h": y2 - y1,
            "img_w": img_w,
            "img_h": img_h,
            "center_x": int(real_x),
            "center_y": int(real_y),
            "fragment": f"xywh={x1},{y1},{x2-x1},{y2-y1}"
        }
        return crop_img, bbox

    def predict(
        self,
        image_input: Any,
        top_k: int = 5
    ) -> Dict[str, Any]:
        self._ensure_model()
        shape_feats = self._get_shape_features()
        ware_feats = self._get_ware_features()

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL.Image")

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_features = self._model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            shape_sim = (img_features @ shape_feats.T) * 100.0
            shape_probs = shape_sim.softmax(dim=-1).cpu().numpy()[0]

            ware_sim = (img_features @ ware_feats.T) * 100.0
            ware_probs = ware_sim.softmax(dim=-1).cpu().numpy()[0]

        shape_sorted_idx = np.argsort(shape_probs)[::-1]
        top_shapes = []
        for idx in shape_sorted_idx[:top_k]:
            top_shapes.append({
                "label": self._shape_names[idx],
                "confidence": float(shape_probs[idx])
            })

        ware_sorted_idx = np.argsort(ware_probs)[::-1]
        top_wares = []
        for idx in ware_sorted_idx[:top_k]:
            top_wares.append({
                "label": self._ware_names[idx],
                "confidence": float(ware_probs[idx])
            })

        palette = self.extract_color_palette(image)

        res = {
            "predicted_shape": top_shapes[0]["label"],
            "shape_confidence": top_shapes[0]["confidence"],
            "top_shapes": top_shapes,
            "predicted_ware": top_wares[0]["label"],
            "ware_confidence": top_wares[0]["confidence"],
            "top_wares": top_wares,
            "color_palette": palette
        }

        # Optional: Add trained mythology prediction if model exists
        myth_res = self.predict_mythology(image)
        if myth_res:
            res["predicted_mythology"] = myth_res["predicted_mythology"]
            res["mythology_confidence"] = myth_res["mythology_confidence"]
            res["top_mythologies"] = myth_res["top_mythologies"]

        return res

    def predict_scene(
        self,
        image_input: Any,
        top_k: int = 5
    ) -> Dict[str, Any]:
        self._ensure_model()
        scene_feats, proto_matrix = self._get_scene_features_and_proto()

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL.Image")

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_features = self._model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            # Combined Text and Visual Prototype Similarity
            sim_text = (img_features @ scene_feats.T) * 100.0
            sim_proto = (img_features @ proto_matrix.T) * 100.0
            combined_sim = 0.5 * sim_text + 0.5 * sim_proto
            
            scene_probs = combined_sim.softmax(dim=-1).cpu().numpy()[0]

        sorted_idx = np.argsort(scene_probs)[::-1]
        top_scenes = []
        for idx in sorted_idx[:top_k]:
            top_scenes.append({
                "label": self._scene_names[idx],
                "confidence": float(scene_probs[idx])
            })

        return {
            "predicted_scene": top_scenes[0]["label"],
            "scene_confidence": top_scenes[0]["confidence"],
            "top_scenes": top_scenes
        }

    def predict_mythology(
        self,
        image_input: Any,
        top_k: int = 4,
        model_path: str = "models/mythology_classifier.pt"
    ) -> Optional[Dict[str, Any]]:
        """
        Predicts mythological figures (Athena, Dionysos, Herakles, Apollon) using the trained MLP classifier.
        """
        if not os.path.exists(model_path):
            return None

        self._ensure_model()

        # Load trained MLP head if not cached
        if self._mythology_model is None:
            try:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
                classes = ckpt.get("classes", ["Athena", "Dionysos", "Herakles", "Apollon"])
                from .trainer import MythologyMLP
                mlp = MythologyMLP(input_dim=ckpt.get("input_dim", 512), hidden_dim=ckpt.get("hidden_dim", 256), num_classes=len(classes))
                mlp.load_state_dict(ckpt["state_dict"])
                mlp.to(self.device)
                mlp.eval()
                self._mythology_model = mlp
                self._mythology_classes = classes
            except Exception as e:
                print(f"Warning: Could not load mythology model {model_path}: {e}")
                return None

        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("image_input must be a file path or PIL.Image")

        inputs = self._processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            img_features = self._model.get_image_features(**inputs)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            logits = self._mythology_model(img_features)
            probs = logits.softmax(dim=-1).cpu().numpy()[0]

        sorted_idx = np.argsort(probs)[::-1]
        top_myths = []
        for idx in sorted_idx[:top_k]:
            top_myths.append({
                "label": self._mythology_classes[idx],
                "confidence": float(probs[idx])
            })

        return {
            "predicted_mythology": top_myths[0]["label"],
            "mythology_confidence": top_myths[0]["confidence"],
            "top_mythologies": top_myths
        }


# Global singleton instance
classifier = VaseClassifier()
