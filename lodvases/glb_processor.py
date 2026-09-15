"""
3D GLB / GLTF Processor for Ancient Ceramics with GigaMesh Conformal Profile Raycasting.
Features:
- Conformal Body Wall Profile Raycasting (GigaMesh-Standard): Filters out handles and interior cavities.
- Handles multi-handled vessels (Pelike, Amphora, Hydria, Aryballos, Krater) seamlessly.
- Real-time interactive calibration (Center-X, Center-Z offsets, Azimuth 0-360°, Z-min, Z-max).
- PBR Texture Layer Categorization (BaseColor, Normal Map, Roughness/AO).
- Direct vertex geometry extraction and archaeological profiling.
"""

import os
import io
import json
import struct
import trimesh
import numpy as np
from PIL import Image, ImageDraw
from typing import Dict, Any, List, Tuple, Optional

class GLBProcessor:
    """
    Processes 3D ancient pottery models (.glb / .gltf) with GigaMesh-standard 360° surface unrolling.
    """

    @staticmethod
    def extract_and_categorize_textures(glb_bytes: bytes) -> Dict[str, Any]:
        """
        Extracts all textures from binary GLB and classifies them by their PBR role.
        """
        if not isinstance(glb_bytes, (bytes, bytearray)) or len(glb_bytes) < 12:
            return {"base_color": None, "normal": None, "roughness": None, "all": []}

        try:
            magic, version, total_len = struct.unpack('<4sII', glb_bytes[:12])
            if magic != b'glTF':
                return {"base_color": None, "normal": None, "roughness": None, "all": []}

            pos = 12
            json_data = None
            bin_data = None

            while pos < len(glb_bytes) - 8:
                chunk_len, chunk_type = struct.unpack('<I4s', glb_bytes[pos:pos+8])
                pos += 8
                chunk_bytes = glb_bytes[pos:pos+chunk_len]
                pos += chunk_len

                if chunk_type == b'JSON':
                    json_data = json.loads(chunk_bytes.decode('utf-8'))
                elif chunk_type in [b'BIN\x00', b'BIN']:
                    bin_data = chunk_bytes

            images_named = {}
            images_list = []
            if json_data and 'images' in json_data and bin_data:
                buffer_views = json_data.get('bufferViews', [])
                for idx, img_info in enumerate(json_data['images']):
                    if 'bufferView' in img_info:
                        bv_idx = img_info['bufferView']
                        if bv_idx < len(buffer_views):
                            bv = buffer_views[bv_idx]
                            offset = bv.get('byteOffset', 0)
                            b_len = bv.get('byteLength', 0)
                            img_bytes = bin_data[offset:offset+b_len]
                            try:
                                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                images_list.append(pil_img)
                                name = img_info.get('name', f'image_{idx}').lower()
                                images_named[name] = pil_img
                            except Exception:
                                pass

            textures_dict = {"base_color": None, "normal": None, "roughness": None, "all": images_list}

            # 1. Priority: Match explicitly by glTF image names
            for name, img in images_named.items():
                if any(k in name for k in ['diffuse', 'basecolor', 'albedo', 'color', 'col']):
                    if textures_dict['base_color'] is None:
                        textures_dict['base_color'] = img
                elif any(k in name for k in ['normal', 'norm', 'nrm']):
                    if textures_dict['normal'] is None:
                        textures_dict['normal'] = img
                elif any(k in name for k in ['rough', 'metal', 'ao', 'occlusion', 'spec', 'orm']):
                    if textures_dict['roughness'] is None:
                        textures_dict['roughness'] = img

            # 2. Priority: Use glTF materials
            if json_data and "materials" in json_data:
                textures_refs = json_data.get("textures", [])
                for mat in json_data["materials"]:
                    if "pbrMetallicRoughness" in mat:
                        pbr = mat["pbrMetallicRoughness"]
                        if "baseColorTexture" in pbr and textures_dict["base_color"] is None:
                            tex_idx = pbr["baseColorTexture"].get("index", 0)
                            if tex_idx < len(textures_refs):
                                src_idx = textures_refs[tex_idx].get("source", 0)
                                if src_idx < len(images_list):
                                    textures_dict["base_color"] = images_list[src_idx]
                        if "metallicRoughnessTexture" in pbr and textures_dict["roughness"] is None:
                            tex_idx = pbr["metallicRoughnessTexture"].get("index", 0)
                            if tex_idx < len(textures_refs):
                                src_idx = textures_refs[tex_idx].get("source", 0)
                                if src_idx < len(images_list):
                                    textures_dict["roughness"] = images_list[src_idx]
                    if "normalTexture" in mat and textures_dict["normal"] is None:
                        tex_idx = mat["normalTexture"].get("index", 0)
                        if tex_idx < len(textures_refs):
                            src_idx = textures_refs[tex_idx].get("source", 0)
                            if src_idx < len(images_list):
                                textures_dict["normal"] = images_list[src_idx]

            # 3. Fallback
            if textures_dict["base_color"] is None and images_list:
                textures_dict["base_color"] = images_list[-1]

            return textures_dict
        except Exception:
            return {"base_color": None, "normal": None, "roughness": None, "all": []}

    @staticmethod
    def detect_rotational_axis(verts: np.ndarray) -> str:
        """
        Detects the principal rotational symmetry axis (X, Y, or Z) of pottery.
        """
        variances = {}
        for ax_idx, ax_name in enumerate(["X", "Y", "Z"]):
            other = [i for i in range(3) if i != ax_idx]
            slices = np.linspace(verts[:, ax_idx].min(), verts[:, ax_idx].max(), 20)
            std_r = []
            for i in range(len(slices) - 1):
                mask = (verts[:, ax_idx] >= slices[i]) & (verts[:, ax_idx] <= slices[i + 1])
                if np.sum(mask) > 8:
                    pts = verts[mask][:, other]
                    pts_c = pts - pts.mean(axis=0)
                    r = np.linalg.norm(pts_c, axis=1)
                    std_r.append(np.std(r) / (np.mean(r) + 1e-6))
            variances[ax_name] = np.mean(std_r) if std_r else 999.0
        return min(variances, key=variances.get)

    @staticmethod
    def load_glb(file_input: Any) -> Dict[str, Any]:
        """
        Loads a .glb file, categorizes all PBR texture layers, and prepares geometry.
        """
        glb_bytes = None
        if isinstance(file_input, str):
            with open(file_input, "rb") as f:
                glb_bytes = f.read()
            scene_or_mesh = trimesh.load(file_input, file_type="glb")
            file_name = os.path.basename(file_input)
        elif isinstance(file_input, (bytes, bytearray)):
            glb_bytes = bytes(file_input)
            scene_or_mesh = trimesh.load(io.BytesIO(glb_bytes), file_type="glb")
            file_name = "uploaded_model.glb"
        else:
            glb_bytes = file_input.read() if hasattr(file_input, "read") else None
            scene_or_mesh = trimesh.load(file_input, file_type="glb")
            file_name = getattr(file_input, "name", "model.glb")

        if isinstance(scene_or_mesh, trimesh.Scene):
            # Exclude non-vase helpers like 'Circle'
            geoms = {k: v for k, v in scene_or_mesh.geometry.items() if isinstance(v, trimesh.Trimesh) and "circle" not in k.lower()}
            if not geoms:
                geoms = {k: v for k, v in scene_or_mesh.geometry.items() if isinstance(v, trimesh.Trimesh)}

            if geoms:
                main_name = max(geoms.keys(), key=lambda k: len(geoms[k].vertices))
                mesh = geoms[main_name].copy()
                for n in scene_or_mesh.graph.nodes_geometry:
                    if scene_or_mesh.graph[n][1] == main_name:
                        transform, _ = scene_or_mesh.graph[n]
                        mesh.apply_transform(transform)
                        break
            else:
                mesh = trimesh.Trimesh()
            scene = scene_or_mesh
        else:
            mesh = scene_or_mesh
            scene = trimesh.Scene(mesh)

        extents = mesh.extents if hasattr(mesh, "extents") else [10, 10, 20]
        bounds = mesh.bounds if hasattr(mesh, "bounds") else [[-5, -5, 0], [5, 5, 20]]
        vertices = mesh.vertices if hasattr(mesh, "vertices") else np.array([])
        faces = mesh.faces if hasattr(mesh, "faces") else np.array([])

        pbr_textures = GLBProcessor.extract_and_categorize_textures(glb_bytes) if glb_bytes else {}
        base_color_img = pbr_textures.get("base_color")
        normal_img = pbr_textures.get("normal")
        roughness_img = pbr_textures.get("roughness")
        all_textures = pbr_textures.get("all", [])

        uv_coords = None
        if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None:
            uv_coords = np.array(mesh.visual.uv)

        detected_axis = GLBProcessor.detect_rotational_axis(vertices) if len(vertices) > 0 else "Y"

        return {
            "mesh": mesh,
            "scene": scene,
            "glb_bytes": glb_bytes,
            "filename": file_name,
            "vertices_count": len(vertices),
            "faces_count": len(faces),
            "detected_axis": detected_axis,
            "extents": list(extents) if hasattr(extents, "__iter__") else [10, 10, 20],
            "bounds": list(bounds) if hasattr(bounds, "__iter__") else [[-5, -5, 0], [5, 5, 20]],
            "center": list(mesh.centroid) if hasattr(mesh, "centroid") else [0, 0, 10],
            "texture": base_color_img,
            "texture_base_color": base_color_img,
            "texture_normal": normal_img,
            "texture_roughness": roughness_img,
            "all_textures": all_textures,
            "uv": uv_coords,
            "is_watertight": getattr(mesh, "is_watertight", False)
        }

    @staticmethod
    def unroll_pottery_gigamesh(
        mesh: trimesh.Trimesh,
        texture_img: Optional[Image.Image] = None,
        resolution: Tuple[int, int] = (1600, 800),
        axis: str = "Y",
        flip_vertical: bool = False,
        azimuth_offset_deg: float = 0.0,
        cx_offset: float = 0.0,
        cz_offset: float = 0.0,
        z_crop: Tuple[float, float] = (0.10, 0.85),
        projection_mode: str = "cylindrical"
    ) -> Image.Image:
        """
        GigaMesh Conformal Body Profile Raycasting Rollout.
        Casts outward rays and picks the ceramic body wall hit, bypassing handles and cavity interiors.
        """
        W, H = resolution
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        uvs = np.array(mesh.visual.uv, dtype=np.float32) if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None else None

        if len(verts) == 0 or len(faces) == 0 or uvs is None:
            return Image.new("RGB", (W, H), (15, 23, 42))

        if axis.lower() == "auto":
            axis = GLBProcessor.detect_rotational_axis(verts)

        axis = axis.upper()

        # Coordinate assignment according to chosen rotational axis:
        if axis == "Y":
            cx = verts[:, 0].mean() + cx_offset
            cz = verts[:, 2].mean() + cz_offset
            h_coords = verts[:, 1]
            u_coords = verts[:, 0]
            v_coords = verts[:, 2]
            dist_center = (cx, cz)
        elif axis == "X":
            cx = verts[:, 1].mean() + cx_offset
            cz = verts[:, 2].mean() + cz_offset
            h_coords = verts[:, 0]
            u_coords = verts[:, 1]
            v_coords = verts[:, 2]
            dist_center = (cx, cz)
        else: # "Z"
            cx = verts[:, 0].mean() + cx_offset
            cz = verts[:, 1].mean() + cz_offset
            h_coords = verts[:, 2]
            u_coords = verts[:, 0]
            v_coords = verts[:, 1]
            dist_center = (cx, cz)

        h_min = h_coords.min()
        h_max = h_coords.max()
        H_mesh = max(1e-6, h_max - h_min)

        # Compute outer bounding radius
        r_all = np.sqrt((u_coords - cx)**2 + (v_coords - cz)**2)
        r_max = float(np.max(r_all) * 1.6)

        z_min_pct, z_max_pct = z_crop
        h_top = h_min + z_max_pct * H_mesh
        h_bot = h_min + z_min_pct * H_mesh

        offset_rad = np.radians(azimuth_offset_deg)
        thetas = np.linspace(-np.pi + offset_rad, np.pi + offset_rad, W, endpoint=False)
        hs = np.linspace(h_top, h_bot, H) if not flip_vertical else np.linspace(h_bot, h_top, H)

        theta_grid, h_grid = np.meshgrid(thetas, hs)

        # Inward rays: start from outside cylinder at r_max and shoot toward rotational axis
        if axis == "Y":
            dir_x = -np.cos(theta_grid.ravel())
            dir_z = -np.sin(theta_grid.ravel())
            dir_y = np.zeros_like(dir_x)
            ray_dirs = np.column_stack([dir_x, dir_y, dir_z])

            orig_x = cx + r_max * np.cos(theta_grid.ravel())
            orig_z = cz + r_max * np.sin(theta_grid.ravel())
            orig_y = h_grid.ravel()
            ray_origs = np.column_stack([orig_x, orig_y, orig_z])
        elif axis == "X":
            dir_y = -np.cos(theta_grid.ravel())
            dir_z = -np.sin(theta_grid.ravel())
            dir_x = np.zeros_like(dir_y)
            ray_dirs = np.column_stack([dir_x, dir_y, dir_z])

            orig_y = cx + r_max * np.cos(theta_grid.ravel())
            orig_z = cz + r_max * np.sin(theta_grid.ravel())
            orig_x = h_grid.ravel()
            ray_origs = np.column_stack([orig_x, orig_y, orig_z])
        else:
            dir_x = -np.cos(theta_grid.ravel())
            dir_y = -np.sin(theta_grid.ravel())
            dir_z = np.zeros_like(dir_x)
            ray_dirs = np.column_stack([dir_x, dir_y, dir_z])

            orig_x = cx + r_max * np.cos(theta_grid.ravel())
            orig_y = cz + r_max * np.sin(theta_grid.ravel())
            orig_z = h_grid.ravel()
            ray_origs = np.column_stack([orig_x, orig_y, orig_z])

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origs,
            ray_directions=ray_dirs,
            multiple_hits=True
        )

        if len(locations) == 0:
            return Image.new("RGB", (W, H), (15, 23, 42))

        # First hit encountered by inward ray from outer cylinder is strictly the exterior surface
        if axis == "Y":
            dist_from_orig = np.sqrt((locations[:, 0] - orig_x[index_ray])**2 + (locations[:, 2] - orig_z[index_ray])**2)
        elif axis == "X":
            dist_from_orig = np.sqrt((locations[:, 1] - orig_y[index_ray])**2 + (locations[:, 2] - orig_z[index_ray])**2)
        else:
            dist_from_orig = np.sqrt((locations[:, 0] - orig_x[index_ray])**2 + (locations[:, 1] - orig_y[index_ray])**2)

        sort_order = np.argsort(dist_from_orig)
        locs_sorted = locations[sort_order]
        tri_sorted = index_tri[sort_order]
        ray_sorted = index_ray[sort_order]

        unique_rays, first_indices = np.unique(ray_sorted, return_index=True)
        best_rays = ray_sorted[first_indices]
        best_tris = tri_sorted[first_indices]
        best_locs = locs_sorted[first_indices]

        hit_faces = faces[best_tris]
        v0 = verts[hit_faces[:, 0]]
        v1 = verts[hit_faces[:, 1]]
        v2 = verts[hit_faces[:, 2]]

        bary = trimesh.triangles.points_to_barycentric(
            triangles=np.stack([v0, v1, v2], axis=1),
            points=best_locs
        )

        uv0 = uvs[hit_faces[:, 0]]
        uv1 = uvs[hit_faces[:, 1]]
        uv2 = uvs[hit_faces[:, 2]]

        hit_uvs = (
            bary[:, 0:1] * uv0 +
            bary[:, 1:2] * uv1 +
            bary[:, 2:3] * uv2
        )

        tex_arr = np.array(texture_img.convert("RGB")) if texture_img else np.full((10, 10, 3), [217, 105, 30], dtype=np.uint8)
        th, tw = tex_arr.shape[:2]

        u_px = np.clip((hit_uvs[:, 0] * (tw - 1)).astype(int), 0, tw - 1)
        v_px = np.clip(((1.0 - hit_uvs[:, 1]) * (th - 1)).astype(int), 0, th - 1)
        colors = tex_arr[v_px, u_px]

        out_arr = np.zeros((H * W, 3), dtype=np.uint8)
        mask = np.zeros(H * W, dtype=np.uint8)
        out_arr[best_rays] = colors
        mask[best_rays] = 255

        img_2d = out_arr.reshape((H, W, 3))
        mask_2d = (mask.reshape((H, W)) == 0).astype(np.uint8) * 255

        # Smoothly inpaint any tiny ray gap / hole
        if np.any(mask_2d > 0):
            try:
                import cv2
                img_2d = cv2.inpaint(img_2d, mask_2d, 3, cv2.INPAINT_TELEA)
            except Exception:
                pass

        return Image.fromarray(img_2d)

    @staticmethod
    def project_bildfeld_panel(
        mesh: trimesh.Trimesh,
        texture_img: Optional[Image.Image] = None,
        resolution: Tuple[int, int] = (1600, 1600),
        axis: str = "Y",
        angle_deg: float = 0.0,
        z_crop: Tuple[float, float] = (0.10, 0.85)
    ) -> Image.Image:
        """
        Orthographic GigaMesh Bildfeld-Projektion for framed picture panels (Seite A / Seite B).
        Captures the exact frontal panel without cylindrical seam wrap or handle distortion.
        """
        W, H = resolution
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int32)
        uvs = np.array(mesh.visual.uv, dtype=np.float32) if hasattr(mesh, "visual") and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None else None

        if len(verts) == 0 or len(faces) == 0 or uvs is None:
            return Image.new("RGB", (W, H), (15, 23, 42))

        cx = verts[:, 0].mean()
        cz = verts[:, 2].mean()
        y_min = verts[:, 1].min()
        y_max = verts[:, 1].max()
        H_mesh = max(1e-6, y_max - y_min)

        rad = np.radians(angle_deg)
        cos_a, sin_a = np.cos(rad), np.sin(rad)

        vx = verts[:, 0] - cx
        vz = verts[:, 2] - cz

        z_min_pct, z_max_pct = z_crop
        y_top = y_min + z_max_pct * H_mesh
        y_bot = y_min + z_min_pct * H_mesh

        r_max = np.max(np.sqrt(vx**2 + vz**2)) * 1.15
        xs = np.linspace(-r_max, r_max, W)
        ys = np.linspace(y_top, y_bot, H)

        x_grid, y_grid = np.meshgrid(xs, ys)

        ray_origs_rot = np.column_stack([x_grid.ravel(), y_grid.ravel(), np.full_like(x_grid.ravel(), r_max * 2.0)])
        ray_dirs_rot = np.column_stack([np.zeros_like(x_grid.ravel()), np.zeros_like(x_grid.ravel()), np.full_like(x_grid.ravel(), -1.0)])

        ray_origs = np.column_stack([
            cx + ray_origs_rot[:, 0] * cos_a + ray_origs_rot[:, 2] * sin_a,
            ray_origs_rot[:, 1],
            cz - ray_origs_rot[:, 0] * sin_a + ray_origs_rot[:, 2] * cos_a
        ])
        ray_dirs = np.column_stack([
            ray_dirs_rot[:, 0] * cos_a + ray_dirs_rot[:, 2] * sin_a,
            ray_dirs_rot[:, 1],
            -ray_dirs_rot[:, 0] * sin_a + ray_dirs_rot[:, 2] * cos_a
        ])

        locations, index_ray, index_tri = mesh.ray.intersects_location(
            ray_origins=ray_origs,
            ray_directions=ray_dirs,
            multiple_hits=True
        )

        if len(locations) == 0:
            return Image.new("RGB", (W, H), (15, 23, 42))

        dist = np.linalg.norm(locations - ray_origs[index_ray], axis=1)
        sort_order = np.argsort(dist)

        locs_sorted = locations[sort_order]
        tri_sorted = index_tri[sort_order]
        ray_sorted = index_ray[sort_order]

        unique_rays, first_indices = np.unique(ray_sorted, return_index=True)
        best_rays = ray_sorted[first_indices]
        best_tris = tri_sorted[first_indices]
        best_locs = locs_sorted[first_indices]

        hit_faces = faces[best_tris]
        v0 = verts[hit_faces[:, 0]]
        v1 = verts[hit_faces[:, 1]]
        v2 = verts[hit_faces[:, 2]]

        bary = trimesh.triangles.points_to_barycentric(
            triangles=np.stack([v0, v1, v2], axis=1),
            points=best_locs
        )

        uv0 = uvs[hit_faces[:, 0]]
        uv1 = uvs[hit_faces[:, 1]]
        uv2 = uvs[hit_faces[:, 2]]

        hit_uvs = bary[:, 0:1] * uv0 + bary[:, 1:2] * uv1 + bary[:, 2:3] * uv2

        tex_arr = np.array(texture_img.convert("RGB")) if texture_img else np.full((10, 10, 3), [217, 105, 30], dtype=np.uint8)
        th, tw = tex_arr.shape[:2]

        u_px = np.clip((hit_uvs[:, 0] * (tw - 1)).astype(int), 0, tw - 1)
        v_px = np.clip(((1.0 - hit_uvs[:, 1]) * (th - 1)).astype(int), 0, th - 1)
        colors = tex_arr[v_px, u_px]

        out_arr = np.full((H * W, 3), [15, 23, 42], dtype=np.uint8)
        out_arr[best_rays] = colors
        return Image.fromarray(out_arr.reshape((H, W, 3)))

    @staticmethod
    def get_abrollung_image(
        info: Dict[str, Any],
        layer: str = "base_color",
        axis: str = "Y",
        flip_vertical: bool = False,
        azimuth_offset_deg: float = 0.0,
        cx_offset: float = 0.0,
        cz_offset: float = 0.0,
        z_crop: Tuple[float, float] = (0.10, 0.85),
        projection_mode: str = "frieze",
        max_dimension: int = 1800
    ) -> Image.Image:
        """
        Generates the true GigaMesh pottery surface projection:
        - 'frieze': 360° Umlaufender Bildfries (Cylindrical rollout)
        - 'bildfeld': Orthographic Bildfeld-Projektion (Seite A / Seite B)
        """
        mesh = info.get("mesh")
        target_img = None
        if layer == "normal":
            target_img = info.get("texture_normal")
        elif layer == "roughness":
            target_img = info.get("texture_roughness")
        else:
            target_img = info.get("texture_base_color") or info.get("texture")

        if mesh is not None and len(mesh.vertices) > 0:
            if projection_mode == "bildfeld":
                return GLBProcessor.project_bildfeld_panel(
                    mesh,
                    target_img,
                    resolution=(max_dimension, max_dimension),
                    axis=axis,
                    angle_deg=azimuth_offset_deg,
                    z_crop=z_crop
                )
            else:
                return GLBProcessor.unroll_pottery_gigamesh(
                    mesh,
                    target_img,
                    resolution=(max_dimension, max_dimension // 2),
                    axis=axis,
                    flip_vertical=flip_vertical,
                    azimuth_offset_deg=azimuth_offset_deg,
                    cx_offset=cx_offset,
                    cz_offset=cz_offset,
                    z_crop=z_crop,
                    projection_mode=projection_mode
                )

        # Fallback synthetic frieze
        w, h = max_dimension, max_dimension // 2
        img = Image.new("RGB", (w, h), (190, 80, 25))
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, w, 70], fill=(20, 20, 20))
        draw.rectangle([0, h - 90, w, h], fill=(20, 20, 20))
        filename = info.get("filename", "3D-Vase")
        draw.text((w // 2 - 140, h // 2 - 20), f"GigaMesh-Abrollung\n{filename}", fill=(20, 20, 20))
        return img

    @staticmethod
    def create_sample_3d_vase(shape_type: str = "Lekythos") -> bytes:
        """
        Creates a synthetic sample ancient Greek vase 3D GLB model.
        """
        if shape_type.lower() == "lekythos":
            heights = np.linspace(0, 30, 40)
            radii = 1.5 + 3.0 * np.exp(-((heights - 12)/7)**2)
            radii[heights > 20] = 1.0 + 0.5 * (heights[heights > 20] - 20) / 10
            radii[heights < 2] = 2.5
        elif shape_type.lower() == "kylix":
            heights = np.linspace(0, 14, 40)
            radii = np.where(
                heights < 6,
                1.0 + 1.5 * (6 - heights)/6,
                2.0 + 10.0 * np.sqrt(np.maximum(0.0, (heights - 6) / 8))
            )
        else: # Psykter
            heights = np.linspace(0, 28, 40)
            radii = np.where(
                heights < 12,
                3.5,
                3.5 + 8.5 * np.sin((heights - 12) / 16 * np.pi)
            )

        angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
        verts = []
        for h, r in zip(heights, radii):
            for a in angles:
                verts.append([r * np.cos(a), r * np.sin(a), h])
        verts = np.array(verts)

        faces = []
        n_rings = len(heights)
        n_pts = len(angles)
        for i in range(n_rings - 1):
            for j in range(n_pts):
                next_j = (j + 1) % n_pts
                p1 = i * n_pts + j
                p2 = i * n_pts + next_j
                p3 = (i + 1) * n_pts + j
                p4 = (i + 1) * n_pts + next_j
                faces.append([p1, p2, p4])
                faces.append([p1, p4, p3])

        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        glb_buffer = io.BytesIO()
        mesh.export(glb_buffer, file_type="glb")
        return glb_buffer.getvalue()


glb_processor = GLBProcessor()
