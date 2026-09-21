import json
import xml.etree.ElementTree as ET
from pathlib import Path
import cv2
import mujoco
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Patch MuJoCo's segmentation renderer: size segid2output to the actual max segid,
# not scene.ngeom (decorator/skybox IDs can exceed it on dense scenes, → IndexError).
from mujoco.rendering.classic import renderer as _mj_cls_renderer
if not getattr(_mj_cls_renderer.Renderer, "_segid_patched", False):
    _orig_render = _mj_cls_renderer.Renderer.render

    def _safe_render(self, *, out=None):
        """Patched Renderer.render supporting segmentation output (segid -> objid/objtype table)."""
        if not self._segmentation_rendering:
            return _orig_render(self, out=out)
        original_flags = np.copy(self._scene.flags)
        self._scene.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        self._scene.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        if self._mjr_context is None:
            raise RuntimeError("render cannot be called after close.")
        if self._gl_context:
            self._gl_context.make_current()
        if out is None:
            out = np.empty((self._height, self._width, 3), dtype=np.uint8)
        mujoco.mjr_render(self._rect, self._scene, self._mjr_context)
        mujoco.mjr_readPixels(out, None, self._rect, self._mjr_context)
        image3 = out.astype(np.uint32)
        segimage = image3[:, :, 0] + image3[:, :, 1] * (2**8) + image3[:, :, 2] * (2**16)
        ngeoms = self._scene.ngeom
        max_id = int(segimage.max()) if segimage.size else 0
        table_size = max(ngeoms + 1, max_id + 1)
        segid2output = np.full((table_size, 2), fill_value=-1, dtype=np.int32)
        visible_geoms = [g for g in self._scene.geoms[:ngeoms] if g.segid != -1]
        if visible_geoms:
            vs = np.array([g.segid + 1 for g in visible_geoms], np.int32)
            segid2output[vs, 0] = np.array([g.objid for g in visible_geoms], np.int32)
            segid2output[vs, 1] = np.array([g.objtype for g in visible_geoms], np.int32)
        result = segid2output[segimage]
        np.copyto(self._scene.flags, original_flags)
        if self._gl_context:
            result = np.flipud(result)
        return result

    _mj_cls_renderer.Renderer.render = _safe_render
    _mj_cls_renderer.Renderer._segid_patched = True

def _strip_bodies(xml_path, name_contains):
    """Copy a scene XML with bodies whose names match any needle removed (for top-down map renders)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    stripped_names = set()
    def _collect_body_names(elem):
        """Gather body names under an element matching the needles."""
        name = elem.get("name", "")
        if name:
            stripped_names.add(name)
        for child in elem:
            if child.tag == "body":
                _collect_body_names(child)
    def _remove_matching(parent):
        """Recursively remove matching body elements."""
        to_remove = []
        for child in parent:
            if child.tag == "body":
                name = child.get("name", "")
                if any(kw in name.lower() for kw in name_contains):
                    to_remove.append(child)
                    _collect_body_names(child)
                    continue
            _remove_matching(child)
        for c in to_remove:
            parent.remove(c)
    _remove_matching(root)
    # Remove contact/exclude pairs that reference stripped bodies.
    if stripped_names:
        for tag in ("contact", "equality"):
            for section in root.iter(tag):
                to_remove = []
                for child in section:
                    attrs = (child.get("body1", ""), child.get("body2", ""),
                             child.get("body", ""))
                    if any(a in stripped_names for a in attrs):
                        to_remove.append(child)
                for c in to_remove:
                    section.remove(c)
    out = xml_path.parent / (xml_path.stem + "_tmp_noceiling.xml")
    tree.write(str(out), xml_declaration=True)
    return out


def _floor_geom_ids(model):
    """Geom ids belonging to floor bodies/planes."""
    ids = []
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if name.startswith("room|") or name.startswith("room_"):
            ids.append(i)
        elif "floor" in name.lower() and model.geom_contype[i] == 0:
            ids.append(i)
    if not ids:
        for i in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
            if name == "floor":
                ids.append(i)
    return ids


def _geom_aabb(model, data, geom_ids):
    """World-frame AABB over the given geoms."""
    mins = np.full(3, np.inf)
    maxs = np.full(3, -np.inf)
    for gid in geom_ids:
        pos = data.geom_xpos[gid]
        size = model.geom_size[gid]
        gtype = model.geom_type[gid]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            half = size
        elif gtype == mujoco.mjtGeom.mjGEOM_MESH:
            meshid = model.geom_dataid[gid]
            vert_start = model.mesh_vertadr[meshid]
            vert_count = model.mesh_vertnum[meshid]
            verts = model.mesh_vert[vert_start:vert_start + vert_count]
            mat = data.geom_xmat[gid].reshape(3, 3)
            world_verts = (mat @ verts.T).T + pos
            mins = np.minimum(mins, world_verts.min(axis=0))
            maxs = np.maximum(maxs, world_verts.max(axis=0))
            continue
        else:
            half = np.array([size[0]] * 3)
        mins = np.minimum(mins, pos - half)
        maxs = np.maximum(maxs, pos + half)
    center = (mins + maxs) / 2.0
    full_size = maxs - mins
    return center, full_size


def _circular_kernel(radius):
    """Disk-shaped uint8 kernel for occupancy dilation."""
    size = radius * 2 + 1
    kernel = np.zeros((size, size), np.uint8)
    cv2.circle(kernel, (radius, radius), radius, 1, -1)
    return kernel


def _door_geom_ids(model):
    """Geom ids of hinged door panels and of the doorframes around them."""
    parent_to_child = {}
    for body_id in range(model.nbody):
        root_id = model.body_rootid[body_id]
        root_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, root_id) or ""
        if root_name.startswith("door_") or root_name.startswith("doorway_"):
            parent_to_child.setdefault(root_id, []).append(body_id)

    door_body_ids, doorway_body_ids = [], []
    for children in parent_to_child.values():
        for child_id in children:
            jntadr = model.body(child_id).jntadr.item()
            if (jntadr >= 0
                    and model.joint(jntadr).type == mujoco.mjtJoint.mjJNT_HINGE
                    and model.joint(jntadr).qpos0.item() != 0.0):
                door_body_ids.append(child_id)
                doorway_body_ids.extend(children)
            if jntadr < 0 and len(children) == 2:
                doorway_body_ids.append(child_id)

    door_gids, doorframe_gids = [], []
    for gid in range(model.ngeom):
        bid = model.geom_bodyid[gid]
        if bid in door_body_ids or model.body(bid).parentid.item() in door_body_ids:
            door_gids.append(gid)
        if model.body_rootid[bid] in doorway_body_ids:
            doorframe_gids.append(gid)
    return door_gids, doorframe_gids


def _render_topdown_segmentation(model, data, floor_ids, px_per_m):
    """Orthographic top-down segmentation render covering the floor's bounding box."""
    center, size = _geom_aabb(model, data, floor_ids)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(size)):
        all_geoms = [i for i in range(model.ngeom)
                     if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE]
        center, size = _geom_aabb(model, data, all_geoms)
    size += np.array([2.0, 2.0, 0.0])
    h = max(round(px_per_m * size[0]), 1)
    w = max(round(px_per_m * size[1]), 1)
    effective_px = h / size[0]
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)

    renderer = mujoco.Renderer(model, height=h, width=w, max_geom=max(20000, model.ngeom * 4))
    renderer.enable_segmentation_rendering()
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = center
    cam.distance = 5.0
    cam.azimuth = 0
    cam.elevation = -90
    scene = mujoco.MjvScene(model, maxgeom=10000)
    opt = mujoco.MjvOption()
    mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
    for i in range(2):
        scene.camera[i].orthographic = 1
        scene.camera[i].frustum_bottom = -size[0] / 2
        scene.camera[i].frustum_top = size[0] / 2
    renderer.update_scene(data, camera=cam)
    for i in range(2):
        renderer._scene.camera[i].orthographic = 1
        renderer._scene.camera[i].frustum_bottom = -size[0] / 2
        renderer._scene.camera[i].frustum_top = size[0] / 2
    seg = renderer.render()
    renderer.disable_segmentation_rendering()

    cam_to_world = np.eye(4)
    cam_to_world[:3, 3] = renderer._scene.camera[0].pos
    cam_x = np.cross(renderer._scene.camera[0].up, -renderer._scene.camera[0].forward)
    cam_to_world[:3, :3] = np.column_stack(
        (cam_x, renderer._scene.camera[0].up, -renderer._scene.camera[0].forward))
    renderer.close()
    return seg[:, :, 0].astype(int), cam_to_world, effective_px, h, w


def _occupancy_from_segmentation(seg_geom, floor_ids, door_gids, doorframe_gids,
                                 agent_radius, effective_px):
    """Free-space mask: anything that is not floor, minus door openings, dilated by the agent radius."""
    occupied = np.ones(seg_geom.shape, dtype=bool)
    for fid in floor_ids:
        occupied &= (seg_geom != fid)

    # A door opening is the doorframe minus the panel, so the robot can pass through it.
    occ_door = np.zeros(seg_geom.shape, dtype=bool)
    for did in door_gids:
        occ_door[seg_geom == did] = True
    occ_doorframe = np.zeros(seg_geom.shape, dtype=bool)
    for did in doorframe_gids:
        occ_doorframe[seg_geom == did] = True
    door_path = cv2.dilate((occ_doorframe & ~occ_door).astype(np.uint8),
                           _circular_kernel(15)).astype(bool)
    occupied[door_path] = False

    if agent_radius is not None and agent_radius > 0:
        rad_px = max(1, int(agent_radius * effective_px))
        occupied = cv2.dilate(occupied.astype(np.uint8), _circular_kernel(rad_px)).astype(bool)
    return ~occupied


def _map_transforms(cam_to_world, effective_px, h, w):
    """World<->map homographies for the rendered top-down image."""
    cam_to_map = np.array([[0, -effective_px, 0, h / 2], [effective_px, 0, 0, w / 2]])
    cam_inv = np.eye(4)
    cam_inv[:3, :3] = cam_to_world[:3, :3].T
    cam_inv[:3, 3] = -cam_to_world[:3, :3].T @ cam_to_world[:3, 3]
    world_to_map = cam_to_map @ cam_inv
    map_to_centered = np.array([[0, 1, -w / 2], [-1, 0, h / 2], [0, 0, 1]])
    centered_to_cam = np.array([[1 / effective_px, 0, 0], [0, 1 / effective_px, 0], [0, 0, 1]])
    cam_to_world_floor = cam_to_world[:-1, [0, 1, 3]].copy()
    cam_to_world_floor[2, 2] = 0
    return world_to_map, cam_to_world_floor @ centered_to_cam @ map_to_centered


class OccupancyMap:

    def __init__(self, occupancy, world_to_map, map_to_world, px_per_m):
        """Wrap a boolean free-space grid with its world<->map transforms."""
        self.occupancy = occupancy
        self.world_to_map = world_to_map
        self.map_to_world = map_to_world
        self.px_per_m = px_per_m

    @staticmethod
    def generate(xml_path, agent_radius=0.15, px_per_m=200, force=False):
        """Render a scene top-down, segment floor vs obstacles, dilate by agent radius, and save the map PNG."""
        xml_path = Path(xml_path)
        out_path = xml_path.with_name(xml_path.stem + "_thormap.png")
        if out_path.exists() and not force:
            return out_path

        tmp_xml = _strip_bodies(xml_path, ["ceiling", "light"])
        try:
            model = mujoco.MjModel.from_xml_path(str(tmp_xml))
        finally:
            tmp_xml.unlink(missing_ok=True)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        floor_ids = _floor_geom_ids(model)
        assert floor_ids, f"No floor geoms found in {xml_path}"

        door_gids, doorframe_gids = _door_geom_ids(model)
        seg_geom, cam_to_world, effective_px, h, w = _render_topdown_segmentation(
            model, data, floor_ids, px_per_m)
        occupancy = _occupancy_from_segmentation(
            seg_geom, floor_ids, door_gids, doorframe_gids, agent_radius, effective_px)
        world_to_map, map_to_world = _map_transforms(cam_to_world, effective_px, h, w)

        img = Image.fromarray(occupancy.astype(np.uint8) * 255)
        metadata = PngInfo()
        metadata.add_text("world_to_map", json.dumps(world_to_map.tolist()))
        metadata.add_text("map_to_world", json.dumps(map_to_world.tolist()))
        metadata.add_text("px_per_m", json.dumps(float(effective_px)))
        img.save(str(out_path), pnginfo=metadata)
        return out_path

    @classmethod
    def from_scene(cls, scene, agent_radius=0.15):
        """Load (generating if missing) the occupancy map next to a scene's XML."""
        map_path = scene.xml_path.with_name(scene.xml_path.stem + "_thormap.png")
        if not map_path.exists():
            cls.generate(scene.xml_path, agent_radius=agent_radius, force=True)
        return cls.load(str(map_path))

    @classmethod
    def load(cls, path):
        """Load an occupancy PNG plus its transforms from PNG metadata."""
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 3:
            occ_raw = arr[:, :, 0]
        else:
            occ_raw = arr
        occupancy = occ_raw > 0
        world_to_map = np.array(json.loads(img.info["world_to_map"]))
        map_to_world = np.array(json.loads(img.info["map_to_world"]))
        px_per_m = float(json.loads(img.info["px_per_m"]))
        return cls(occupancy, world_to_map, map_to_world, px_per_m)

    def _world_to_px(self, xy):
        """World xy to integer (row, col) map coordinates."""
        pos = np.array([xy[0], xy[1], 0.0, 1.0])
        rc = self.world_to_map @ pos
        return np.round(rc).astype(int)

    def is_free(self, xy):
        """True if the world position lands on a free map cell."""
        rc = self._world_to_px(xy)
        r, c = rc[0], rc[1]
        h, w = self.occupancy.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        return bool(self.occupancy[r, c])

    def dilated(self, extra_radius_m):
        """Return a copy with obstacles inflated by extra_radius_m. World transforms shared."""
        if extra_radius_m <= 0:
            return self
        rad_px = max(1, int(extra_radius_m * self.px_per_m))
        kernel = _circular_kernel(rad_px)
        free = self.occupancy.astype(np.uint8)
        free_eroded = cv2.erode(free, kernel).astype(bool)
        return type(self)(free_eroded, self.world_to_map.copy(),
                          self.map_to_world.copy(), self.px_per_m)

    def _free_labels(self):
        """Cached connected-component labeling of the free space."""
        lab = getattr(self, '_free_labels_cache', None)
        if lab is None:
            from scipy import ndimage
            lab, _ = ndimage.label(self.occupancy.astype(np.uint8))
            self._free_labels_cache = lab
        return lab

    def label_at(self, xy):
        """Connected-component id of the free cell at xy. 0 = occupied / out-of-bounds."""
        rc = self._world_to_px(xy)
        r, c = int(rc[0]), int(rc[1])
        h, w = self.occupancy.shape
        if r < 0 or r >= h or c < 0 or c >= w:
            return 0
        return int(self._free_labels()[r, c])

    def nearest_free_label(self, xy, max_radius_px=80):
        """Component label of the free cell nearest to xy within max_radius_px."""
        lab = self._free_labels()
        rc = self._world_to_px(xy)
        r0, c0 = int(rc[0]), int(rc[1])
        h, w = self.occupancy.shape
        if 0 <= r0 < h and 0 <= c0 < w and lab[r0, c0] > 0:
            return int(lab[r0, c0])
        for rad in range(1, max_radius_px + 1):
            r_lo, r_hi = max(0, r0 - rad), min(h - 1, r0 + rad)
            c_lo, c_hi = max(0, c0 - rad), min(w - 1, c0 + rad)
            sub = lab[r_lo:r_hi + 1, c_lo:c_hi + 1]
            free = sub[sub > 0]
            if free.size:
                return int(free.flat[0])
        return 0

    def same_free_component(self, xy_a, xy_b):
        """True if two world points are in the same connected free region."""
        la = self.label_at(xy_a)
        if la == 0:
            return False
        return la == self.nearest_free_label(xy_b)

    def any_free_in_annulus(self, center_xy, r_min, r_max):
        """True if any free cell exists in the ring [r_min, r_max] around center."""
        rc = self._world_to_px(center_xy)
        r0, c0 = int(rc[0]), int(rc[1])
        rad = int(np.ceil(r_max * self.px_per_m))
        h, w = self.occupancy.shape
        rlo, rhi = max(0, r0 - rad), min(h, r0 + rad + 1)
        clo, chi = max(0, c0 - rad), min(w, c0 + rad + 1)
        if rlo >= rhi or clo >= chi:
            return False
        ys, xs = np.ogrid[rlo:rhi, clo:chi]
        d2 = (ys - r0) ** 2 + (xs - c0) ** 2
        rmin_px2 = (r_min * self.px_per_m) ** 2
        ring = (d2 >= rmin_px2) & (d2 <= rad * rad)
        return bool(self.occupancy[rlo:rhi, clo:chi][ring].any())

    def sample_near(self, target_xy, radius_min=0.0, radius_max=0.7, max_attempts=500, np_random=None):
        """Rejection-sample a free world xy in an annulus around the target."""
        if np_random is None:
            np_random = np.random.default_rng()
        for _ in range(max_attempts):
            theta = np_random.uniform(0, 2 * np.pi)
            r = np_random.uniform(radius_min, radius_max)
            pt = target_xy[:2] + np.array([r * np.cos(theta), r * np.sin(theta)])
            if self.is_free(pt):
                return pt
        return None
    