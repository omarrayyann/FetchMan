import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from molmospaces import ASSETS_DIR
from molmospaces.components.constants import is_pickup_type
from molmospaces.components.object import Object
from molmospaces.components.occupancy_map import OccupancyMap


def _strip_skybox(xml_path: Path) -> Path:
    """Cache a skybox-free copy of the scene XML (saves ~18 MB texture memory)."""
    out = xml_path.with_name(xml_path.stem + "_noskybox.xml")
    if out.exists():
        return out
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    removed = False
    for parent in root.iter():
        for tex in list(parent.findall("texture")):
            if tex.attrib.get("type") == "skybox":
                parent.remove(tex)
                removed = True
    if not removed:
        return xml_path
    # Atomic publish: parallel workers race on a cold cache, and a reader that
    # catches a half-written file dies with an XML parse error. Write to a
    # per-process temp name, then os.replace (atomic on POSIX).
    import os
    tmp = out.with_name(f"{out.stem}.{os.getpid()}.tmp")
    tree.write(str(tmp))
    os.replace(tmp, out)
    return out


class Scene:
    # Number of solid-color materials pre-allocated for the texture randomizer
    # to pick from as a non-textured alternative.
    _N_SOLID_COLOR_MATERIALS = 8

    def __init__(self, xml_path, robot_xml, mobile_regex=None, scene_textures=None,
                 ):
        """Compile a scene XML with the robot attached, apply texture overrides, and index all objects."""
        xml_path = Path(xml_path)
        if not xml_path.is_absolute():
            xml_path = ASSETS_DIR / xml_path
        self.xml_path = xml_path
        self.robot_xml = Path(robot_xml)
        self._robot_prefix = "robot_0/"

        spec = self._build_spec()
        self._add_scene_textures(spec, scene_textures)
        metadata = self._load_metadata()
        self._optimize(spec, metadata, mobile_regex)
        self._add_grasp_probe(spec)
        self._add_pelvis_weld(spec)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self._damp_freejoints()
        self._snapshot_init_arrays()
        self._index_scene_materials()
        self._index_objects(metadata)

    def _build_spec(self):
        """Load the scene XML and attach the robot under the robot prefix."""
        spec = mujoco.MjSpec.from_file(str(_strip_skybox(self.xml_path)))
        robot_spec = mujoco.MjSpec.from_file(str(self.robot_xml))
        frame = spec.worldbody.add_frame()
        frame.attach_body(robot_spec.worldbody.first_body(), self._robot_prefix, '')
        spec.option.enableflags |= int(mujoco.mjtEnableBit.mjENBL_SLEEP)
        return spec

    def _add_scene_textures(self, spec, scene_textures):
        """Add a material per randomizable texture, plus solid-color placeholders."""
        self.scene_matids: dict[str, list[int]] = {}
        self.scene_color_matids: list[int] = []
        self._scene_texture_paths: dict[str, list[str]] = dict(scene_textures or {})
        for cat, paths in self._scene_texture_paths.items():
            for i, tex_path in enumerate(paths):
                spec.add_texture(
                    name=f"__scene_tex_{cat}_{i}",
                    type=int(mujoco.mjtTexture.mjTEXTURE_2D),
                    file=str(Path(tex_path).resolve()),
                )
                spec.add_material(
                    name=f"__scene_mat_{cat}_{i}",
                    textures=["", f"__scene_tex_{cat}_{i}"],
                    texrepeat=[2.0, 2.0],
                    texuniform=1,
                )
        if self._scene_texture_paths:
            for i in range(self._N_SOLID_COLOR_MATERIALS):
                spec.add_material(name=f"__scene_color_{i}", rgba=[1.0, 1.0, 1.0, 1.0])

    def _load_metadata(self):
        """Per-object metadata from the scene's _metadata.json sidecar, or empty."""
        meta_path = self.xml_path.with_name(self.xml_path.stem + "_metadata.json")
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f).get("objects", {})

    def _add_pelvis_weld(self, spec):
        """Add an inactive weld that can lock the robot base (G1 only)."""
        if not any(b.name == f"{self._robot_prefix}pelvis" for b in spec.bodies):
            return
        weld = spec.add_equality()
        weld.type = mujoco.mjtEq.mjEQ_WELD
        weld.name = "pelvis_weld"
        weld.objtype = mujoco.mjtObj.mjOBJ_BODY
        weld.name1 = f"{self._robot_prefix}pelvis"
        weld.name2 = ""
        weld.active = False
        weld.solref = [0.0002, 1.0]
        weld.solimp = [0.999, 0.9999, 0.0001, 0.5, 2.0]

    def _damp_freejoints(self):
        """Rate-capped anti-tumble damping; a flat value NaN-explodes gram-scale objects."""
        LAMBDA_MAX = 20.0
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                d0 = int(self.model.jnt_dofadr[jid])
                for k in range(6):
                    self.model.dof_damping[d0 + k] = min(
                        1.0, LAMBDA_MAX * float(self.model.dof_M0[d0 + k]))

    def _snapshot_init_arrays(self):
        """Copy the model arrays that reset() restores after randomization mutates them."""
        self._init_body_pos = self.model.body_pos.copy()
        if self.model.nlight > 0:
            self._init_light_pos = self.model.light_pos.copy()
            self._init_light_dir = self.model.light_dir.copy()
            self._init_light_specular = self.model.light_specular.copy()
            self._init_light_ambient = self.model.light_ambient.copy()
            self._init_light_diffuse = self.model.light_diffuse.copy()
            self._init_light_active = self.model.light_active.copy()
            if hasattr(self.model, "light_castshadow"):
                self._init_light_castshadow = self.model.light_castshadow.copy()
        self._init_body_simple = self.model.body_simple.copy()
        self._init_body_sameframe = self.model.body_sameframe.copy()
        self._init_geom_matid = self.model.geom_matid.copy()

    def _index_scene_materials(self):
        """Map each texture category to its material IDs and the non-collidable geoms using it."""
        self.scene_matids = {}
        self.scene_geom_ids: dict[str, list[int]] = {}
        if not self._scene_texture_paths:
            return
        from molmospaces.components.constants import classify_scene_geom
        for cat, paths in self._scene_texture_paths.items():
            mids = [mid for i in range(len(paths))
                    if (mid := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL,
                                                 f"__scene_mat_{cat}_{i}")) >= 0]
            if mids:
                self.scene_matids[cat] = mids
                self.scene_geom_ids[cat] = []
        for i in range(self._N_SOLID_COLOR_MATERIALS):
            mid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MATERIAL, f"__scene_color_{i}")
            if mid >= 0:
                self.scene_color_matids.append(mid)
        for gid in range(self.model.ngeom):
            if int(self.model.geom_contype[gid]) != 0 or int(self.model.geom_conaffinity[gid]) != 0:
                continue
            bname = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                       int(self.model.geom_bodyid[gid])) or "").lower()
            if bname.startswith(self._robot_prefix.lower()) or bname in ("grasp_probe", "gripper_probe"):
                continue
            gname = (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or "").lower()
            cat = classify_scene_geom(bname, gname)
            if cat is not None and cat in self.scene_geom_ids:
                self.scene_geom_ids[cat].append(gid)

    def _index_objects(self, metadata):
        """Build the Object list for every non-robot body, keyed by name."""
        self._objects: list[Object] = []
        for i in range(self.model.nbody):
            name = self.model.body(i).name
            if not name or name.startswith(self._robot_prefix) or name in ("grasp_probe",):
                continue
            jnt_adr = self.model.body(i).jntadr[0]
            has_fj = jnt_adr >= 0 and self.model.jnt_type[jnt_adr] == mujoco.mjtJoint.mjJNT_FREE
            meta = metadata.get(name, {})
            self._objects.append(Object(
                body_id=i, name=name,
                category=meta.get("category", name.split("_")[0]),
                asset_id=meta.get("asset_id", ""),
                is_static=meta.get("is_static", not has_fj),
                has_freejoint=has_fj,
            ))
        self._by_name = {o.name: o for o in self._objects}

    @property
    def objects(self):
        """All indexed objects by name."""
        return self._objects
    @property
    def pickable(self):
        """Objects with a freejoint (candidates for pick tasks)."""
        return [o for o in self._objects if o.has_freejoint]
    @property
    def get(self, name):
        """Look up an object by name."""
        return self._by_name[name]

    def forward(self):
        """Run mj_forward on the scene."""
        mujoco.mj_forward(self.model, self.data)

    def step(self, n=1):
        """Advance physics n steps."""
        mujoco.mj_step(self.model, self.data, nstep=n)

    def reset(self):
        """Reset data and restore all model fields mutated by randomization."""
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[:] = self._init_body_pos
        self.model.body_simple[:] = self._init_body_simple
        self.model.body_sameframe[:] = self._init_body_sameframe
        self.model.geom_matid[:] = self._init_geom_matid
        if self.model.nlight > 0:
            self.model.light_pos[:] = self._init_light_pos
            self.model.light_dir[:] = self._init_light_dir
            self.model.light_specular[:] = self._init_light_specular
            self.model.light_ambient[:] = self._init_light_ambient
            self.model.light_diffuse[:] = self._init_light_diffuse
            self.model.light_active[:] = self._init_light_active
            if hasattr(self.model, "light_castshadow"):
                self.model.light_castshadow[:] = self._init_light_castshadow
        mujoco.mj_forward(self.model, self.data)

    def get_body_descendants(self, body_id):
        """Set of body ids at or below body_id in the kinematic tree."""
        body_set = {body_id}
        for i in range(self.model.nbody):
            bid = i
            while bid > 0:
                if bid == body_id:
                    body_set.add(i)
                    break
                bid = self.model.body_parentid[bid]
        return body_set

    def occupancy_map(self, agent_radius=0.35):
        """Cached OccupancyMap for this scene."""
        return OccupancyMap.from_scene(self, agent_radius)

    def _optimize(self, spec, metadata, mobile_regex):
        """Strip joints and collisions from non-candidate bodies so they become inert static geometry."""
        pattern = re.compile(mobile_regex, re.IGNORECASE) if mobile_regex is not None else None

        def _strip(parent):
            """Recursively drop invisible and collision-irrelevant bodies."""
            for child in parent.bodies:
                if child.name and child.name.startswith(self._robot_prefix):
                    _strip(child)
                    continue

                keep = False
                if pattern is not None:
                    meta = metadata.get(child.name, {})
                    cat = meta.get("category", child.name.split("_")[0])
                    name_match = bool(pattern.search(child.name))
                    # Keep if THOR pickup whitelist OR metadata explicitly marks it non-static (objaverse).
                    is_static_meta = meta.get("is_static", None)
                    if is_static_meta is False and name_match:
                        keep = True
                    elif is_pickup_type(cat) and name_match:
                        keep = True

                if not keep:
                    joints = list(child.joints)
                    for jnt in joints:
                        spec.delete(jnt)
                    if joints:
                        for geom in child.geoms:
                            geom.contype = 0
                            geom.conaffinity = 0

                _strip(child)

        _strip(spec.worldbody)

    def _add_grasp_probe(self, spec):
        """Inject the gripper probe body used for grasp clearance sweeps."""
        W, L, H = 0.08, 0.03, 0.01
        BASE = np.array([0.0, 0.0, -0.04])
        hw = W / 2
        probe = spec.worldbody.add_body(name="grasp_probe", pos=[0, 0, 10], gravcomp=1)
        probe.add_freejoint(name="grasp_probe_joint")
        _kw = dict(type=mujoco.mjtGeom.mjGEOM_CYLINDER, rgba=[1, 0, 0, 0.6],
                   group=0, contype=0, conaffinity=0b1111)
        for fromto in [
            ([0, -hw, 0], [0, hw, 0]),
            ([0, -hw, 0], [0, -hw, L]),
            ([0, hw, 0], [0, hw, L]),
        ]:
            g = probe.add_geom(**_kw)
            g.size[0] = H / 2
            g.fromto[:3] = np.array(fromto[0]) + BASE
            g.fromto[3:] = np.array(fromto[1]) + BASE

        gripper_probe_xml = self.robot_xml.parent / "gripper_probe.xml"
        if gripper_probe_xml.exists():
            gprobe_spec = mujoco.MjSpec.from_file(str(gripper_probe_xml))
            frame = spec.worldbody.add_frame()
            frame.attach_body(gprobe_spec.worldbody.first_body(), "", "")

