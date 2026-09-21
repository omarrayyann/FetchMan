"""Cubemap-based fisheye renderer for MuJoCo (5 pinhole faces, equidistant warp)."""
from __future__ import annotations

import contextlib

import cv2
import mujoco
import numpy as np
import torch
import torch.nn.functional as F


_FACES = (
    ('center', np.array([1, 0, 0], dtype=np.float32),
               np.array([0, 1, 0], dtype=np.float32),
               np.array([0, 0, 1], dtype=np.float32)),
    ('up',     np.array([1, 0, 0], dtype=np.float32),
               np.array([0, 0, 1], dtype=np.float32),
               np.array([0, -1, 0], dtype=np.float32)),
    ('down',   np.array([1, 0, 0], dtype=np.float32),
               np.array([0, 0, -1], dtype=np.float32),
               np.array([0, 1, 0], dtype=np.float32)),
    ('left',   np.array([0, 0, -1], dtype=np.float32),
               np.array([0, 1, 0], dtype=np.float32),
               np.array([1, 0, 0], dtype=np.float32)),
    ('right',  np.array([0, 0, 1], dtype=np.float32),
               np.array([0, 1, 0], dtype=np.float32),
               np.array([-1, 0, 0], dtype=np.float32)),
)
HEAD_FISHEYE_K = np.array([
    [801.6382129934864, 0.0,               976.1246839545557],
    [0.0,               802.1081824931498, 542.7122090223202],
    [0.0,               0.0,               1.0],
], dtype=np.float64)
HEAD_FISHEYE_D = np.array([
    -0.02559442829261663,
     0.008371943913215045,
    -0.006921566406199126,
     0.0010132813066123071,
], dtype=np.float64)
HEAD_FISHEYE_IMAGE_SIZE = (1920, 1080)  # (W, H) of the calibration capture.


class FisheyeRenderer:
    def __init__(self, model, tile_cam_names, tile_size=256, output_h=240, output_w=240,
                 weight_power=4.0, K=None, D=None, image_size=None):
        """Build the pixel-to-ray LUT from the OpenCV fisheye model calibrated to the real head camera."""
        if len(tile_cam_names) != 5:
            raise ValueError(f"need exactly 5 tile cameras, got {len(tile_cam_names)}")
        self.model = model
        self.tile_cam_ids = []
        for name in tile_cam_names:
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0:
                raise ValueError(f"tile camera {name!r} not found in model")
            self.tile_cam_ids.append(cid)
        self.tile_size = int(tile_size)
        self.output_h = int(output_h)
        self.output_w = int(output_w)
        self.weight_power = float(weight_power)
        tile_fovy = float(model.cam_fovy[self.tile_cam_ids[0]])
        if tile_fovy < 90.0:
            raise ValueError(f"tile fovy {tile_fovy} too small; need >=90 (recommend 100)")
        self.tile_fovy_rad = np.radians(tile_fovy)
        self.K = (HEAD_FISHEYE_K if K is None else np.asarray(K, dtype=np.float64)).copy()
        self.D = (HEAD_FISHEYE_D if D is None else np.asarray(D, dtype=np.float64)).copy()
        self.image_size = tuple(HEAD_FISHEYE_IMAGE_SIZE if image_size is None else image_size)
        self._build_lut()

    def set_intrinsics(self, K=None, D=None):
        """Replace K and/or D in place and rebuild the LUT."""
        if K is not None:
            self.K[:] = K
        if D is not None:
            self.D[:] = D
        self._build_lut()
        self._gpu_ready = False
        self._grids_gpu = None
        self._weights_gpu = None
        self._weight_sum_gpu = None
        self._in_circle_gpu = None

    def project_camera_point(self, p_cam):
        """Project a 3D camera-frame point to a fisheye pixel, or None if it is behind the optical plane."""
        x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        X, Y, Z = x, -y, -z   # MuJoCo cam -> OpenCV cam (z forward, y down)
        if Z <= 1e-6:
            return None
        a, b = X / Z, Y / Z
        r = float(np.hypot(a, b))
        theta = np.arctan(r)
        k1, k2, k3, k4 = self.D
        theta_d = theta * (1.0 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
        scale = (theta_d / r) if r > 1e-9 else 1.0
        xp, yp = a * scale, b * scale
        fx, fy, cx, cy = self._scaled_intrinsics()
        return fx * xp + cx, fy * yp + cy

    def _scaled_intrinsics(self):
        """Calibration K scaled from image_size to the renderer's output size."""
        cal_w, cal_h = self.image_size
        sx = self.output_w / float(cal_w)
        sy = self.output_h / float(cal_h)
        return (self.K[0, 0] * sx, self.K[1, 1] * sy,
                self.K[0, 2] * sx, self.K[1, 2] * sy)

    def _undistort(self, theta_d):
        """Invert the OpenCV fisheye polynomial for theta; 10 Newton steps suffice below ~1.3 rad."""
        k1, k2, k3, k4 = self.D
        theta = theta_d.copy()
        for _ in range(10):
            t2 = theta * theta
            t4 = t2 * t2
            t6 = t4 * t2
            t8 = t4 * t4
            resid = theta * (1.0 + k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8) - theta_d
            deriv = 1.0 + 3.0 * k1 * t2 + 5.0 * k2 * t4 + 7.0 * k3 * t6 + 9.0 * k4 * t8
            theta = theta - resid / np.where(np.abs(deriv) > 1e-12, deriv, 1e-12)
        return theta

    def _output_rays(self):
        """Unproject every output pixel to a camera-frame ray; returns (ray, valid mask)."""
        ys, xs = np.mgrid[:self.output_h, :self.output_w].astype(np.float64)
        fx, fy, cx, cy = self._scaled_intrinsics()
        x_norm = (xs - cx) / fx
        y_norm = (ys - cy) / fy
        theta_d = np.sqrt(x_norm * x_norm + y_norm * y_norm)
        phi = np.arctan2(y_norm, x_norm)
        theta = self._undistort(theta_d)
        in_circle = np.isfinite(theta) & (theta >= 0.0) & (theta < np.pi * 0.5 + 0.3)
        sin_t = np.sin(theta)
        ray = (sin_t * np.cos(phi), -sin_t * np.sin(phi), -np.cos(theta))
        return ray, in_circle

    def _project_onto_tile(self, ray, axes):
        """Project rays onto one cubemap face; returns (u, v, blend weight)."""
        rx, ry, rz = ray
        ax_v, ay_v, az_v = axes
        f = (self.tile_size * 0.5) / np.tan(self.tile_fovy_rad * 0.5)
        c = (self.tile_size - 1) * 0.5
        px = rx * ax_v[0] + ry * ax_v[1] + rz * ax_v[2]
        py = rx * ay_v[0] + ry * ay_v[1] + rz * ay_v[2]
        pz = rx * az_v[0] + ry * az_v[1] + rz * az_v[2]
        depth = -pz
        with np.errstate(divide='ignore', invalid='ignore'):
            u = px / depth * f + c
            v = -py / depth * f + c
        in_bounds = ((depth > 1e-6)
                     & (u >= 0) & (u <= self.tile_size - 1)
                     & (v >= 0) & (v <= self.tile_size - 1))
        weight = np.where(in_bounds, np.clip(depth, 0.0, 1.0) ** self.weight_power, 0.0)
        u = np.clip(u, 0, self.tile_size - 1).astype(np.float32)
        v = np.clip(v, 0, self.tile_size - 1).astype(np.float32)
        return u, v, weight.astype(np.float32)

    def _build_lut(self):
        """Precompute the fisheye pixel -> (tile, u, v) sampling LUT from the intrinsics."""
        ray, self._in_circle = self._output_rays()
        self._projs = []
        grids_norm, weights = [], []
        for _, ax_v, ay_v, az_v in _FACES:
            u, v, weight = self._project_onto_tile(ray, (ax_v, ay_v, az_v))
            self._projs.append((u, v, weight))
            gx = 2.0 * u / max(self.tile_size - 1, 1) - 1.0
            gy = 2.0 * v / max(self.tile_size - 1, 1) - 1.0
            grids_norm.append(np.stack([gx, gy], axis=-1).astype(np.float32))
            weights.append(weight)

        # Uploaded lazily on first render so CUDA initializes in the right worker.
        self._grids_np = np.stack(grids_norm, axis=0)    # (5, H, W, 2)
        self._weights_np = np.stack(weights, axis=0)     # (5, H, W)
        self._gpu_ready = False
        self._grids_gpu = None
        self._weights_gpu = None
        self._weight_sum_gpu = None
        self._in_circle_gpu = None

    @contextlib.contextmanager
    def _seamless_headlight(self):
        """Fold the headlight into ambient so the 5 faces shade identically and seams vanish."""
        hl = self.model.vis.headlight
        orig_amb, orig_dif, orig_spc = hl.ambient.copy(), hl.diffuse.copy(), hl.specular.copy()
        hl.ambient[:] = orig_amb + orig_dif
        hl.diffuse[:] = 0.0
        hl.specular[:] = 0.0
        try:
            yield
        finally:
            hl.ambient[:] = orig_amb
            hl.diffuse[:] = orig_dif
            hl.specular[:] = orig_spc

    def _render_tiles(self, data, renderer, scene_option=None):
        """Render the 5 cubemap faces; only the first needs a full scene traversal."""
        if scene_option is not None:
            renderer.update_scene(data, self.tile_cam_ids[0], scene_option)
        else:
            renderer.update_scene(data, self.tile_cam_ids[0])
        tiles = [renderer.render()]
        for cid in self.tile_cam_ids[1:]:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            cam.fixedcamid = cid
            mujoco.mjv_updateCamera(self.model, data, cam, renderer.scene)
            tiles.append(renderer.render())
        return tiles

    def _upload_lut(self):
        """Move the LUT to the GPU on first use, once CUDA is live in this process."""
        if self._gpu_ready:
            return
        dev = torch.device('cuda')
        self._grids_gpu = torch.from_numpy(self._grids_np).to(dev, non_blocking=True)
        self._weights_gpu = torch.from_numpy(self._weights_np).to(dev, non_blocking=True)
        self._weight_sum_gpu = self._weights_gpu.sum(dim=0).clamp(min=1e-6)
        self._in_circle_gpu = torch.from_numpy(self._in_circle).to(dev, non_blocking=True)
        self._gpu_ready = True

    def _resample_gpu(self, tiles):
        """Weighted grid_sample of the 5 faces into one fisheye image."""
        self._upload_lut()
        dev = self._grids_gpu.device
        tiles_gpu = (torch.from_numpy(np.stack(tiles, axis=0))
                     .to(dev, non_blocking=True).float().permute(0, 3, 1, 2))
        sampled = F.grid_sample(tiles_gpu, self._grids_gpu, mode='bilinear',
                                padding_mode='border', align_corners=True)
        weighted = (sampled * self._weights_gpu.unsqueeze(1)).sum(dim=0)
        out = (weighted / self._weight_sum_gpu.unsqueeze(0)).clamp(0, 255).permute(1, 2, 0)
        out = torch.where(self._in_circle_gpu.unsqueeze(-1), out, torch.zeros_like(out))
        return out.byte().cpu().numpy()

    def _resample_cpu(self, tiles):
        """cv2.remap fallback for the same weighted resample when CUDA is unavailable."""
        sample_acc = np.zeros((self.output_h, self.output_w, 3), dtype=np.float32)
        weight_acc = np.zeros((self.output_h, self.output_w), dtype=np.float32)
        for tile, (u, v, w) in zip(tiles, self._projs):
            sample = cv2.remap(tile, u, v, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
            sample_acc += sample.astype(np.float32) * w[..., None]
            weight_acc += w
        valid = (weight_acc > 1e-6) & self._in_circle
        out = np.zeros_like(sample_acc)
        out[valid] = sample_acc[valid] / weight_acc[valid][..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def render(self, data, renderer, scene_option=None):
        """Render the 5 tile cameras and resample them through the LUT into one fisheye image."""
        with self._seamless_headlight():
            tiles = self._render_tiles(data, renderer, scene_option)
        return self._resample_gpu(tiles) if torch.cuda.is_available() else self._resample_cpu(tiles)
