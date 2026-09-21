import json
import os
import subprocess
import sys

_EGL_BY_CUDA: dict[int, int] = {}


def cuda_to_egl(gpu_ids):
    """Map nvidia-smi CUDA indices to EGL device IDs, in a subprocess so CUDA is not yet initialised."""
    if not _EGL_BY_CUDA:
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        out = subprocess.check_output(
            [
                sys.executable,
                "-c",
                "from ctypes import *;import ctypes.util,json;"
                "egl=CDLL(ctypes.util.find_library('EGL'));"
                "egl.eglGetProcAddress.restype=c_void_p;"
                "egl.eglGetProcAddress.argtypes=[c_char_p];"
                "qd=CFUNCTYPE(c_bool,c_int,POINTER(c_void_p),POINTER(c_int))(egl.eglGetProcAddress(b'eglQueryDevicesEXT'));"
                "qa=CFUNCTYPE(c_bool,c_void_p,c_int,POINTER(c_long))(egl.eglGetProcAddress(b'eglQueryDeviceAttribEXT'));"
                "ds=(c_void_p*16)();n=c_int();qd(16,ds,byref(n));"
                "m={};"
                "[m.__setitem__(i,(lambda c:(qa(ds[i],0x323A,byref(c)),c.value)[1])(c_long(-1))) for i in range(n.value)];"
                "print(json.dumps(m))",
            ],
            env=env,
            text=True,
        )
        _EGL_BY_CUDA.update({v: int(k) for k, v in json.loads(out.strip()).items()})
    return [_EGL_BY_CUDA[g] for g in gpu_ids]


def set_egl_device(cuda_gpu: int) -> int:
    """Resolve a CUDA index to its EGL device and set MuJoCo's render env vars; call before CUDA_VISIBLE_DEVICES."""
    egl_id = cuda_to_egl([int(cuda_gpu)])[0]
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl_id)
    os.environ["EGL_DEVICE_ID"] = str(egl_id)
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    return egl_id
