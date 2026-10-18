"""Dataset assembly: read the LeRobot columns, filter episodes, and window them for training."""
import hashlib
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from absl import flags
from tqdm import tqdm

from utils.action_transforms import DELTA_DIMS, anchor_from_cols, target_point_xyv

FLAGS = flags.FLAGS

PRIVILEGED_OBS = {
    'observation.base_position',
    'observation.base_yaw',
    'observation.base_rpy',
    'observation.base_velocity',
    'observation.base_angular_velocity',
    'observation.joint_pos',
    'observation.right_hand_pose',
    'observation.target_object_pose',
    'observation.target_point',
}


def _resolve_stats(probe, keys):
    """Per-key min/max stats from dataset metadata, computing missing keys with a full pass."""
    meta_stats = getattr(probe.meta, 'stats', None) or {}
    out = {}
    missing = []
    for k in keys:
        s = meta_stats.get(k)
        if s and 'min' in s and 'max' in s:
            out[k] = {'min': np.asarray(s['min'], dtype=np.float32).reshape(-1),
                      'max': np.asarray(s['max'], dtype=np.float32).reshape(-1)}
        else:
            missing.append(k)
    if missing:
        running = {k: [None, None] for k in missing}
        for i in range(len(probe)):
            sample = probe[i]
            for k in missing:
                v = np.asarray(sample[k], dtype=np.float32).reshape(-1)
                if running[k][0] is None:
                    running[k] = [v.copy(), v.copy()]
                else:
                    np.minimum(running[k][0], v, out=running[k][0])
                    np.maximum(running[k][1], v, out=running[k][1])
        for k in missing:
            out[k] = {'min': running[k][0], 'max': running[k][1]}
    return out


def _resize_chw_batch_uint8(arr, size_hw):
    """Antialiased bilinear resize of a uint8 NCHW batch (no-op when already sized)."""
    if size_hw is None or tuple(arr.shape[-2:]) == tuple(size_hw):
        return arr
    import torch.nn.functional as F
    x = torch.from_numpy(arr).float()
    x = F.interpolate(x, size=tuple(size_hw), mode='bilinear', align_corners=False, antialias=True)
    return x.round().clamp(0, 255).to(torch.uint8).numpy()


def _decode_video_to_disk(mp4_paths, n_frames, out_path, resize_hw=None):
    """Decode mp4s to a uint8 memmap on disk with torchcodec (the video cache builder)."""
    from torchcodec.decoders import VideoDecoder
    if isinstance(mp4_paths, (str, Path)):
        mp4_paths = [mp4_paths]
    t0 = time.time()
    decoders = []
    per_file_counts = []
    for path in mp4_paths:
        dec = VideoDecoder(str(path))
        m = dec.metadata
        per_file = int(getattr(m, "num_frames", None) or m.num_frames_from_header or 0)
        if per_file <= 0:
            raise RuntimeError(f"Could not determine frame count for {path}")
        decoders.append(dec)
        per_file_counts.append(per_file)
    total = sum(per_file_counts)

    if resize_hw is None:
        sample = decoders[0].get_frames_at(indices=[0]).data.numpy()
        if sample.dtype != np.uint8:
            sample = (sample * 255).clip(0, 255).astype(np.uint8)
        H, W = sample.shape[-2], sample.shape[-1]
    else:
        H, W = resize_hw
    shape = (total, 3, H, W)

    # Atomic .partial→.npy rename so a crashed decode doesn't leave a half-filled cache.
    out_path = Path(out_path)
    partial = out_path.with_suffix(out_path.suffix + '.partial')
    out = np.lib.format.open_memmap(str(partial), mode='w+', dtype=np.uint8, shape=shape)

    batch_size = 1024
    cursor = 0
    for dec, path, per_file in zip(decoders, mp4_paths, per_file_counts):
        for start in tqdm(range(0, per_file, batch_size),
                          desc=f"decoding {Path(path).name}", unit="batch"):
            end = min(start + batch_size, per_file)
            batch = dec.get_frames_at(indices=list(range(start, end)))
            arr = batch.data.numpy()
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            arr = _resize_chw_batch_uint8(arr, resize_hw)
            out[cursor:cursor + arr.shape[0]] = arr
            cursor += arr.shape[0]
    out.flush()
    del out
    os.replace(partial, out_path)

    if total != n_frames:
        print(f"    [warn] decoded {total} frames across {len(mp4_paths)} file(s), expected {n_frames}")
    print(f"    decoded {total} frames in {time.time()-t0:.1f}s, shape={shape}, dtype=uint8, size={total*3*H*W/1e9:.2f}GB")


def _dl_worker_init(_worker_id):
    """Each DataLoader worker only decodes video; it has no use for torch's big intra-op thread pool (inherited = num_cores)."""
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    os.environ["OMP_NUM_THREADS"] = "1"


def _video_cum_counts_from_meta(probe_root, video_keys, video_mp4_paths):
    """Per-file cumulative frame counts for each video key, derived from meta/episodes/*.parquet (sum of episode `length` grouped by the episode's video chunk/file index)."""
    import glob as _glob
    ep_files = sorted(_glob.glob(str(Path(probe_root) / "meta" / "episodes" / "**" / "*.parquet"),
                                 recursive=True))
    if not ep_files:
        return {}
    need = ['length']
    for vk in video_keys:
        need += [f'videos/{vk}/chunk_index', f'videos/{vk}/file_index']
    df = pd.concat([pd.read_parquet(f, columns=need) for f in ep_files], ignore_index=True)
    cre = re.compile(r'chunk-(\d+)')
    fre = re.compile(r'file-(\d+)')
    out: dict[str, np.ndarray] = {}
    for vk in video_keys:
        counts = df.groupby([f'videos/{vk}/chunk_index', f'videos/{vk}/file_index'])['length'].sum()
        try:
            per_file = []
            for p in video_mp4_paths[vk]:
                c = int(cre.search(p).group(1))
                f = int(fre.search(p).group(1))
                per_file.append(int(counts.loc[(c, f)]))
        except (KeyError, AttributeError) as e:
            print(f"  [video] could not resolve frame counts for {vk} from metadata ({e!r}); "
                  f"falling back to per-worker mp4 scan for this key")
            continue
        out[vk] = np.concatenate([[0], np.cumsum(per_file)]).astype(np.int64)
    return out


class _WindowedDataset(torch.utils.data.Dataset):
    def __init__(self, probe, cols, state_keys, video_keys, t_obs, t_pred,
                 video_cache=None, video_mp4_paths=None, image_size_hw=None,
                 debug_every=0, use_action_delta=True, obj_emb_unique=None,
                 video_cum_counts=None):
        """Frame-window dataset: obs history + action chunk per index over packed columns."""
        self.probe = probe
        # dino.txt: (n_unique, dim) table; cols['object_name_idx'] indexes it per frame
        self.obj_emb_unique = obj_emb_unique
        self.cols = cols
        self.state_keys = state_keys
        self.video_keys = video_keys
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.use_action_delta = bool(use_action_delta)
        self.video_cache = video_cache or {}
        # On-the-fly decode path: for each video key, list of mp4 paths and
        # cumulative frame counts so global_idx → (file_idx, within_file_idx).
        # Decoders themselves are lazily created per-worker in __getitem__.
        self.video_mp4_paths: dict[str, list[str]] = video_mp4_paths or {}
        # Precomputed once in the main process from metadata (see
        # _video_cum_counts_from_meta) and inherited by every worker via fork,
        # so the per-worker "open every mp4 to count frames" scan is skipped.
        self._mp4_cum_counts: dict[str, np.ndarray] = dict(video_cum_counts) if video_cum_counts else {}
        self._decoder_cache: dict[tuple[int, str, int], "VideoDecoder"] = {}
        # Bound the per-worker open-decoder set. With random shuffle each worker
        # would otherwise open ~all 1042 files/key and keep them forever; across
        # many DDP-rank workers that exhausts memory (silent OOM kill). Reopen is
        # ~4ms (approximate seek), so a small LRU cap is essentially free.
        self._decoder_cache_cap = 128
        self._image_size_hw = image_size_hw  # (H, W) — resize target if mp4 native is different
        # Debug timing: per-worker rolling stats reset every `debug_every` calls.
        self._debug_every = int(debug_every)
        self._dbg = {
            "calls": 0, "cold_opens": 0,
            "decode_s": 0.0, "resize_s": 0.0, "getitem_s": 0.0,
            "frames_decoded": 0,
        }
        for required in ('observation.base_height', 'observation.upper_joint_pos'):
            if required not in cols:
                raise ValueError(f"delta-action transform requires {required} in cols")
        self.ep = cols['episode_index']
        self.n = len(self.ep)
        # Episodes are contiguous blocks of equal episode_index. The old code
        # found block bounds with a ~25M-iteration Python loop (minutes, and it
        # runs in every DDP rank). Vectorized boundary detection is ~instant.
        ep_arr = np.asarray(self.ep)
        change = np.flatnonzero(ep_arr[1:] != ep_arr[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [self.n]))          # exclusive
        lengths = ends - starts
        self.ep_start = np.repeat(starts, lengths).astype(np.int64)
        self.ep_end = np.repeat(ends - 1, lengths).astype(np.int64)

    def __len__(self):
        """Number of sampleable frames."""
        return self.n

    def _decode_video_indices(self, vk: str, global_indices: np.ndarray) -> np.ndarray:
        """On-the-fly decode of video frames for a given video key, mapping global dataset indices to (mp4_file, within_file_idx) and using a per-worker torchcodec VideoDecoder cache."""
        import time as _t
        from torchcodec.decoders import VideoDecoder
        paths = self.video_mp4_paths[vk]
        # Lazy: build cum_counts on first call per (worker, key)
        if vk not in self._mp4_cum_counts:
            per_file = []
            for p in paths:
                tmp_dec = VideoDecoder(str(p))
                m = tmp_dec.metadata
                n = int(getattr(m, "num_frames", None) or m.num_frames_from_header or 0)
                per_file.append(n)
                del tmp_dec
            self._mp4_cum_counts[vk] = np.concatenate([[0], np.cumsum(per_file)])
        cum = self._mp4_cum_counts[vk]
        # Group global indices by mp4 file
        file_ids = np.searchsorted(cum, global_indices, side='right') - 1
        out_frames: list[np.ndarray] = [None] * len(global_indices)  # type: ignore
        worker_id = (torch.utils.data.get_worker_info().id
                     if torch.utils.data.get_worker_info() is not None else -1)
        for fi in np.unique(file_ids):
            mask = file_ids == fi
            local_idx = (global_indices[mask] - cum[fi]).tolist()
            key = (worker_id, vk, int(fi))
            dec = self._decoder_cache.get(key)
            if dec is None:
                self._dbg["cold_opens"] += 1
                # One ffmpeg thread per decoder: more just oversubscribes the CPU
                # across DataLoader workers. Approximate seek skips a ~200ms index
                # scan per open and is bit-exact at a constant 10fps.
                dec = VideoDecoder(str(paths[int(fi)]), num_ffmpeg_threads=1,
                                   seek_mode="approximate")
                if len(self._decoder_cache) >= self._decoder_cache_cap:
                    old_key = next(iter(self._decoder_cache))
                    old_dec = self._decoder_cache.pop(old_key)
                    del old_dec
                self._decoder_cache[key] = dec
            else:
                # LRU: mark as most-recently-used so the cap evicts cold files.
                self._decoder_cache[key] = self._decoder_cache.pop(key)
            t1 = _t.perf_counter()
            batch = dec.get_frames_at(indices=local_idx).data.numpy()
            self._dbg["decode_s"] += _t.perf_counter() - t1
            self._dbg["frames_decoded"] += batch.shape[0]
            if batch.dtype != np.uint8:
                batch = (batch * 255).clip(0, 255).astype(np.uint8)
            t2 = _t.perf_counter()
            batch = _resize_chw_batch_uint8(batch, self._image_size_hw)
            self._dbg["resize_s"] += _t.perf_counter() - t2
            for j, src_idx in zip(np.flatnonzero(mask), batch):
                out_frames[int(j)] = src_idx
        return np.stack(out_frames, axis=0)

    def __getitem__(self, idx):
        """Build one sample: clamped obs-history window, action chunk, images from cache or decode."""
        import time as _t
        t0 = _t.perf_counter()
        s, e = self.ep_start[idx], self.ep_end[idx]
        obs_idx = np.clip(np.arange(idx - self.t_obs + 1, idx + 1), s, e)
        act_idx = np.clip(np.arange(idx, idx + self.t_pred), s, e)
        sample = {k: torch.from_numpy(self.cols[k][obs_idx]) for k in self.state_keys}
        actions = self.cols['action'][act_idx]
        if self.use_action_delta:
            actions[:, DELTA_DIMS] -= anchor_from_cols(self.cols, idx)
        sample['action'] = torch.from_numpy(actions)
        if 'object_name_idx' in self.cols and self.obj_emb_unique is not None:
            ui = int(self.cols['object_name_idx'][idx])
            sample['object_name_embedding'] = torch.from_numpy(self.obj_emb_unique[ui].copy())
        # Video lookup must use ORIGINAL frame indices: filters (task_filter,
        # bad_episodes_csv) drop rows from cols, but the mp4 stream / probe are
        # unfiltered. video_cache is masked alongside cols, so it stays positional.
        vid_idx = self.cols['_video_idx'][obs_idx] if '_video_idx' in self.cols else obs_idx
        for k in self.video_keys:
            if k in self.video_cache:
                arr = self.video_cache[k][obs_idx]
                sample[k] = torch.from_numpy(arr.copy())
            elif k in self.video_mp4_paths:
                arr = self._decode_video_indices(k, vid_idx)
                sample[k] = torch.from_numpy(arr)
            else:
                sample[k] = torch.stack([self.probe[int(i)][k] for i in vid_idx])
        self._dbg["getitem_s"] += _t.perf_counter() - t0
        self._dbg["calls"] += 1
        if self._debug_every and self._dbg["calls"] % self._debug_every == 0:
            wid = (torch.utils.data.get_worker_info().id
                   if torch.utils.data.get_worker_info() is not None else -1)
            d = self._dbg
            n_open = len(self._decoder_cache)
            avg_get = d["getitem_s"] / d["calls"] * 1000
            avg_dec_per_frame = d["decode_s"] / max(d["frames_decoded"], 1) * 1000
            avg_resize_per_frame = d["resize_s"] / max(d["frames_decoded"], 1) * 1000
            print(f"[worker {wid}] calls={d['calls']} "
                  f"getitem={avg_get:.1f}ms/call  "
                  f"decode={avg_dec_per_frame:.2f}ms/frame  "
                  f"resize={avg_resize_per_frame:.2f}ms/frame  "
                  f"frames={d['frames_decoded']} "
                  f"open_decoders={n_open} cold_opens={d['cold_opens']}",
                  flush=True)
            # Reset rolling window so we see recent behavior, not lifetime avg
            for k_ in ("calls", "decode_s", "resize_s", "getitem_s", "frames_decoded"):
                d[k_] = 0 if isinstance(d[k_], int) else 0.0
        return sample


def _filter_episodes(cols, probe, parquet_paths, video_cache, is_rank_zero):
    """Drop blacklisted episodes, off-task frames, and anything past --max_episodes."""
    if FLAGS.bad_episodes_csv:
        bad_path = Path(FLAGS.bad_episodes_csv)
        if not bad_path.exists():
            raise FileNotFoundError(f"--bad_episodes_csv path does not exist: {bad_path}")
            bad_df = pd.read_csv(bad_path)
        if 'episode_index' not in bad_df.columns:
            raise ValueError(f"{bad_path} missing required 'episode_index' column")
        bad_set = set(int(x) for x in bad_df['episode_index'].tolist())
        ep = cols['episode_index']
        n_before_frames = len(ep)
        n_before_eps = len(np.unique(ep))
        keep_mask = ~np.isin(ep, np.fromiter(bad_set, dtype=ep.dtype))
        for k in list(cols.keys()):
            cols[k] = cols[k][keep_mask]
        for k in list(video_cache.keys()):
            video_cache[k] = video_cache[k][keep_mask]
        n_after_frames = len(cols['episode_index'])
        n_after_eps = len(np.unique(cols['episode_index']))
        if is_rank_zero:
            print(f"  filtered bad episodes from {bad_path}: "
                  f"dropped {n_before_eps - n_after_eps} episodes / "
                  f"{n_before_frames - n_after_frames} frames "
                  f"({100.0 * (n_before_frames - n_after_frames) / max(n_before_frames, 1):.1f}%); "
                  f"remaining: {n_after_eps} episodes / {n_after_frames} frames")

    if FLAGS.task_filter:
            # tasks.parquet: prompt string is the row index, task_index is the column.
        tdf = pq.read_table(str(Path(probe.root) / 'meta' / 'tasks.parquet')).to_pandas()
        idx_to_task = {int(r['task_index']): str(name) for name, r in tdf.iterrows()}
        ti = pa.concat_tables([pq.read_table(str(p), columns=['task_index'])
                               for p in parquet_paths]).column('task_index').to_numpy()

        ti = ti[cols['_video_idx']]
        pat = re.compile(FLAGS.task_filter)
        keep_idx = {i for i, t in idx_to_task.items() if pat.search(str(t))}
        if is_rank_zero:
            print(f"  task_filter {FLAGS.task_filter!r} -> keeping task_index {sorted(keep_idx)} "
                  f"({[idx_to_task[i] for i in sorted(keep_idx)]})")
        keep_mask = np.isin(ti, np.fromiter(keep_idx, dtype=ti.dtype) if keep_idx
                            else np.array([], dtype=ti.dtype))
        ep = cols['episode_index']
        n_before_frames = len(ep); n_before_eps = len(np.unique(ep))
        for k in list(cols.keys()):
            cols[k] = cols[k][keep_mask]
        for k in list(video_cache.keys()):
            video_cache[k] = video_cache[k][keep_mask]
        n_after_frames = len(cols['episode_index']); n_after_eps = len(np.unique(cols['episode_index']))
        if is_rank_zero:
            print(f"  task_filter dropped {n_before_eps - n_after_eps} episodes / "
                  f"{n_before_frames - n_after_frames} frames; "
                  f"remaining: {n_after_eps} episodes / {n_after_frames} frames")
        if n_after_frames == 0:
            raise ValueError(f"--task_filter={FLAGS.task_filter!r} matched no episodes")


    if FLAGS.max_episodes > 0:
        ep = cols['episode_index']
        unique_eps = np.unique(ep)
        if len(unique_eps) > FLAGS.max_episodes:
            last_ep = unique_eps[FLAGS.max_episodes - 1]
            cutoff = int(np.searchsorted(ep, last_ep, side='right'))
            for k in list(cols.keys()):
                cols[k] = cols[k][:cutoff]
            for k in list(video_cache.keys()):
                video_cache[k] = video_cache[k][:cutoff]
            if is_rank_zero:
                print(f"  capping training data: {FLAGS.max_episodes} episodes / {cutoff} frames "
                      f"(out of {len(unique_eps)} episodes / {len(ep)} frames)")
        elif is_rank_zero:
            print(f"  --max_episodes={FLAGS.max_episodes} >= dataset size ({len(unique_eps)}); using all")
    return cols


def _load_state_columns(probe, state_keys, config, data_root, dataset_stats, is_rank_zero):
    """Read every state/action column out of the dataset parquets into one dict of arrays."""
    if is_rank_zero:
        print("Preloading state columns into RAM...")
    t0 = time.time()
    parquet_paths = sorted((Path(probe.root) / 'data').rglob('file-*.parquet'))
    if not parquet_paths:
        raise RuntimeError(f"No parquet files found under {probe.root}/data")
    keys_to_load = state_keys + ['action', 'episode_index']
    encoder_name = str(config.get('encoder', ''))
    encoder_uses_text = 'dinotxt' in encoder_name
    schema_names = set(pq.read_schema(str(parquet_paths[0])).names)
    load_object_name = encoder_uses_text and 'object_name' in schema_names
    if load_object_name:
        keys_to_load.append('object_name')
    table = pa.concat_tables([pq.read_table(str(p), columns=keys_to_load) for p in parquet_paths])
    cols = {}
    obj_emb_unique = None  # dino.txt: (n_unique, dim) table; per-frame int index goes in cols
    iterator = tqdm(keys_to_load, desc="state columns", unit="col") if is_rank_zero else keys_to_load
    for k in iterator:
        col = table.column(k).combine_chunks()
        t = col.type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            # Only reachable for object_name under the dino.txt encoder. Pre-embed
            # via its own text tower (it is not a transformers model), matching
            # DINOv3DinoTxtEncoder._encode_text so train and eval agree.
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            m, tk = torch.hub.load('facebookresearch/dinov3',
                                   'dinov3_vitl16_dinotxt_tet1280d20h24l', trust_repo=True)
            m = m.to(device).eval()
            names = col.to_pylist(); unique = sorted(set(names))
            if is_rank_zero:
                print(f"  pre-embedding {k}: {len(unique)} unique strings via dino.txt text tower...")
            # Disk cache keyed by (model, exact unique-string set) so reruns
            # skip re-encoding entirely. First run: BATCHED encode (the old
            # per-string loop was ~100x slower), with a per-string fallback.
            ck = hashlib.md5(('dinov3_vitl16_dinotxt_tet1280d20h24l|'
                              + '\n'.join(unique)).encode()).hexdigest()
            cache_dir = Path(data_root) if data_root else Path(probe.root)
            cache_path = cache_dir / 'text_embed_cache' / f'objname_dinotxt_{ck}.npz'
            if cache_path.exists():
                z = np.load(cache_path, allow_pickle=True)
                uemb = z['emb']
                if is_rank_zero:
                    print(f"    loaded cached object_name embeddings from {cache_path}")
            else:
                with torch.no_grad():
                    try:
                        parts = []
                        for i in range(0, len(unique), 256):
                            f = m.encode_text(tk.tokenize(unique[i:i + 256]).to(device))
                            parts.append(f[:, f.shape[1] // 2:].float().cpu().numpy())
                        uemb = np.concatenate(parts, axis=0)
                    except Exception as e:
                        if is_rank_zero:
                            print(f"    batched encode failed ({e}); falling back to per-string")
                        uemb = np.stack([
                            m.encode_text(tk.tokenize([s]).to(device))[0][None][:, :].float().cpu().numpy()[0]
                            for s in unique])
                        uemb = uemb[:, uemb.shape[1] // 2:]
                if is_rank_zero:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(cache_path, names=np.array(unique, dtype=object),
                             emb=uemb.astype(np.float32))
                    print(f"    encoded {len(unique)} strings (batched) -> cached at {cache_path}")
            del m
            # One row per unique name plus a per-frame index; a dense
            # (N, dim) array is ~51 GB at 25M frames. uemb follows the sorted
            # `unique` order, so searchsorted maps each frame to its row.
            unique_arr = np.asarray(unique, dtype=object)
            inverse = np.searchsorted(unique_arr, np.asarray(names, dtype=object))
            assert len(unique) == len(uemb), f"unique mismatch {len(unique)} vs {len(uemb)}"
            obj_emb_unique = np.ascontiguousarray(uemb, dtype=np.float32)
            cols['object_name_idx'] = inverse.astype(np.int32)   # per-frame; rides every keep_mask
            if is_rank_zero:
                print(f"    {len(uemb)} unique embeddings ({obj_emb_unique.nbytes/1e6:.1f} MB) + "
                      f"per-frame int32 index ({cols['object_name_idx'].nbytes/1e6:.0f} MB) "
                      f"[was {len(names)*obj_emb_unique.shape[1]*4/1e9:.0f} GB dense]")
            continue
        if pa.types.is_fixed_size_list(t):
            arr = col.values.to_numpy(zero_copy_only=False).reshape(-1, t.list_size)
        elif pa.types.is_list(t) or pa.types.is_large_list(t):
            inner = len(col[0])
            arr = col.values.to_numpy(zero_copy_only=False).reshape(-1, inner)
        else:
            arr = col.to_numpy(zero_copy_only=False)
        cols[k] = np.array(arr, copy=not arr.flags.writeable)
    for k in state_keys:
        if cols[k].ndim == 1:
            cols[k] = cols[k][:, None]
    # Canonicalize quaternions (w >= 0) for pose fields shape (N, 7) = [xyz, w, qxyz].
    for k in state_keys:
        if cols[k].ndim == 2 and cols[k].shape[-1] == 7:
            flip = cols[k][:, 3] < 0
            cols[k][flip, 3:] *= -1
            n_flipped = int(flip.sum())
            if is_rank_zero and n_flipped > 0:
                print(f"  canonicalized {k}: flipped {n_flipped}/{len(cols[k])} quaternions")
    # Point conditioning: 2D target_point -> [x, y, visible] (coords zeroed on the
    # (-1,-1) not-visible sentinel), with matching 3D normalization stats. Same
    # transform is applied to the eval obs in eval_callback so train/eval agree.
    if FLAGS.point_condition and 'observation.target_point' in cols:
        if cols['observation.target_point'].shape[-1] == 2:
            cols['observation.target_point'] = target_point_xyv(cols['observation.target_point'])
        tp3 = cols['observation.target_point']
        dataset_stats['observation.target_point'] = {
            'mean': tp3.mean(axis=0), 'std': np.clip(tp3.std(axis=0), 1e-6, None)}
        if is_rank_zero:
            print(f"  point_condition: target_point -> {tp3.shape} [x,y,visible], "
                  f"{100.0 * (1.0 - float(tp3[:, 2].mean())):.2f}% hidden")
    if is_rank_zero:
        print(f"  bulk read: {time.time()-t0:.1f}s")
    return cols, obj_emb_unique, parquet_paths


def _build_video_cache(probe, cols, video_keys, image_size, image_size_w, is_rank_zero):
    """Optionally pre-decode videos into a frame cache keyed by dataset + image size."""
    def _cache_safe(k):
        """Sanitize a key for use in cache filenames."""
        return k.replace('.', '_').replace('/', '_')

    cache_tag = hashlib.sha1(
        f"{probe.root}|{len(cols['action'])}|{','.join(sorted(video_keys))}|image_size={image_size}x{image_size_w}".encode()
    ).hexdigest()[:12]
    cache_dir = Path(probe.root) / '.train_cache' / cache_tag
    ready_file = cache_dir / 'ready'

    video_cache = {}
    if video_keys and not FLAGS.no_video_cache:
        if is_rank_zero:
            if not ready_file.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)
                n_frames = len(cols['action'])
                for vk in video_keys:
                    out_path = cache_dir / f"video__{_cache_safe(vk)}.npy"
                    if out_path.exists():
                        continue
                    print(f"  pre-decoding {vk} into cache {cache_dir}...")
                    mp4_paths = sorted((Path(probe.root) / "videos" / vk).rglob("file-*.mp4"))
                    if not mp4_paths:
                        raise RuntimeError(f"No MP4 files found for {vk} under {probe.root}/videos/{vk}")
                    _decode_video_to_disk(mp4_paths, n_frames, out_path, resize_hw=(image_size, image_size_w))
                tmp = cache_dir / 'ready.tmp'
                tmp.write_text('ok')
                os.replace(tmp, ready_file)
            else:
                print(f"  reusing cached videos at {cache_dir}")
        else:
            while not ready_file.exists():
                time.sleep(2)
        for vk in video_keys:
            video_cache[vk] = np.load(cache_dir / f"video__{_cache_safe(vk)}.npy", mmap_mode='r')
    elif video_keys and FLAGS.no_video_cache and is_rank_zero:
        print(f"  --no_video_cache: skipping precompute, will decode mp4 frames on-the-fly per batch")

    # Pre-filter frame index: video lookup is against the full concatenated
    # stream, so positional indices would pair states with the wrong images once
    # any filter drops rows. Rides through every keep_mask.
    cols['_video_idx'] = np.arange(len(cols['episode_index']), dtype=np.int64)
    return video_cache


def _action_stats(probe, cols, T_pred, dataset_stats, train_dataset, is_rank_zero):
    """Mean/std over the action chunks, cached on disk next to the dataset."""
    stats_cache_dir = Path(probe.root) / '.train_cache'
    stats_cache_dir.mkdir(parents=True, exist_ok=True)
    stats_key = hashlib.sha1(
        f"{probe.root}|{len(cols['action'])}|{T_pred}|"
        f"{'delta' if train_dataset.use_action_delta else 'abs'}-action-v2-zscore".encode()
    ).hexdigest()[:12]
    stats_cache_path = stats_cache_dir / f"action_stats_{stats_key}.npz"

    if not stats_cache_path.exists():
        if is_rank_zero:
            print(f"  computing delta-action mean/std (will cache to {stats_cache_path})...")
            t_stats = time.time()
            action_dim = int(cols['action'].shape[-1])
            N = len(train_dataset)
            ep_start = train_dataset.ep_start
            ep_end = train_dataset.ep_end
            CHUNK = 16384
            sum_ = np.zeros(action_dim, dtype=np.float64)
            sq_sum = np.zeros(action_dim, dtype=np.float64)
            n = 0
            base_h = np.asarray(cols['observation.base_height'], dtype=np.float32).reshape(N, -1)
            upper = np.asarray(cols['observation.upper_joint_pos'], dtype=np.float32)
            anchor_all = np.concatenate([base_h, upper[:, 0:3], upper[:, 3:10]], axis=1)
            t_pred_range = np.arange(T_pred)
            for start in tqdm(range(0, N, CHUNK), desc="action stats", unit="chunk"):
                end = min(start + CHUNK, N)
                idx = np.arange(start, end)
                act_idx = np.clip(
                    idx[:, None] + t_pred_range[None, :],
                    ep_start[idx, None],
                    ep_end[idx, None],
                )
                a = cols['action'][act_idx].astype(np.float64, copy=True)
                if train_dataset.use_action_delta:
                    a[:, :, DELTA_DIMS] -= anchor_all[idx, None, :]
                sum_ += a.sum(axis=(0, 1))
                sq_sum += (a * a).sum(axis=(0, 1))
                n += a.shape[0] * a.shape[1]
            amean = sum_ / max(n, 1)
            avar = np.maximum(sq_sum / max(n, 1) - amean ** 2, 0.0)
            amean = amean.astype(np.float32)
            astd = np.sqrt(avar).astype(np.float32)
            tmp = stats_cache_path.with_suffix('.npz.tmp')
            with open(tmp, 'wb') as f:
                np.savez(f, mean=amean, std=astd)
            os.replace(tmp, stats_cache_path)
            print(f"  delta-action stats done in {time.time()-t_stats:.1f}s")
        else:
            while not stats_cache_path.exists():
                time.sleep(2)

    z = np.load(stats_cache_path)
    amean = z['mean'].astype(np.float32)
    astd = z['std'].astype(np.float32)
    dataset_stats['action'] = {'mean': amean, 'std': astd}
    if is_rank_zero:
        print(f"  reusing delta-action stats at {stats_cache_path}")
        print(f"    action mean: {amean}")
        print(f"    action std:  {astd}")
    return dataset_stats




def split_val_episodes(cols, video_cache, n_val, is_rank_zero, split_seed=0):
    """Hold out `n_val` whole episodes for validation.

    The split is by episode, never by frame: neighbouring frames of one episode
    are near-duplicates, so a frame-level split would leak the answer into the
    val set and report a loss far below the truth. `split_seed` is deliberately
    its own knob rather than `--seed`, so changing the training seed re-rolls the
    model and not the data split.

    Returns (train_cols, train_video_cache, val_cols, val_video_cache); the val
    pair is (None, None) when `n_val` is 0.
    """
    if n_val <= 0:
        return cols, video_cache, None, None

    ep = cols['episode_index']
    unique_eps = np.unique(ep)
    if n_val >= len(unique_eps):
        raise ValueError(f"--val_episodes={n_val} leaves no training data "
                         f"(dataset has {len(unique_eps)} episodes)")
    rng = np.random.default_rng(split_seed)
    val_eps = np.sort(rng.choice(unique_eps, size=n_val, replace=False))
    val_mask = np.isin(ep, val_eps)

    train_cols = {k: v[~val_mask] for k, v in cols.items()}
    val_cols = {k: v[val_mask] for k, v in cols.items()}
    train_cache = {k: v[~val_mask] for k, v in video_cache.items()}
    val_cache = {k: v[val_mask] for k, v in video_cache.items()}
    if is_rank_zero:
        print(f"  val split (seed {split_seed}): {len(unique_eps) - n_val} train episodes / "
              f"{len(train_cols['episode_index'])} frames, "
              f"{n_val} val episodes / {len(val_cols['episode_index'])} frames")
    return train_cols, train_cache, val_cols, val_cache
