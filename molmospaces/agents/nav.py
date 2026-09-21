"""Grid A* over the scene occupancy map, used to plan the base path."""
from __future__ import annotations

import heapq

import numpy as np


_ASTAR_COARSE_CACHE: dict = {}


def _coarsen_and_dist(occ, downscale):
    """Cached per (occ, downscale) — outputs depend only on these and are constant within a scene."""
    from collections import deque
    key = (occ.tobytes(), downscale, occ.shape)
    cached = _ASTAR_COARSE_CACHE.get(key)
    if cached is not None:
        return cached
    if downscale <= 1:
        coarse = occ.copy()
    else:
        d = downscale
        H, W = occ.shape
        padded = np.pad(occ, ((0, (-H) % d), (0, (-W) % d)))
        coarse = padded.reshape(padded.shape[0] // d, d, padded.shape[1] // d, d).min(1).min(-1)
    h, w = coarse.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    q = deque()
    for r in range(h):
        for c in range(w):
            if not coarse[r, c]:
                dist[r, c] = 0
                q.append((r, c))
    while q:
        r, c = q.popleft()
        b = dist[r, c]
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > b + 1:
                dist[nr, nc] = b + 1
                q.append((nr, nc))
    if len(_ASTAR_COARSE_CACHE) > 32:
        _ASTAR_COARSE_CACHE.clear()
    _ASTAR_COARSE_CACHE[key] = (coarse, dist)
    return coarse, dist


def _astar(occ, start_rc, goal_rc, downscale=4, wall_radius=10, wall_gain=6, wall_exp=2):
    """A* over the coarsened occupancy grid with a wall-proximity penalty; returns fine-grid (row, col) waypoints."""
    from collections import deque
    coarse, dist = _coarsen_and_dist(occ, downscale)
    h, w, D = coarse.shape[0], coarse.shape[1], downscale

    def nearest(o, rc):
        """Closest free cell to rc via BFS (handles start/goal inside obstacles)."""
        sr, sc = max(0, min(int(rc[0]), h-1)), max(0, min(int(rc[1]), w-1))
        if o[sr, sc]: return sr, sc
        q, v = deque([(sr, sc)]), {(sr, sc)}
        while q:
            r, c = q.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<h and 0<=nc<w and (nr,nc) not in v:
                    v.add((nr,nc))
                    if o[nr,nc]: return nr, nc
                    q.append((nr,nc))
        return None

    sf = nearest(coarse, (int(start_rc[0]//D), int(start_rc[1]//D)))
    gf = nearest(coarse, (int(goal_rc[0]//D), int(goal_rc[1]//D)))
    if sf is None or gf is None: return []
    sr, sc = sf; gr, gc = gf
    if (sr,sc)==(gr,gc): return [(gr*D+D//2, gc*D+D//2)]

    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    costs = [1.0,1.0,1.0,1.0,1.414,1.414,1.414,1.414]
    def wp(r,c):
        """Wall-proximity penalty for a cell (grows as distance-to-obstacle shrinks)."""
        d = dist[r,c]
        return 100.0 if d<=0 else (0.0 if d>=wall_radius else wall_gain*(1-max(d,1e-3)/wall_radius)**wall_exp)

    open_set = [(((sr-gr)**2+(sc-gc)**2)**.5, 0.0, sr, sc)]
    gs, cf, cl = {(sr,sc):0.0}, {}, set()
    while open_set:
        _, g, r, c = heapq.heappop(open_set)
        if (r,c) in cl: continue
        cl.add((r,c))
        if (r,c)==(gr,gc):
            path = [(r,c)]
            while (r,c) in cf: r,c=cf[(r,c)]; path.append((r,c))
            path.reverse()
            return [(p*D+D//2, q*D+D//2) for p,q in path]
        for (dr,dc),cost in zip(dirs,costs):
            nr,nc = r+dr,c+dc
            if 0<=nr<h and 0<=nc<w and coarse[nr,nc] and (nr,nc) not in cl:
                ng = g+cost+wp(nr,nc)
                if ng < gs.get((nr,nc),float("inf")):
                    gs[(nr,nc)]=ng; cf[(nr,nc)]=(r,c)
                    heapq.heappush(open_set, (ng+((nr-gr)**2+(nc-gc)**2)**.5, ng, nr, nc))
    return []


def _simplify_path(path, occ, clearance=6):
    """Greedily drop waypoints whose straight-line shortcut keeps `clearance` cells of margin."""
    if len(path) <= 2: return path
    def free(a, b):
        """True if the straight segment a->b stays `clearance` cells away from obstacles."""
        n = int(max(abs(b[0]-a[0]), abs(b[1]-a[1])))+1
        for t in np.linspace(0,1,n):
            r,c = int(round((1-t)*a[0]+t*b[0])), int(round((1-t)*a[1]+t*b[1]))
            for dr in range(-clearance, clearance+1):
                for dc in range(-clearance, clearance+1):
                    if not occ[max(0,min(r+dr,occ.shape[0]-1)), max(0,min(c+dc,occ.shape[1]-1))]: return False
        return True
    s = [path[0]]; i = 0
    while i < len(path)-1:
        j = len(path)-1
        while j > i+1 and not free(path[i], path[j]): j -= 1
        s.append(path[j]); i = j
    return s
