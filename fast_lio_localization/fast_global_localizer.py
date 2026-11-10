#!/usr/bin/env python3
"""
Fast multi-hypothesis GICP global localization (localization_method:=fast_gicp).

Chosen with the localization benchmark (unitree_nav_stack, Localize tab): on a real dataset it found
the correct pose in 97.6 % of cases (initial estimates up to 8 m and 180 deg off) against 22 % for the
point-to-point ICP with yaw retries, in about 40 ms single-threaded on a PC.

    loc = FastGlobalLocalizer(map_xyz)          # once: precomputes grids and voxel maps (a few seconds)
    T_map_body, score, info = loc.localize(cloud_body, guess_map_body)

1. Hypotheses: every heading (yaw_step_deg) at every position of a grid (step) within `radius` of
   the guess. Height, roll and pitch are taken from the guess, so they must be roughly right
   (within ~4 deg tilt and ~0.5 m height).
2. All hypotheses are scored at once with a precomputed grid of closeness to the map
   (exp(-d^2 / 2 sigma^2)): a table lookup per point instead of a nearest-neighbour search.
3. The best `top_k` hypotheses that are not near each other (nms_dist, nms_yaw) are refined with
   coarse VGICP, then the best `keep` with fine VGICP. The result with the highest score wins.

score = fraction of cloud points within 0.3 m of the map (as localization_monitor's fit).
Needs numpy, scipy and small_gicp (pip3 install small_gicp).
"""

import math
import time

import numpy as np
from scipy import ndimage
import small_gicp

DEFAULTS = {
    'radius': 6.0,            # m: search radius around the initial estimate
    'step': 1.0,              # m: position grid of the hypotheses (multiple of grid_res)
    'yaw_step_deg': 15.0,     # heading step of the hypotheses
    'grid_res': 0.25,         # m: cell size of the closeness grid
    'near': 1.0,              # m: closeness scale of the hypothesis score (sigma = near / 2)
    'map_voxel': 0.1,         # m: map downsampling before everything else
    'pre_voxel': 0.5,         # m: cloud downsampling for the hypothesis score
    'pre_points': 600,        # max cloud points for the hypothesis score
    'top_k': 8,               # hypotheses refined with coarse VGICP
    'nms_dist': 1.5,          # m: picked hypotheses must differ by this distance ...
    'nms_yaw_deg': 30.0,      # ... or this heading
    'keep': 2,                # coarse results refined with fine VGICP
    'stages': ((0.6, 2.0), (0.3, 0.8)),   # (voxel m, max correspondence m): coarse, fine
    'vgicp_leaf': 2.0,        # VGICP voxel = stage voxel * this
    'max_iterations': 30,
    'num_threads': 1,
    'score_dist': 0.3,        # m: inlier distance of the final score
    'score_voxel': 0.2,
    'score_points': 3000,
    # whole-map search (localize_global, used at startup)
    'global_step': 1.0,               # m: position grid
    'global_top_k': 24,               # distinct hypotheses refined with coarse VGICP
    'global_keep': 3,                 # refined with fine VGICP
    'global_footprint_margin': 2.0,   # m: positions up to this far from map points (xy)
}


def _rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _voxel(points, size):
    return np.asarray(small_gicp.voxelgrid_sampling(points, size).points())[:, :3]


def _subsample(points, n):
    if len(points) <= n:
        return points
    return points[np.linspace(0, len(points) - 1, n).astype(np.int64)]


class FastGlobalLocalizer:
    def __init__(self, map_xyz, params=None):
        p = dict(DEFAULTS)
        p.update(params or {})
        self.p = p
        t0 = time.perf_counter()
        pts = _voxel(np.asarray(map_xyz, dtype=np.float64)[:, :3], p['map_voxel'])
        if len(pts) < 100:
            raise ValueError(f'map has only {len(pts)} points')
        res = p['grid_res']

        # Closeness grid, padded so that a guess up to `radius` off the map plus a shift of up to
        # `radius` stays inside: a hypothesis shift is then a constant offset of the flat index.
        pad_xy = 2.0 * p['radius'] + 1.0
        pad = np.array([pad_xy, pad_xy, 2.0 * p['near'] + 0.5])
        self.lo = pts.min(axis=0) - pad
        self.shape = (np.ceil((pts.max(axis=0) + pad - self.lo) / res).astype(np.int64) + 1)
        occ = np.ones(self.shape, dtype=bool)
        ijk = np.floor((pts - self.lo) / res).astype(np.int64)
        occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = False
        dist = ndimage.distance_transform_edt(occ) * res
        del occ
        sigma = p['near'] / 2.0
        self.near = np.round(255.0 * np.exp(-(dist ** 2) / (2.0 * sigma ** 2))).astype(np.uint8).ravel()
        self.inlier = (dist <= p['score_dist'])
        del dist
        self.strides = np.array([self.shape[1] * self.shape[2], self.shape[2], 1], dtype=np.int64)
        self.margin = int(math.ceil(p['radius'] / res)) + 1

        n = int(round(p['radius'] / p['step']))
        self.offsets = np.array([(i * p['step'], j * p['step']) for i in range(-n, n + 1) for j in range(-n, n + 1)
                                 if math.hypot(i, j) * p['step'] <= p['radius'] + 1e-6])
        self.offset_flat = (np.round(self.offsets[:, 0] / res) * self.strides[0]
                            + np.round(self.offsets[:, 1] / res) * self.strides[1]).astype(np.int64)
        self.yaws = np.radians(np.arange(0.0, 360.0, p['yaw_step_deg']))

        self.voxelmaps = {}
        for size, _ in p['stages']:
            target, _ = small_gicp.preprocess_points(pts, downsampling_resolution=size, num_threads=p['num_threads'])
            vm = small_gicp.GaussianVoxelMap(size * p['vgicp_leaf'])
            vm.insert(target)
            self.voxelmaps[size] = vm
        self.build_time = time.perf_counter() - t0
        self.map_points = len(pts)

    # ------------------------------------------------------------------ scoring
    def hypothesis_scores(self, pts_body, guess):
        """Closeness score of every (yaw, offset) hypothesis: array [n_yaws, n_offsets] in 0..1."""
        res, sh = self.p['grid_res'], self.shape
        base = (pts_body @ guess[:3, :3].T).astype(np.float32)        # rotated by the guess about the body
        c, s = np.cos(self.yaws).astype(np.float32), np.sin(self.yaws).astype(np.float32)
        t = ((guess[:3, 3] - self.lo) / res).astype(np.float32)
        x = (c[:, None] * base[None, :, 0] - s[:, None] * base[None, :, 1]) / res + t[0]
        y = (s[:, None] * base[None, :, 0] + c[:, None] * base[None, :, 1]) / res + t[1]
        z = base[None, :, 2] / res + t[2]
        m = self.margin
        i = np.clip(np.floor(x), m, sh[0] - 1 - m).astype(np.int64)
        j = np.clip(np.floor(y), m, sh[1] - 1 - m).astype(np.int64)
        k = np.clip(np.floor(z), 0, sh[2] - 1).astype(np.int64)
        flat = i * self.strides[0] + j * self.strides[1] + k                      # yaws x points
        vals = self.near[flat[:, None, :] + self.offset_flat[None, :, None]]     # yaws x offsets x points
        return vals.sum(axis=2, dtype=np.uint32) / (255.0 * max(1, len(pts_body)))

    def score(self, pts_body, T):
        """Fraction of the points within score_dist of the map at pose T."""
        if len(pts_body) == 0:
            return 0.0
        ijk = np.floor((pts_body @ T[:3, :3].T + T[:3, 3] - self.lo) / self.p['grid_res']).astype(np.int64)
        inside = ((ijk >= 0) & (ijk < self.shape)).all(axis=1)
        ijk = ijk[inside]
        return float(self.inlier[ijk[:, 0], ijk[:, 1], ijk[:, 2]].sum()) / len(pts_body)

    def _pick(self, scores):
        """Best top_k hypotheses that are not near an already picked one: [(yaw index, offset index)]."""
        p = self.p
        min_yaw = math.radians(p['nms_yaw_deg'])
        picked = []
        for f in np.argsort(-scores, axis=None):
            yi, oi = divmod(int(f), scores.shape[1])
            if all(math.hypot(*(self.offsets[oi] - self.offsets[po])) >= p['nms_dist']
                   or abs(_wrap(self.yaws[yi] - self.yaws[py])) >= min_yaw for py, po in picked):
                picked.append((yi, oi))
                if len(picked) >= p['top_k']:
                    break
        return picked

    def _align(self, size, source, T, dist):
        r = small_gicp.align(self.voxelmaps[size], source, init_T_target_source=T,
                             max_correspondence_distance=dist, num_threads=self.p['num_threads'],
                             max_iterations=self.p['max_iterations'])
        return r.T_target_source

    # ------------------------------------------------------------------ whole-map search
    def _map_positions(self, step):
        """Positions (x, y), on a `step` grid aligned to the closeness grid, inside the map footprint
        (map points within global_footprint_margin in xy)."""
        res = self.p['grid_res']
        cells = max(1, int(round(step / res)))
        near2d = self.inlier.any(axis=2)                      # columns with a map point (0.3 m dilated)
        occ = near2d[::cells, ::cells]
        margin = max(1, int(round(self.p['global_footprint_margin'] / step)))
        foot = ndimage.binary_dilation(occ, iterations=margin)
        ii, jj = np.nonzero(foot)
        return np.stack([ii * cells, jj * cells], axis=1)     # cell indices (i, j) of the positions

    def global_hypothesis_scores(self, pts_body, attitude, z, step):
        """Scores [n_yaws, n_positions] of every heading at every map position, and the positions (x, y)."""
        res, sh = self.p['grid_res'], self.shape
        cells = self._map_positions(step)
        heading = math.atan2(attitude[1, 0], attitude[0, 0])
        base = (pts_body @ (_rot_z(-heading) @ attitude[:3, :3]).T).astype(np.float32)
        kz = np.clip(np.floor((base[:, 2] + z - self.lo[2]) / res), 0, sh[2] - 1).astype(np.int64)
        scores = np.empty((len(self.yaws), len(cells)))
        for yi, yaw in enumerate(self.yaws):
            c, s = math.cos(yaw), math.sin(yaw)
            fx = np.floor((c * base[:, 0] - s * base[:, 1]) / res).astype(np.int64)
            fy = np.floor((s * base[:, 0] + c * base[:, 1]) / res).astype(np.int64)
            i = np.clip(cells[:, 0:1] + fx[None, :], 0, sh[0] - 1)   # the padding at the edges is empty
            j = np.clip(cells[:, 1:2] + fy[None, :], 0, sh[1] - 1)
            flat = i * self.strides[0] + j * self.strides[1] + kz[None, :]
            scores[yi] = self.near[flat].sum(axis=1, dtype=np.uint32) / (255.0 * max(1, len(pts_body)))
        xy = self.lo[:2] + cells * res                       # position of cell (i, j) as scored
        return scores, xy, heading

    def localize_global(self, cloud_body, attitude, z, step=None, top_k=None):
        """Search the whole map. attitude: 4x4 (or 3x3) map->body rotation with the right roll/pitch
        (heading is ignored), z: body height in the map. Returns (T_map_body, score, info)."""
        p = self.p
        t0 = time.perf_counter()
        step = step or p['global_step']
        top_k = top_k or p['global_top_k']
        cloud = np.asarray(cloud_body, dtype=np.float64)[:, :3]
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        if len(cloud) < 50:
            raise ValueError(f'cloud has only {len(cloud)} points')
        A = np.eye(4)
        A[:3, :3] = np.asarray(attitude)[:3, :3]
        pts = _subsample(_voxel(cloud, p['pre_voxel']), p['pre_points'])
        scores, xy, heading = self.global_hypothesis_scores(pts, A, z, step)

        min_yaw = math.radians(p['nms_yaw_deg'])
        picks = []
        for f in np.argsort(-scores, axis=None):
            yi, pi = divmod(int(f), scores.shape[1])
            if all(math.hypot(*(xy[pi] - xy[pp])) >= p['nms_dist'] or abs(_wrap(self.yaws[yi] - self.yaws[py])) >= min_yaw
                   for py, pp in picks):
                picks.append((yi, pi))
                if len(picks) >= top_k:
                    break
        R0 = _rot_z(-heading) @ A[:3, :3]
        guesses = []
        for yi, pi in picks:
            T = np.eye(4)
            T[:3, :3] = _rot_z(self.yaws[yi]) @ R0
            T[:3, 3] = [xy[pi, 0], xy[pi, 1], z]
            guesses.append(T)
        T, score, alternatives = self._refine(cloud, guesses, keep=p['global_keep'])
        info = {'hypotheses': int(scores.size), 'refined': len(picks), 'positions': int(scores.shape[1]),
                'hypothesis_best': round(float(scores.max()), 3), 'alternatives': alternatives,
                'time': round(time.perf_counter() - t0, 3)}
        return T, score, info

    def _refine(self, cloud, guesses, keep):
        """Coarse VGICP of every guess, fine VGICP of the best `keep`. Returns (T, score, fine scores)."""
        p = self.p
        sources = {size: small_gicp.preprocess_points(cloud, downsampling_resolution=size,
                                                      num_threads=p['num_threads'])[0]
                   for size, _ in p['stages']}
        score_pts = _subsample(_voxel(cloud, p['score_voxel']), p['score_points'])
        coarse_size, coarse_dist = p['stages'][0]
        coarse = []
        for T in guesses:
            T = self._align(coarse_size, sources[coarse_size], T, coarse_dist)
            coarse.append((self.score(score_pts, T), T))
        coarse.sort(key=lambda c: -c[0])
        fine = []
        for _, T in coarse[:keep]:
            for size, dist in p['stages'][1:]:
                T = self._align(size, sources[size], T, dist)
            fine.append((self.score(score_pts, T), T))
        fine.sort(key=lambda c: -c[0])
        return fine[0][1], fine[0][0], [(round(s, 3), T) for s, T in fine]

    # ------------------------------------------------------------------ main entry
    def localize(self, cloud_body, guess):
        """cloud_body: Nx3 points in the body frame (a few merged frames work best),
        guess: 4x4 map->body estimate. Returns (T_map_body, score, info)."""
        p = self.p
        t0 = time.perf_counter()
        cloud = np.asarray(cloud_body, dtype=np.float64)[:, :3]
        cloud = cloud[np.isfinite(cloud).all(axis=1)]
        guess = np.asarray(guess, dtype=np.float64)
        if len(cloud) < 50:
            raise ValueError(f'cloud has only {len(cloud)} points')

        pts = _subsample(_voxel(cloud, p['pre_voxel']), p['pre_points'])
        scores = self.hypothesis_scores(pts, guess)
        picks = self._pick(scores)

        sources = {size: small_gicp.preprocess_points(cloud, downsampling_resolution=size,
                                                      num_threads=p['num_threads'])[0]
                   for size, _ in p['stages']}
        score_pts = _subsample(_voxel(cloud, p['score_voxel']), p['score_points'])
        coarse_size, coarse_dist = p['stages'][0]
        coarse = []
        for yi, oi in picks:
            T = guess.copy()
            T[:3, :3] = _rot_z(self.yaws[yi]) @ guess[:3, :3]
            T[0, 3] += self.offsets[oi, 0]
            T[1, 3] += self.offsets[oi, 1]
            T = self._align(coarse_size, sources[coarse_size], T, coarse_dist)
            coarse.append((self.score(score_pts, T), T))
        coarse.sort(key=lambda c: -c[0])

        best = None
        for _, T in coarse[:p['keep']]:
            for size, dist in p['stages'][1:]:
                T = self._align(size, sources[size], T, dist)
            s = self.score(score_pts, T)
            if best is None or s > best[0]:
                best = (s, T)
        info = {'hypotheses': int(scores.size), 'refined': len(picks),
                'hypothesis_best': round(float(scores.max()), 3),
                'time': round(time.perf_counter() - t0, 3)}
        return best[1], best[0], info
