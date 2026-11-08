"""Synthetic, spatial, and T-Drive traces. Every method on a seed shares one draw."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .constants import MAX_N, PAPER
from .rng import clip, gauss, mulberry32, poisson
from .types import Slot

SIGMA = 1.25

SYNTH = {
    "T": 2000,
    "N": 20,
    "sigma": SIGMA,
    "change_at": [0, 400, 800, 1200, 1600, 1800],
    "lambda_factor": [1, 2.2, 1 / 2.2, 1.9, 1 / 2.4, 2.1],
    "mu_shift_sigma": [0, 0.4, -0.55, 0.6, -0.3, 0.5],
    # Min availability × N still covers the peak arrival mean.
    "pi": [0.75, 0.62, 0.8, 0.58, 0.76, 0.6],
    "mu0": 4.0,
    "lambda0": 5.0,
}


def _regime_table():
    lam_list = []
    mu_list = []
    lam = SYNTH["lambda0"]
    mu = SYNTH["mu0"]
    for r in range(len(SYNTH["change_at"])):
        if r > 0:
            lam = lam * SYNTH["lambda_factor"][r]
            mu = mu + SYNTH["mu_shift_sigma"][r] * SYNTH["sigma"]
        lam_list.append(lam)
        mu_list.append(mu)
    return {"lambda": lam_list, "mu": mu_list}


SYNTH_REGIMES = _regime_table()


def regime_index(t: int, cuts: list[int]) -> int:
    r = 0
    for i, cut in enumerate(cuts):
        if t >= cut:
            r = i
    return r


def _homo_tasks(arrivals: int, value: float) -> list[list[float]]:
    qv: list[list[float]] = [[], [], [], []]
    for i in range(arrivals):
        qv[i % 4].append(value)
    return qv


def make_synthetic(seed: int, horizon: int | None = None) -> dict:
    rng = mulberry32(seed)
    T = SYNTH["T"] if horizon is None else min(SYNTH["T"], horizon)
    slots: list[Slot] = []
    for t in range(T):
        r = regime_index(t, SYNTH["change_at"])
        arrivals = min(MAX_N, poisson(rng, SYNTH_REGIMES["lambda"][r]))
        c, a, e, g = [], [], [], []
        mu = SYNTH_REGIMES["mu"][r]
        pi = SYNTH["pi"][r]
        for i in range(SYNTH["N"]):
            cost = clip(mu + SYNTH["sigma"] * gauss(rng), PAPER["cMin"], PAPER["cMax"])
            c.append(cost)
            a.append(1 if rng() < pi else 0)
            e.append(0.45 + 0.7 * rng())
            g.append(i % 4)
        slots.append(
            Slot(
                q_values=_homo_tasks(arrivals, PAPER["taskValue"]),
                c=c,
                a=a,
                e=e,
                g=g,
                hour=(t % 288) // 12,
            )
        )
    return {
        "meta": {
            "id": "synthetic",
            "seed": seed,
            "T": T,
            "description": (
                "Piecewise-stationary synthetic trace. Poisson arrivals and truncated-normal "
                "costs jump together at five change points."
            ),
        },
        "slots": slots,
    }


SPATIAL = {
    "T": 2000,
    "N": 18,
    "grid": 20,
    "hotspots": [(4, 5), (15, 6), (8, 16)],
    "weights": [
        [0.7, 0.2, 0.1],
        [0.15, 0.7, 0.15],
        [0.1, 0.2, 0.7],
        [0.45, 0.45, 0.1],
        [0.2, 0.15, 0.65],
    ],
    "change_at": [0, 400, 800, 1200, 1600],
    "sigma_cells": 2.1,
    "cell_km": 0.75,
    "lambda": 10,
}


def make_spatial(seed: int, horizon: int | None = None) -> dict:
    rng = mulberry32(seed)
    G = SPATIAL["grid"]
    raw = []
    total_v = 0.0
    for y in range(G):
        row = []
        for x in range(G):
            v = math.exp(0.55 * gauss(rng))
            row.append(v)
            total_v += v
        raw.append(row)
    mean = total_v / (G * G)
    value = [[(v / mean) * PAPER["taskValue"] for v in row] for row in raw]
    cells = []
    cdf = []
    acc = 0.0
    for y in range(G):
        for x in range(G):
            acc += value[y][x]
            cells.append((x, y, value[y][x]))
            cdf.append(acc)
    total = acc

    def sample_cell():
        u = rng() * total
        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi) >> 1
            if cdf[mid] < u:
                lo = mid + 1
            else:
                hi = mid
        return cells[lo]

    def quadrant(x, y) -> int:
        return (0 if x < 10 else 1) + (0 if y < 10 else 2)

    cursor = {"t": 0}

    def sample_worker():
        r = regime_index(cursor["t"], SPATIAL["change_at"])
        w = SPATIAL["weights"][r]
        u = rng()
        h = 0
        while h < 2:
            u -= w[h]
            if u <= 0:
                break
            h += 1
        hx, hy = SPATIAL["hotspots"][h]
        x = clip(hx + SPATIAL["sigma_cells"] * gauss(rng), 0, G - 1e-6)
        y = clip(hy + SPATIAL["sigma_cells"] * gauss(rng), 0, G - 1e-6)
        return x, y

    T = SPATIAL["T"] if horizon is None else min(SPATIAL["T"], horizon)
    slots: list[Slot] = []
    for t in range(T):
        cursor["t"] = t
        arrivals = min(MAX_N, poisson(rng, SPATIAL["lambda"]))
        q_values: list[list[float]] = [[], [], [], []]
        task_xy = []
        for _ in range(arrivals):
            x, y, v = sample_cell()
            q = quadrant(math.floor(x), math.floor(y))
            q_values[q].append(v)
            task_xy.append((x, y, v, q))
        for q in q_values:
            q.sort(reverse=True)
        c, a, e, g = [], [], [], []
        for _i in range(SPATIAL["N"]):
            wx, wy = sample_worker()
            best = 1e9
            for tx, ty, _v, _q in task_xy:
                dx = (wx - tx) * SPATIAL["cell_km"]
                dy = (wy - ty) * SPATIAL["cell_km"]
                d = math.hypot(dx, dy)
                if d < best:
                    best = d
            if not task_xy:
                best = 4.0
            cost = clip(1.6 + 0.62 * best + 0.35 * abs(gauss(rng)), PAPER["cMin"], PAPER["cMax"])
            p_avail = 1 / (1 + math.exp(-(1.2 - 0.36 * best)))
            c.append(cost)
            a.append(1 if rng() < p_avail else 0)
            e.append(0.45 + 0.7 * rng())
            g.append(quadrant(math.floor(wx), math.floor(wy)))
        slots.append(Slot(q_values=q_values, c=c, a=a, e=e, g=g, hour=(t % 288) // 12))
    return {
        "meta": {
            "id": "spatial",
            "seed": seed,
            "T": T,
            "description": "20×20 log-normal zone values. Three hotspots. Quadrant-local matching, then spillover.",
        },
        "slots": slots,
    }


TDRIVE_LOGIT = {
    "a0": -0.05,
    "a1": 0.85,
    "theta_d": -0.22,
    "rush_theta_d": -0.32,
    "theta_b": 1.5,
    "rush_hours": [7, 8, 9, 17, 18, 19],
    "cost_intercept": 1.7,
    "cost_per_km": 0.48,
    "cost_battery": 1.15,
    "cost_noise": 0.3,
}


def _sigmoid(z: float) -> float:
    if z > 30:
        return 1.0
    if z < -30:
        return 0.0
    return 1 / (1 + math.exp(-z))


def haversine_km(lon1, lat1, lon2, lat2) -> float:
    R = 6371.0
    to_r = math.pi / 180
    d_lat = (lat2 - lat1) * to_r
    d_lon = (lon2 - lon1) * to_r
    s = math.sin(d_lat / 2) ** 2 + math.cos(lat1 * to_r) * math.cos(lat2 * to_r) * math.sin(d_lon / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(s)))


def load_drive(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_tdrive(drive: dict, seed: int, horizon: int | None = None) -> dict:
    """
    Participation is logistic in haversine distance to the nearest other fix
    in the bin, taxi density, and a disclosed battery proxy.

    Battery is not in the T-Drive sample. The proxy is a diurnal sinusoid plus
    the bin's real inter-sample gap. Each worker also gets N(0, 0.15) battery noise.
    """
    rng = mulberry32(seed)
    bins = drive["bins"]
    first = 0
    last = len(bins) - 1
    while first < len(bins) and bins[first]["c"] <= 0:
        first += 1
    while last > first and bins[last]["c"] <= 0:
        last -= 1
    window = bins[first : last + 1]
    if horizon is not None:
        window = window[:horizon]
        # Keep `first` so the hour index matches the untruncated trace.

    taxi_counts = sorted(b["taxis"] for b in window if b["taxis"] > 0)
    median_t = taxi_counts[len(taxi_counts) // 2] if taxi_counts else 1
    p90 = taxi_counts[int(len(taxi_counts) * 0.9)] if taxi_counts else median_t

    slots: list[Slot] = []
    carry = None
    rush = set(TDRIVE_LOGIT["rush_hours"])
    for i, b in enumerate(window):
        if b["pts"]:
            carry = b
        src = b if b["pts"] else carry
        hour = ((first + i) % 288) // 12
        dens_ratio = b["taxis"] / median_t if median_t > 0 else 0.0
        lam = clip(9 * max(0.25, dens_ratio), 2.5, 20)
        arrivals = min(MAX_N, poisson(rng, lam)) if src else 0
        n_workers = round(clip(8 + 12 * (b["taxis"] / p90), 8, 22)) if src else 0
        theta_d = TDRIVE_LOGIT["rush_theta_d"] if hour in rush else TDRIVE_LOGIT["theta_d"]
        gap = b["gap"] if b["gap"] is not None else 180
        battery = clip(
            0.62 + 0.18 * math.sin(((hour - 5) / 24) * 2 * math.pi) - 0.22 * clip(gap / 900, 0, 1),
            0.12,
            0.97,
        )
        pts = src["pts"] if src else []
        lat0 = src["lat"] if src and src["lat"] is not None else 39.9
        lon0 = src["lon"] if src and src["lon"] is not None else 116.4
        nn = []
        for pi, p in enumerate(pts):
            best = 8.0
            for j, q in enumerate(pts):
                if pi == j:
                    continue
                d = haversine_km(p[0], p[1], q[0], q[1])
                if d > 0.05 and d < best:
                    best = d
            nn.append(best)
        q_values: list[list[float]] = [[], [], [], []]
        for _k in range(arrivals):
            p = pts[int(math.floor(rng() * len(pts)))] if pts else [lon0, lat0]
            q = (0 if p[0] < lon0 else 1) + (0 if p[1] < lat0 else 2)
            q_values[q].append(PAPER["taskValue"])
        c, a, e, g = [], [], [], []
        for _w in range(n_workers):
            idx = int(math.floor(rng() * len(pts))) if pts else 0
            p = pts[idx] if pts else [lon0, lat0]
            best = nn[idx] if idx < len(nn) else 3.0
            battery_i = clip(battery + 0.15 * gauss(rng), 0.08, 0.98)
            z = (
                TDRIVE_LOGIT["a0"]
                + TDRIVE_LOGIT["a1"] * min(1.8, dens_ratio)
                + theta_d * best
                + TDRIVE_LOGIT["theta_b"] * battery_i
            )
            cost = clip(
                TDRIVE_LOGIT["cost_intercept"]
                + TDRIVE_LOGIT["cost_per_km"] * best
                + TDRIVE_LOGIT["cost_battery"] * (1 - battery_i)
                + TDRIVE_LOGIT["cost_noise"] * gauss(rng),
                PAPER["cMin"],
                PAPER["cMax"],
            )
            c.append(cost)
            a.append(1 if rng() < _sigmoid(z) else 0)
            e.append(clip(0.5 + 0.8 * (1 - battery_i), 0.35, 1.3))
            g.append((0 if p[0] < lon0 else 1) + (0 if p[1] < lat0 else 2))
        slots.append(Slot(q_values=q_values, c=c, a=a, e=e, g=g, hour=hour))
    return {
        "meta": {
            "id": "tdrive",
            "seed": seed,
            "T": len(slots),
            "taxis": drive.get("taxis"),
            "points_parsed": drive.get("pointsParsed"),
            "points_in_window": drive.get("pointsInWeekWindow"),
            "slot_minutes": drive.get("slotMinutes"),
            "min_ts": drive.get("minTs"),
            "max_ts": drive.get("maxTs"),
            "median_active_taxis": median_t,
            "description": (
                "Microsoft Research T-Drive sample, February 2008, Beijing. "
                "Five-minute bins. Distance is haversine to the nearest other reservoir fix."
            ),
        },
        "slots": slots,
    }


def default_drive_path() -> Path | None:
    here = Path(__file__).resolve()
    candidates = [
        here.parents[1] / "data" / "tdrive-derived.json",
        here.parents[2] / "src" / "lib" / "sim" / "tdrive-derived.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None
