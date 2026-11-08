"""Paired evaluation. One shared trace per seed; every method sees that draw."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .constants import PAPER
from .methods import apply_decision, build_ablations, build_catalog, build_sensitivity
from .trace import default_drive_path, load_drive, make_spatial, make_synthetic, make_tdrive
from .types import Queues, task_count
from .wasserstein import acceptance_bounds, dual_infimum

SEEDS = [101, 202, 303, 404, 505]
STRIDE = 10


def run_once(slots, method, seed: int) -> dict:
    method.reset(seed)
    qb = qe = qf = 0.0
    utility = 0.0
    completed = 0
    arrivals = 0
    back_sum = 0.0
    task_q = [0, 0, 0, 0]
    done_q = [0, 0, 0, 0]
    cum_u, back, cov, eps, price, S = [], [], [], [], [], []
    detections = 0
    t0 = time.perf_counter()
    for t, slot in enumerate(slots):
        decision = method.decide(slot, t, Queues(qb, qe, qf))
        outcome = apply_decision(slot, decision)
        method.observe(slot, outcome)
        A = task_count(slot)
        for q in range(4):
            task_q[q] += len(slot.q_values[q])
            done_q[q] += outcome.by_task_q[q]
        qb = max(0.0, qb + A - outcome.completed)
        qe = max(0.0, qe + outcome.energy - PAPER["Ebar"])
        qf = max(0.0, qf + outcome.deficit - PAPER["Dbar"])
        utility += outcome.utility
        completed += outcome.completed
        arrivals += A
        back_sum += qb
        diag = method.last_diag()
        if diag is not None:
            detections = diag.detections
        if t % STRIDE == 0:
            cum_u.append(utility)
            back.append(qb)
            cov.append(completed / arrivals if arrivals else 0.0)
            eps.append(diag.eps if diag else 0.0)
            price.append(diag.price if diag else decision.price)
            S.append(diag.S if diag else 0.0)
    min_q = 1.0
    for q in range(4):
        if task_q[q] > 0:
            min_q = min(min_q, done_q[q] / task_q[q])
    return {
        "utility": utility,
        "coverage": completed / arrivals if arrivals else 0.0,
        "backlog": back_sum / len(slots),
        "final_backlog": qb,
        "runtime_ms": (time.perf_counter() - t0) * 1000 / len(slots),
        "min_quadrant": min_q,
        "detections": detections,
        "trace": {"utility": cum_u, "backlog": back, "coverage": cov, "eps": eps, "price": price, "S": S},
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / max(1, len(xs))


def _std(xs: list[float]) -> float:
    m = _mean(xs)
    return (_mean([(x - m) ** 2 for x in xs])) ** 0.5


def _round(x: float, d: int = 4) -> float:
    p = 10**d
    return round(x * p) / p


def aggregate(runs: list[dict], greedy: list[dict] | None, lyap: list[dict] | None) -> dict:
    def rel(num, den):
        return [(u - den[i]) / abs(den[i]) if abs(den[i]) > 1e-9 else 0.0 for i, u in enumerate(num)]

    U = [r["utility"] for r in runs]
    C = [r["coverage"] for r in runs]
    B = [r["backlog"] for r in runs]
    n = len(runs[0]["trace"]["utility"])

    def avg(key):
        return [_round(_mean([r["trace"][key][i] for r in runs]), 2) for i in range(n)]

    def rel_or_none(base, key, scale_runs):
        if base is None:
            return None
        return _round(100 * _mean(rel([r[key] for r in scale_runs], [r[key] for r in base])), 2)

    return {
        "utility": _round(_mean(U), 2),
        "utility_std": _round(_std(U), 2),
        "coverage": _round(_mean(C), 4),
        "coverage_std": _round(_std(C), 4),
        "backlog": _round(_mean(B), 2),
        "backlog_std": _round(_std(B), 2),
        "final_backlog": _round(_mean([r["final_backlog"] for r in runs]), 2),
        "runtime_ms": _round(_mean([r["runtime_ms"] for r in runs]), 3),
        "min_quadrant": _round(_mean([r["min_quadrant"] for r in runs]), 4),
        "detections": _round(_mean([r["detections"] for r in runs]), 2),
        "rel_utility_vs_greedy": rel_or_none(greedy, "utility", runs),
        "rel_coverage_vs_greedy": rel_or_none(greedy, "coverage", runs),
        "rel_backlog_vs_greedy": rel_or_none(greedy, "backlog", runs),
        "rel_utility_vs_lyapunov": rel_or_none(lyap, "utility", runs),
        "rel_coverage_vs_lyapunov": rel_or_none(lyap, "coverage", runs),
        "rel_backlog_vs_lyapunov": rel_or_none(lyap, "backlog", runs),
        "trace": {k: avg(k) for k in ("utility", "backlog", "coverage", "eps", "price", "S")},
    }


BUILDERS = {
    "catalog": build_catalog,
    "ablations": build_ablations,
    "sensitivity": build_sensitivity,
}


def build_traces(dataset: str, seeds: list[int], horizon: int | None, drive_path: str | None):
    if dataset == "synthetic":
        return [make_synthetic(s, horizon)["slots"] for s in seeds], None
    if dataset == "spatial":
        return [make_spatial(s, horizon)["slots"] for s in seeds], None
    path = Path(drive_path) if drive_path else default_drive_path()
    if path is None:
        raise SystemExit("T-Drive bins not found. Pass --tdrive path/to/tdrive-derived.json")
    drive = load_drive(path)
    meta = make_tdrive(drive, seeds[0], horizon)["meta"]
    return [make_tdrive(drive, s, horizon)["slots"] for s in seeds], meta


def evaluate(dataset: str, suite: str, seeds: list[int], horizon: int | None, drive_path: str | None) -> dict:
    traces, meta = build_traces(dataset, seeds, horizon, drive_path)
    methods = BUILDERS[suite]()
    by_id: dict[str, list[dict]] = {}
    names = {}
    for method in methods:
        names[method.id] = method.name
        runs = []
        for seed, slots in zip(seeds, traces):
            t0 = time.perf_counter()
            run = run_once(slots, method, seed)
            runs.append(run)
            print(
                f"  {method.id} seed {seed} {time.perf_counter() - t0:.1f}s "
                f"U={run['utility']:.0f} cov={run['coverage']:.3f}",
                flush=True,
            )
        by_id[method.id] = runs
    greedy = by_id.get("greedy") or by_id.get("adapt") or by_id.get("default")
    lyap = by_id.get("lyapunov") or by_id.get("no-dro")
    out = {}
    for mid, runs in by_id.items():
        out[mid] = {"id": mid, "name": names[mid], **aggregate(runs, greedy, lyap)}
    return {"dataset": dataset, "suite": suite, "seeds": seeds, "horizon": horizon, "tdrive": meta, "rows": out}


def self_check() -> None:
    """Dual identity, plus a short synthetic ADAPT step that must stay finite."""
    costs = [1.2, 3.4, 5.0, 8.1, 2.2, 6.6, 4.0, 9.5]
    avails = [1, 1, 0, 1, 1, 0, 1, 1]
    worst = 0.0
    for p in (2.0, 4.5, 7.0):
        for eps in (0.0, 0.05, 0.4, 1.5):
            b = acceptance_bounds(costs, avails, len(costs), p, eps)
            d = dual_infimum(costs, avails, len(costs), p, eps)
            err = abs(d["J"] / p - b["alpha_ub"])
            worst = max(worst, err)
            if not (0 <= b["alpha_lb"] <= b["alpha_emp"] + 1e-12 <= b["alpha_ub"] + 1e-12 <= 1 + 1e-9):
                raise SystemExit(f"bounds out of order at p={p} eps={eps}: {b}")
    if worst > 1e-8:
        raise SystemExit(f"dual identity failed, max |J/p - alpha_ub| = {worst}")
    slots = make_synthetic(101, horizon=60)["slots"]
    from .methods import Adapt

    method = Adapt(id="adapt", name="ADAPT")
    run = run_once(slots, method, 101)
    if not (0 <= run["coverage"] <= 1 and run["final_backlog"] >= 0):
        raise SystemExit(f"short run looks broken: {run}")
    print(
        f"self-check ok  dual_err={worst:.3e}  "
        f"T=60 U={run['utility']:.2f} cov={run['coverage']:.3f} Q={run['final_backlog']:.1f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ADAPT simulator: adaptive Wasserstein posted prices with Lyapunov tracking."
    )
    parser.add_argument("--check", action="store_true", help="Dual identity and a 60-slot smoke run.")
    parser.add_argument("--dataset", choices=["synthetic", "spatial", "tdrive"], default="synthetic")
    parser.add_argument("--suite", choices=["catalog", "ablations", "sensitivity"], default="catalog")
    parser.add_argument("--seeds", default="101,202,303,404,505")
    parser.add_argument("--horizon", type=int, default=None, help="Truncate each trace. Default is the full horizon.")
    parser.add_argument("--tdrive", default=None, help="Path to tdrive-derived.json")
    parser.add_argument("--out", default=None, help="Write the aggregate JSON here.")
    args = parser.parse_args(argv)
    if args.check:
        self_check()
        return
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"{args.dataset} / {args.suite}  seeds={seeds}  horizon={args.horizon or 'full'}", flush=True)
    result = evaluate(args.dataset, args.suite, seeds, args.horizon, args.tdrive)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        for row in result["rows"].values():
            print(
                f"{row['id']:16} U={row['utility']:>12}  cov={row['coverage']:.3f}  "
                f"Q={row['backlog']:.1f}  vsLyap={row['rel_utility_vs_lyapunov']}"
            )


if __name__ == "__main__":
    main()
