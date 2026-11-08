"""
Ground metric on worker responses ξ = (cost, availability):

    ||ξ − ξ'||_1 = |c − c'| + (c_max − c_min) |a − a'|.

Availability is weighted by the cost span so that flipping a ∈ {0, 1} costs
as much as moving a cost from c_min to c_max. Distances are in currency.
A radius of ε0 = 0.05 is then a five-cent hedge. On a normalized metric of
diameter 2 the same numeric radius covers the whole space.

The Page residual stays on normalized features (c̃, a) ∈ [0, 1]^2, because
ν = 0.1 is the right size for that noise. γ converts the statistic into
currency inside the radius.

For ℓ(p, ξ) = p · 1{a = 1 and c ≤ p} the worst-case acceptance probabilities
inside a W1 ball are an exact fractional knapsack, equal to the Kantorovich
dual
    inf_{λ≥0} { λ ε + (1/n) Σ_k sup_ξ (ℓ − λ ||ξ − ξ_k||) }.
"""

from __future__ import annotations

from .constants import PAPER

CMIN = PAPER["cMin"]
CMAX = PAPER["cMax"]
SPAN = CMAX - CMIN

_out_d: list[float] = []
_in_d: list[float] = []


def cost_tilde(c: float) -> float:
    return (c - CMIN) / SPAN


def acceptance_bounds(costs: list[float], avails: list[float], n: int, p: float, eps: float) -> dict:
    """Exact α_emp, α_lb = inf P(accept), α_ub = sup P(accept) over W1 ≤ eps."""
    if n <= 0:
        return {"alpha_emp": 0.0, "alpha_lb": 0.0, "alpha_ub": 0.0, "mean_transport_ub": 0.0}
    budget = max(0.0, n * max(0.0, eps))
    avail_cost = SPAN
    n_acc = 0
    out_d = _out_d
    in_d = _in_d
    out_d.clear()
    in_d.clear()
    for i in range(n):
        c = costs[i]
        a_on = avails[i] >= 0.5
        if a_on and c <= p + 1e-12:
            n_acc += 1
            out_d.append(max(0.0, p - c))
        elif a_on and c > p:
            in_d.append(c - p)
        elif (not a_on) and c <= p + 1e-12:
            in_d.append(avail_cost)
        else:
            in_d.append(avail_cost + max(0.0, c - p))
    out_d.sort()
    in_d.sort()

    b = budget
    moved_out = 0.0
    for d in out_d:
        if b <= 1e-15:
            break
        if d <= 1e-15:
            moved_out += 1.0
            continue
        if b >= d:
            b -= d
            moved_out += 1.0
        else:
            moved_out += b / d
            b = 0.0

    b = budget
    moved_in = 0.0
    transport_in = 0.0
    for d in in_d:
        if b <= 1e-15:
            break
        if d <= 1e-15:
            moved_in += 1.0
            continue
        if b >= d:
            b -= d
            moved_in += 1.0
            transport_in += d
        else:
            frac = b / d
            moved_in += frac
            transport_in += b
            b = 0.0

    return {
        "alpha_emp": n_acc / n,
        "alpha_lb": max(0.0, (n_acc - moved_out) / n),
        "alpha_ub": min(1.0, (n_acc + moved_in) / n),
        "mean_transport_ub": transport_in / n,
    }


def dual_at(costs, avails, n: int, p: float, eps: float, lam: float) -> dict:
    """Dual objective J(λ) and subgradient ε − mean transport of the sup plan."""
    if n <= 0:
        return {"J": lam * eps, "subgrad": eps, "mean_loss": 0.0}
    avail_cost = SPAN
    total = 0.0
    sum_dist = 0.0
    for i in range(n):
        c = costs[i]
        a = 1 if avails[i] >= 0.5 else 0
        ell = p if (a == 1 and c <= p + 1e-12) else 0.0
        if c <= p + 1e-12:
            d_acc = 0.0 if a == 1 else avail_cost
        else:
            d_acc = (c - p) + (0.0 if a == 1 else avail_cost)
        val_acc = p - lam * d_acc
        if val_acc > ell:
            total += val_acc
            sum_dist += d_acc
        else:
            total += ell
    mean = total / n
    mean_dist = sum_dist / n
    return {"J": lam * eps + mean, "subgrad": eps - mean_dist, "mean_loss": mean}


def dual_infimum(costs, avails, n: int, p: float, eps: float) -> dict:
    """Exact inf over λ ≥ 0 by bisection on the monotone subgradient."""
    if eps <= 1e-12 or n <= 0:
        emp = 0.0
        for i in range(n):
            if avails[i] >= 0.5 and costs[i] <= p + 1e-12:
                emp += p
        return {"lambda": 0.0, "J": (emp / n) if n > 0 else 0.0}
    lo = 0.0
    hi = max(p, 1.0)
    g_hi = dual_at(costs, avails, n, p, eps, hi)["subgrad"]
    guard = 0
    while g_hi < 0 and hi < 1e6 and guard < 40:
        hi *= 2
        g_hi = dual_at(costs, avails, n, p, eps, hi)["subgrad"]
        guard += 1
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g = dual_at(costs, avails, n, p, eps, mid)["subgrad"]
        if g > 0:
            hi = mid
        else:
            lo = mid
    lam = 0.5 * (lo + hi)
    J = dual_at(costs, avails, n, p, eps, lam)["J"]
    return {"lambda": lam, "J": J}


def projected_dual_ascent(costs, avails, n, p, eps, K, eta) -> dict:
    """
    K-step projected gradient descent on J(λ).

    The manuscript writes λ ← [λ + η ∇]_+. J is an infimum, so the descent
    map is λ ← [λ − η ∂J/∂λ]_+. The plus in the manuscript is ascent on −J;
    both target the same stationary point when the sign of ∇ is the descent
    direction. This function descends on J.
    """
    lam = 0.0
    path: list[float] = []
    for k in range(K):
        g = dual_at(costs, avails, n, p, eps, lam)["subgrad"]
        lam = max(0.0, lam - eta * g)
        if k == K - 1 or k % 10 == 0:
            path.append(lam)
    J = dual_at(costs, avails, n, p, eps, lam)["J"]
    return {"lambda": lam, "J": J, "path": path}


def payment_matches_knapsack(costs, avails, n, p, eps, tol: float = 1e-8) -> float:
    """Return |J* / p − α_ub| when p > 0. Zero means the dual identity holds."""
    bounds = acceptance_bounds(costs, avails, n, p, eps)
    dual = dual_infimum(costs, avails, n, p, eps)
    if p <= 0:
        return abs(dual["J"])
    return abs(dual["J"] / p - bounds["alpha_ub"])
