"""
Posted-price ADAPT and the nine comparison rules.

Posted-price methods do not read the current bids when choosing p.
The CUSUM residual does see the current slot's mean cost and availability;
the price grid is scored on the historical window only.
Auctions see the current bids.
"""

from __future__ import annotations

import math

from .constants import PAPER
from .realize import match_auction, match_posted
from .rng import clip, mulberry32
from .types import Decision, Outcome, Queues, Slot, StepDiag, task_count
from .wasserstein import acceptance_bounds, cost_tilde


def _grid() -> list[float]:
    G = PAPER["priceGrid"]
    span = PAPER["cMax"] - PAPER["cMin"]
    return [PAPER["cMin"] + (i / (G - 1)) * span for i in range(G)]


GRID = _grid()


def mean_task_value(slot: Slot) -> float:
    arrivals = task_count(slot)
    if arrivals <= 0:
        return PAPER["taskValue"]
    total = 0.0
    for q in slot.q_values:
        total += sum(q)
    return total / arrivals


class Buffer:
    def __init__(self):
        self.c: list[float] = []
        self.a: list[float] = []
        self.e: list[float] = []
        self.g: list[int] = []

    def clear(self):
        self.c.clear()
        self.a.clear()
        self.e.clear()
        self.g.clear()

    def push_slot(self, slot: Slot):
        self.c.extend(slot.c)
        self.a.extend(slot.a)
        self.e.extend(slot.e)
        self.g.extend(slot.g)
        extra = len(self.c) - PAPER["W"]
        if extra > 0:
            del self.c[:extra]
            del self.a[:extra]
            del self.e[:extra]
            del self.g[:extra]

    def keep_last(self, n: int):
        if len(self.c) > n:
            self.c = self.c[-n:]
            self.a = self.a[-n:]
            self.e = self.e[-n:]
            self.g = self.g[-n:]


def group_counts(slot: Slot) -> list[int]:
    n_g = [0, 0, 0, 0]
    for g in slot.g:
        n_g[g] += 1
    return n_g


def predicted_deficit(c, a, g, p, cur) -> float:
    n = len(c)
    n_g = [0, 0, 0, 0]
    a_g = [0, 0, 0, 0]
    acc = 0
    for i in range(n):
        gi = g[i] & 3
        n_g[gi] += 1
        if a[i] == 1 and c[i] <= p:
            a_g[gi] += 1
            acc += 1
    alpha = acc / n if n > 0 else 0.4
    tot = 0.0
    supply = [0.0, 0.0, 0.0, 0.0]
    for gi in range(4):
        ag = a_g[gi] / n_g[gi] if n_g[gi] > 0 else alpha
        supply[gi] = cur[gi] * ag
        tot += supply[gi]
    if tot <= 1e-9:
        return 1.0
    D = 0.0
    for gi in range(4):
        D += max(0.0, 0.25 - supply[gi] / tot)
    return D


def saa_alpha(c, a, p, eps, rng, reps: int):
    n = len(c)
    if n == 0:
        return {"alpha_lb": 0.0, "alpha_ub": 0.0, "alpha_emp": 0.0}
    emp = 0
    for i in range(n):
        if a[i] == 1 and c[i] <= p:
            emp += 1
    alpha_emp = emp / n
    lo = 1.0
    hi = 0.0
    span = PAPER["cMax"] - PAPER["cMin"]
    for _r in range(reps):
        acc = 0
        for i in range(n):
            u = rng()
            budget = eps * u
            flip = 1 if rng() < min(1.0, eps - budget) else 0
            direction = -1 if rng() < 0.5 else 1
            cc = clip(c[i] + direction * budget * span, PAPER["cMin"], PAPER["cMax"])
            aa = (1 - a[i]) if flip else a[i]
            if aa == 1 and cc <= p:
                acc += 1
        al = acc / n
        if al < lo:
            lo = al
        if al > hi:
            hi = al
    return {"alpha_lb": min(alpha_emp, lo), "alpha_ub": max(alpha_emp, hi), "alpha_emp": alpha_emp}


class Method:
    id: str
    name: str
    family: str

    def reset(self, seed: int = 1) -> None:
        return None

    def decide(self, slot: Slot, t: int, Q: Queues) -> Decision:
        raise NotImplementedError

    def observe(self, slot: Slot, outcome: Outcome) -> None:
        return None

    def last_diag(self) -> StepDiag | None:
        return None


class Adapt(Method):
    def __init__(
        self,
        id: str,
        name: str,
        family: str = "ADAPT",
        eps0: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        eta: float | None = None,
        V: float | None = None,
        fixed_eps: float | None = None,
        use_cusum: bool | None = None,
        use_queue_in_radius: bool | None = None,
        use_lyapunov: bool = True,
        extract: str = "dual",
    ):
        self.id = id
        self.name = name
        self.family = family
        self.eps0 = PAPER["eps0"] if eps0 is None else eps0
        self.beta = PAPER["beta"] if beta is None else beta
        self.gamma = PAPER["gamma"] if gamma is None else gamma
        self.eta = PAPER["eta"] if eta is None else eta
        self.V = PAPER["V"] if V is None else V
        self.fixed_eps = fixed_eps
        self.use_cusum = (fixed_eps is None) if use_cusum is None else use_cusum
        self.use_queue = (fixed_eps is None) if use_queue_in_radius is None else use_queue_in_radius
        self.use_lyap = use_lyapunov
        self.extract = extract
        self.buf = Buffer()
        self.S = 0.0
        self.detections = 0
        self.diag: StepDiag | None = None
        self.rng = mulberry32(1)
        self.pending_reset = False

    def reset(self, seed: int = 1) -> None:
        self.buf.clear()
        self.S = 0.0
        self.detections = 0
        self.diag = None
        self.pending_reset = False
        self.rng = mulberry32(seed + 17)

    def last_diag(self):
        return self.diag

    def _bounds_at(self, p: float, eps: float):
        if self.extract == "saa":
            s = saa_alpha(self.buf.c, self.buf.a, p, eps, self.rng, 10)
            return s
        return acceptance_bounds(self.buf.c, self.buf.a, len(self.buf.c), p, eps)

    def decide(self, slot: Slot, t: int, Q: Queues) -> Decision:
        n = len(self.buf.c)
        N = len(slot.c)
        arrivals = task_count(slot)
        norm = 0.0
        if n > 0 and N > 0:
            mc = ma = bc = ba = 0.0
            for i in range(N):
                mc += cost_tilde(slot.c[i])
                ma += slot.a[i]
            mc /= N
            ma /= N
            for i in range(n):
                bc += cost_tilde(self.buf.c[i])
                ba += self.buf.a[i]
            bc /= n
            ba /= n
            norm = math.hypot(mc - bc, ma - ba)
        if self.use_cusum and n > 0:
            self.S = max(0.0, self.S + (norm - PAPER["nu"]))
        conc = self.beta * math.sqrt(math.log(t + 1) / n) if n > 0 else 0.0
        s_term = self.gamma * self.S if (self.use_cusum and self.fixed_eps is None) else 0.0
        pressure = min(1.0, max(0.0, Q.b / PAPER["Qmax"]))
        q_term = self.eta * pressure if (self.use_queue and self.fixed_eps is None) else 0.0
        eps = self.fixed_eps if self.fixed_eps is not None else self.eps0 + conc + s_term + q_term
        if self.use_cusum and self.fixed_eps is None and self.S > PAPER["tau"]:
            self.pending_reset = True

        cur = group_counts(slot)
        e_mean = 0.8
        if n > 0:
            e_mean = sum(self.buf.e) / n
        v_bar = mean_task_value(slot)

        best_p = min(4.5, v_bar)
        best_phi = math.inf
        best_b = {"alpha_emp": 0.0, "alpha_lb": 0.0, "alpha_ub": 0.0}
        if n >= 8 and arrivals > 0 and N > 0:
            last_p = -1.0
            for p0 in GRID:
                p = min(p0, v_bar)
                if abs(p - last_p) < 1e-9:
                    continue
                last_p = p
                b = self._bounds_at(p, max(0.0, eps))
                mu_lb = min(arrivals, N * b["alpha_lb"])
                mu_emp = min(arrivals, N * b["alpha_emp"])
                pay_emp = p * mu_emp
                Df = predicted_deficit(self.buf.c, self.buf.a, self.buf.g, p, cur)
                drift = (
                    Q.b * (arrivals - mu_lb)
                    + Q.e * (e_mean * mu_lb - PAPER["Ebar"])
                    + Q.f * (Df - PAPER["Dbar"])
                )
                # One Wasserstein plan — the service-minimizing transport —
                # prices and serves. p·α_ub is recorded, not added on top.
                if self.extract == "myopic":
                    phi = pay_emp - v_bar * mu_emp
                elif self.use_lyap:
                    phi = drift + self.V * (p - v_bar) * mu_lb
                else:
                    phi = self.V * p * min(arrivals, N * b["alpha_ub"])
                if phi < best_phi:
                    best_phi = phi
                    best_p = p
                    best_b = b
        self.diag = StepDiag(
            eps=eps,
            S=self.S,
            conc=conc,
            s_term=s_term,
            q_term=q_term,
            price=best_p,
            alpha_emp=best_b["alpha_emp"],
            alpha_lb=best_b["alpha_lb"],
            alpha_ub=best_b["alpha_ub"],
            phi=best_phi if math.isfinite(best_phi) else 0.0,
            norm=norm,
            detections=self.detections,
            n_buf=n,
        )
        return Decision(price=best_p, mode="posted", diag={"eps": eps, "S": self.S, "price": best_p})

    def observe(self, slot: Slot, outcome: Outcome) -> None:
        self.buf.push_slot(slot)
        if self.pending_reset:
            self.buf.keep_last(len(slot.c))
            self.S = 0.0
            self.detections += 1
            self.pending_reset = False


def _flatten_tasks(slot: Slot) -> list[float]:
    tasks = [v for q in slot.q_values for v in q]
    tasks.sort(reverse=True)
    return tasks


class Greedy(Method):
    id = "greedy"
    name = "Greedy critical auction"
    family = "Auction"

    def decide(self, slot: Slot, t: int, Q: Queues) -> Decision:
        tasks = _flatten_tasks(slot)
        idx = [i for i in range(len(slot.c)) if slot.a[i] == 1]
        idx.sort(key=lambda i: slot.c[i])
        used = [False] * len(tasks)
        chosen: list[tuple[int, float]] = []
        for i in idx:
            for ti in range(len(tasks)):
                if not used[ti] and slot.c[i] <= tasks[ti] + 1e-12:
                    used[ti] = True
                    chosen.append((i, tasks[ti]))
                    break
        chosen_set = {i for i, _v in chosen}
        loser = math.inf
        for i in idx:
            if i not in chosen_set:
                loser = slot.c[i]
                break
        pay = [0.0] * len(slot.c)
        for i, v in chosen:
            critical = loser if math.isfinite(loser) else slot.c[i]
            pay[i] = min(v, max(slot.c[i], critical))
        price = (sum(pay[i] for i, _v in chosen) / len(chosen)) if chosen else 0.0
        return Decision(price=price, mode="auction", pay=pay)


class Biobj(Method):
    """
    Worst weight ω ∈ {0.3, 0.5, 0.7} on normalized cost versus normalized energy.
    Payments use the same critical-price rule as the greedy auction.
    This is the reduction described for the bi-objective baseline, not a MILP port.
    """

    id = "biobj"
    name = "Bi-objective robust"
    family = "Auction"
    omegas = (0.3, 0.5, 0.7)

    def decide(self, slot: Slot, t: int, Q: Queues) -> Decision:
        v_ref = mean_task_value(slot)
        scored: list[tuple[int, float]] = []
        for i in range(len(slot.c)):
            if slot.a[i] != 1:
                continue
            worst = -math.inf
            for w in self.omegas:
                s = w * (slot.c[i] / v_ref) + (1 - w) * (slot.e[i] / 1.2)
                if s > worst:
                    worst = s
            scored.append((i, worst))
        scored.sort(key=lambda sc: (sc[1], slot.c[sc[0]]))
        tasks = _flatten_tasks(slot)
        used = [False] * len(tasks)
        chosen: list[tuple[int, float]] = []
        for i, _s in scored:
            for ti in range(len(tasks)):
                if not used[ti] and slot.c[i] <= tasks[ti] + 1e-12:
                    used[ti] = True
                    chosen.append((i, tasks[ti]))
                    break
        chosen_set = {i for i, _v in chosen}
        loser = math.inf
        by_cost = sorted((i for i, _s in scored), key=lambda i: slot.c[i])
        for i in by_cost:
            if i not in chosen_set:
                loser = slot.c[i]
                break
        pay = [0.0] * len(slot.c)
        for i, v in chosen:
            critical = loser if math.isfinite(loser) else slot.c[i]
            pay[i] = min(v, max(slot.c[i], critical))
        price = (sum(pay[i] for i, _v in chosen) / len(chosen)) if chosen else 0.0
        return Decision(price=price, mode="auction", pay=pay)


class Empirical(Method):
    id = "empirical"
    name = "Empirical mean"
    family = "Posted price"

    def __init__(self):
        self.buf = Buffer()

    def reset(self, seed: int = 1) -> None:
        self.buf.clear()

    def observe(self, slot: Slot, outcome: Outcome) -> None:
        self.buf.push_slot(slot)

    def decide(self, slot: Slot, t: int, Q: Queues) -> Decision:
        n = len(self.buf.c)
        N = len(slot.c)
        arrivals = task_count(slot)
        v_bar = mean_task_value(slot)
        best_p = 4.5
        best_u = -math.inf
        if n >= 8 and arrivals > 0:
            for p in GRID:
                acc = 0
                for i in range(n):
                    if self.buf.a[i] == 1 and self.buf.c[i] <= p:
                        acc += 1
                mu = min(arrivals, N * (acc / n))
                u = v_bar * mu - p * mu
                if u > best_u:
                    best_u = u
                    best_p = p
        return Decision(price=best_p, mode="posted")


class DroIgr(Method):
    id = "droigr"
    name = "DRO-IGR"
    family = "Posted price"

    def __init__(self):
        self.buf = Buffer()

    def reset(self, seed: int = 1) -> None:
        self.buf.clear()

    def observe(self, slot: Slot, outcome: Outcome) -> None:
        self.buf.push_slot(slot)

    def decide(self, slot: Slot, t: int, Q: Queues) -> Decision:
        n = len(self.buf.c)
        N = len(slot.c)
        arrivals = task_count(slot)
        best_p = 4.5
        if n >= 8 and arrivals > 0 and N > 0:
            mean = sum(self.buf.a) / n
            var = 0.0
            for i in range(n):
                d = self.buf.a[i] - mean
                var += d * d
            std = math.sqrt(var / n)
            importance = min(1.5, arrivals / 10)
            eps = clip(0.05 + 0.45 * std + 0.12 * importance, 0.02, 0.85)
            target = min(arrivals, N) * 0.9
            chosen = None
            best_mu = -1.0
            best_mu_p = GRID[0]
            for p in GRID:
                b = acceptance_bounds(self.buf.c, self.buf.a, n, p, eps)
                mu = N * b["alpha_lb"]
                if mu > best_mu:
                    best_mu = mu
                    best_mu_p = p
                if chosen is None and mu >= target:
                    chosen = p
            best_p = best_mu_p if chosen is None else chosen
        return Decision(price=best_p, mode="posted")


class BpUcb(Method):
    id = "bpucb"
    name = "BP-UCB"
    family = "Posted price"

    def __init__(self):
        self.arms = [p for i, p in enumerate(GRID) if i % 2 == 0]
        self.pulls: list[int] = []
        self.sum_r: list[float] = []
        self.t = 0
        self.last_arm = 0
        self.reset()

    def reset(self, seed: int = 1) -> None:
        self.pulls = [0] * len(self.arms)
        self.sum_r = [0.0] * len(self.arms)
        self.t = 0
        self.last_arm = 0

    def decide(self, slot: Slot, t: int, Q: Queues) -> Decision:
        self.t = t + 1
        arm = 0
        for i in range(len(self.arms)):
            if self.pulls[i] == 0:
                self.last_arm = i
                return Decision(price=self.arms[i], mode="posted")
        best = -math.inf
        for i in range(len(self.arms)):
            mean = self.sum_r[i] / self.pulls[i]
            ucb = mean + math.sqrt((2 * math.log(self.t)) / self.pulls[i])
            if ucb > best:
                best = ucb
                arm = i
        self.last_arm = arm
        return Decision(price=self.arms[arm], mode="posted")

    def observe(self, slot: Slot, outcome: Outcome) -> None:
        self.pulls[self.last_arm] += 1
        self.sum_r[self.last_arm] += outcome.utility


class NsTd(Method):
    """
    Tabular TD on a coarse (backlog × recent availability) state.
    Same problem class as a non-stationary actor-critic posted price.
    Not a weight-level copy of any unpublished network.
    """

    id = "nstd"
    name = "Non-stat. TD pricing"
    family = "Posted price"

    def __init__(self):
        self.arms = [p for i, p in enumerate(GRID) if i % 2 == 0]
        self.n_s = 16
        self.lr = 0.12
        self.gamma = 0.9
        self.eps_g = 0.08
        self.Qtab: list[list[float]] = []
        self.ewma = 0.5
        self.prev_s = 0
        self.prev_a = 0
        self.have = False
        self.rng = mulberry32(1)
        self.reset()

    def reset(self, seed: int = 1) -> None:
        self.Qtab = [[0.0] * len(self.arms) for _ in range(self.n_s)]
        self.ewma = 0.5
        self.have = False
        self.rng = mulberry32(seed + 99)

    @staticmethod
    def _bin(qb: float, avail: float) -> int:
        b = 0 if qb < 8 else 1 if qb < 25 else 2 if qb < 60 else 3
        a = 0 if avail < 0.35 else 1 if avail < 0.55 else 2 if avail < 0.75 else 3
        return b * 4 + a

    def decide(self, slot: Slot, t: int, Q: Queues) -> Decision:
        s = self._bin(Q.b, self.ewma)
        if self.rng() < self.eps_g:
            a = int(math.floor(self.rng() * len(self.arms)))
        else:
            a = 0
            best = -math.inf
            row = self.Qtab[s]
            for i, qv in enumerate(row):
                if qv > best:
                    best = qv
                    a = i
        self.prev_s = s
        self.prev_a = a
        self.have = True
        return Decision(price=self.arms[a], mode="posted")

    def observe(self, slot: Slot, outcome: Outcome) -> None:
        rate = outcome.n_willing / len(slot.c) if slot.c else 0.0
        self.ewma = 0.85 * self.ewma + 0.15 * rate
        if not self.have:
            return
        s_next = self._bin(max(0.0, outcome.unfinished * 4), self.ewma)
        max_next = max(self.Qtab[s_next])
        reward = outcome.utility / 20
        cur = self.Qtab[self.prev_s][self.prev_a]
        self.Qtab[self.prev_s][self.prev_a] = cur + self.lr * (reward + self.gamma * max_next - cur)


def make_adapt(**kwargs) -> Adapt:
    return Adapt(**kwargs)


class AdaptQ(Method):
    """
    Posted price. Current bids stay hidden.

    Frozen after a search on synthetic seed 101:
    ball capped at 0.08, window of 200, price capped at the highest
    available cost in the window, Lyapunov objective with V = 20 on the
    worst-case acceptance plan. The value of a fractional plan is the
    sum of the highest task values, not the mean.
    """

    def __init__(self, id: str = "adaptq", name: str = "ADAPT-Q", eps_cap: float = 0.08, window: int = 200, V: float = 20.0):
        self.id = id
        self.name = name
        self.family = "Posted price"
        self.eps_cap = eps_cap
        self.window = window
        self.V = V
        self.buf = Buffer()
        self.S = 0.0
        self.detections = 0
        self.diag = None
        self.pending_reset = False

    def reset(self, seed: int = 1) -> None:
        self.buf.clear()
        self.S = 0.0
        self.detections = 0
        self.diag = None
        self.pending_reset = False

    def last_diag(self):
        return self.diag

    def decide(self, slot: Slot, t: int, Q: Queues) -> Decision:
        n = len(self.buf.c)
        N = len(slot.c)
        arrivals = task_count(slot)
        norm = 0.0
        if n > 0 and N > 0:
            mc = sum(cost_tilde(c) for c in slot.c) / N
            ma = sum(slot.a) / N
            bc = sum(cost_tilde(c) for c in self.buf.c) / n
            ba = sum(self.buf.a) / n
            norm = math.hypot(mc - bc, ma - ba)
        if n > 0:
            self.S = max(0.0, self.S + (norm - PAPER["nu"]))
        conc = PAPER["beta"] * math.sqrt(math.log(t + 1) / n) if n > 0 else 0.0
        pressure = min(1.0, max(0.0, Q.b / PAPER["Qmax"]))
        eps = min(self.eps_cap, PAPER["eps0"] + conc)
        if self.S > PAPER["tau"]:
            self.pending_reset = True
        values = sorted((v for q in slot.q_values for v in q), reverse=True)
        v_cap = values[0] if values else PAPER["taskValue"]
        p_hi = v_cap
        if n > 0:
            hi = PAPER["cMin"]
            for i in range(n):
                if self.buf.a[i] >= 0.5 and self.buf.c[i] > hi:
                    hi = self.buf.c[i]
            p_hi = min(v_cap, hi)
        e_mean = sum(self.buf.e) / n if n else 0.8
        best_p = min(4.5, v_cap)
        best_s = -math.inf
        best_b = {"alpha_emp": 0.0, "alpha_lb": 0.0, "alpha_ub": 0.0}
        if n >= 8 and arrivals > 0 and N > 0:
            steps = 61
            span = PAPER["cMax"] - PAPER["cMin"]
            cur = group_counts(slot)
            for i in range(steps):
                p = min(p_hi, PAPER["cMin"] + (i / (steps - 1)) * span)
                b = acceptance_bounds(self.buf.c, self.buf.a, n, p, max(0.0, eps))
                K = min(arrivals, N * b["alpha_lb"])
                left = K
                value = 0.0
                for v in values:
                    if left <= 1e-12:
                        break
                    take = 1.0 if left >= 1 else left
                    value += v * take
                    left -= take
                Df = predicted_deficit(self.buf.c, self.buf.a, self.buf.g, p, cur)
                phi = (
                    Q.b * (arrivals - K)
                    + Q.e * (e_mean * K - PAPER["Ebar"])
                    + Q.f * (Df - PAPER["Dbar"])
                    + self.V * (p * K - value)
                )
                score = -phi
                if score > best_s + 1e-9 or (abs(score - best_s) <= 1e-9 and p < best_p):
                    best_s = score
                    best_p = p
                    best_b = b
        self.diag = StepDiag(
            eps=eps,
            S=self.S,
            conc=conc,
            s_term=0.0,
            q_term=0.0,
            price=best_p,
            alpha_emp=best_b["alpha_emp"],
            alpha_lb=best_b["alpha_lb"],
            alpha_ub=best_b["alpha_ub"],
            phi=best_s if math.isfinite(best_s) else 0.0,
            norm=norm,
            detections=self.detections,
            n_buf=n,
        )
        return Decision(price=best_p, mode="posted", diag={"eps": eps, "S": self.S, "price": best_p})

    def observe(self, slot: Slot, outcome: Outcome) -> None:
        self.buf.push_slot(slot)
        extra = len(self.buf.c) - self.window
        if extra > 0:
            del self.buf.c[:extra]
            del self.buf.a[:extra]
            del self.buf.e[:extra]
            del self.buf.g[:extra]
        if self.pending_reset:
            self.buf.keep_last(len(slot.c))
            self.S = 0.0
            self.detections += 1
            self.pending_reset = False


def build_catalog() -> list[Method]:
    return [
        AdaptQ(),
        Adapt(id="adapt", name="ADAPT"),
        Greedy(),
        Biobj(),
        Adapt(
            id="lyapunov",
            name="Lyapunov online",
            family="Posted price",
            fixed_eps=0.0,
            use_cusum=False,
            use_queue_in_radius=False,
            use_lyapunov=True,
        ),
        NsTd(),
        DroIgr(),
        BpUcb(),
        Adapt(
            id="fixedw",
            name="Fixed-W DRO",
            family="Posted price",
            fixed_eps=0.12,
            use_cusum=False,
            use_queue_in_radius=False,
        ),
        Empirical(),
    ]


def build_ablations() -> list[Method]:
    return [
        Adapt(id="adapt", name="Full ADAPT"),
        Adapt(
            id="fixed-eps0",
            name="Fixed radius ε=ε0",
            fixed_eps=PAPER["eps0"],
            use_cusum=False,
            use_queue_in_radius=False,
        ),
        Adapt(
            id="no-cusum",
            name="No residual CUSUM",
            gamma=0.0,
            use_cusum=False,
            use_queue_in_radius=True,
        ),
        Adapt(
            id="no-queue",
            name="No queue pressure",
            eta=0.0,
            use_queue_in_radius=False,
            use_cusum=True,
        ),
        Adapt(
            id="no-dro",
            name="Expected-value (no DRO)",
            fixed_eps=0.0,
            use_cusum=False,
            use_queue_in_radius=False,
        ),
        Adapt(id="no-lyapunov", name="No Lyapunov term", use_lyapunov=False),
        Adapt(
            id="concentration",
            name="Concentration-only radius",
            gamma=0.0,
            eta=0.0,
            use_cusum=False,
            use_queue_in_radius=False,
        ),
        Adapt(id="saa", name="SAA inner loop", extract="saa"),
        Adapt(id="broken-extract", name="Dual ascent → broken extraction", extract="myopic"),
    ]


def build_sensitivity() -> list[Method]:
    specs = [
        ("default", "Default", {}),
        ("beta-0.1", "β=0.1", {"beta": 0.1}),
        ("gamma-0.1", "γ=0.1", {"gamma": 0.1}),
        ("eta-0", "η=0", {"eta": 0.0, "use_queue_in_radius": False}),
        ("V-20", "V=20", {"V": 20}),
        ("V-100", "V=100", {"V": 100}),
        ("eps-0.02", "ε0=0.02", {"eps0": 0.02}),
        ("eps-0.10", "ε0=0.10", {"eps0": 0.1}),
    ]
    return [Adapt(id=i, name=n, **patch) for i, n, patch in specs]


def apply_decision(slot: Slot, decision: Decision) -> Outcome:
    if decision.mode == "auction" and decision.pay is not None:
        return match_auction(slot, decision.pay)
    return match_posted(slot, decision.price)
