"""
Assign willing workers to tasks.

Quadrant-local matches first (coverage fairness), then spillover to the
highest-value remaining tasks. Posted prices pay `price` per completed task.
Auction callers pass an explicit payment total.
"""

from __future__ import annotations

from .constants import PAPER
from .types import Outcome, Slot, task_count


def match_posted(slot: Slot, price: float) -> Outcome:
    willing = [i for i in range(len(slot.c)) if slot.a[i] == 1 and slot.c[i] <= price + 1e-12]
    willing.sort(key=lambda i: slot.c[i])
    return _settle(slot, willing, price, None)


def match_auction(slot: Slot, pay: list[float]) -> Outcome:
    willing: list[int] = []
    payment = 0.0
    for i, pmt in enumerate(pay):
        if pmt > 0:
            willing.append(i)
            payment += pmt
    pseudo = payment / len(willing) if willing else 0.0
    return _settle(slot, willing, pseudo, payment)


def _settle(slot: Slot, willing: list[int], price: float, payment_override: float | None) -> Outcome:
    buckets: list[list[int]] = [[], [], [], []]
    for i in willing:
        g = slot.g[i] if i < len(slot.g) else 0
        buckets[g].append(i)
    remain_w = [len(b) for b in buckets]
    taken_local = [0, 0, 0, 0]
    by_group = [0, 0, 0, 0]
    by_task_q = [0, 0, 0, 0]
    value = 0.0
    energy = 0.0
    completed = 0

    for q in range(4):
        tasks = slot.q_values[q]
        m = min(remain_w[q], len(tasks))
        for t in range(m):
            value += tasks[t]
        for t in range(m):
            energy += slot.e[buckets[q][t]]
        remain_w[q] -= m
        taken_local[q] = m
        by_group[q] += m
        by_task_q[q] += m
        completed += m

    spill_w: list[int] = []
    for q in range(4):
        b = buckets[q]
        start = len(b) - remain_w[q]
        for t in range(start, len(b)):
            spill_w.append(b[t])
    spill: list[tuple[int, float]] = []
    for q in range(4):
        tasks = slot.q_values[q]
        for t in range(taken_local[q], len(tasks)):
            spill.append((q, tasks[t]))
    spill.sort(key=lambda item: -item[1])
    m = min(len(spill_w), len(spill))
    for i in range(m):
        q, v = spill[i]
        value += v
        energy += slot.e[spill_w[i]]
        gi = slot.g[spill_w[i]] if spill_w[i] < len(slot.g) else 0
        by_group[gi] += 1
        by_task_q[q] += 1
        completed += 1

    arrivals = task_count(slot)
    unfinished = max(0, arrivals - completed)
    payment = price * completed if payment_override is None else payment_override
    utility = value - payment
    cost = payment + PAPER["rhoUnfinished"] * unfinished
    if completed <= 0:
        deficit = 1.0
    else:
        deficit = 0.0
        for q in range(4):
            deficit += max(0.0, 0.25 - by_group[q] / completed)
    return Outcome(
        completed=completed,
        payment=payment,
        unfinished=unfinished,
        energy=energy,
        deficit=deficit,
        utility=utility,
        cost=cost,
        by_group=by_group,
        by_task_q=by_task_q,
        price=price,
        n_willing=len(willing),
    )
