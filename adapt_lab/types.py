"""Slot, queue, and outcome records. Lists are plain Python so the trace is inspectable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Slot:
    q_values: list[list[float]]
    c: list[float]
    a: list[int]
    e: list[float]
    g: list[int]
    hour: int


@dataclass
class Queues:
    b: float = 0.0
    e: float = 0.0
    f: float = 0.0


@dataclass
class Outcome:
    completed: int
    payment: float
    unfinished: int
    energy: float
    deficit: float
    utility: float
    cost: float
    by_group: list[int]
    by_task_q: list[int]
    price: float
    n_willing: int


@dataclass
class Decision:
    price: float
    mode: str
    pay: list[float] | None = None
    diag: dict | None = None


@dataclass
class StepDiag:
    eps: float
    S: float
    conc: float
    s_term: float
    q_term: float
    price: float
    alpha_emp: float
    alpha_lb: float
    alpha_ub: float
    phi: float
    norm: float
    detections: int
    n_buf: int


def task_count(slot: Slot) -> int:
    return sum(len(q) for q in slot.q_values)


def empty_queues() -> Queues:
    return Queues()
