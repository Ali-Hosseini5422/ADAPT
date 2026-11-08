"""Defaults from the manuscript hyperparameter table (Table IV)."""

PAPER = {
    "eps0": 0.05,
    "beta": 0.3,
    "gamma": 0.4,
    "eta": 0.3,
    "V": 50,
    "Qmax": 100,
    "nu": 0.1,
    "tau": 5,
    "W": 500,
    "K": 50,
    # Step size for the K-step projected dual descent (not numerically specified).
    "etaLambda": 0.2,
    "priceGrid": 31,
    "cMin": 1.0,
    "cMax": 10.0,
    "taskValue": 9.0,
    "Ebar": 7.5,
    "Dbar": 0.2,
    "rhoUnfinished": 2.0,
    "groups": 4,
}

MAX_N = 24
