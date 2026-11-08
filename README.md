# ADAPT laboratory (Python)

Line-level port of the paired simulator for adaptive Wasserstein posted prices
with Lyapunov tracking, plus the developed rule ADAPT-Q. The manuscript’s
12–18% bands are analytical estimates. They are not targets, and this code
does not fit them.

Standard library only. Python 3.11 or newer.

## Check the dual identity

```bash
python3 -m adapt_lab --check
```

The Wasserstein acceptance bounds are a fractional knapsack. `--check` confirms
they match the Kantorovich dual of `ℓ = p · 1{a = 1, c ≤ p}`, then runs 60
synthetic slots.

## Reproduce a campaign

```bash
python3 -m adapt_lab --dataset synthetic --suite catalog --out synthetic.json
python3 -m adapt_lab --dataset synthetic --suite catalog --seeds 101 --horizon 400
python3 -m adapt_lab --dataset synthetic --suite ablations
python3 -m adapt_lab --dataset synthetic --suite sensitivity
python3 -m adapt_lab --dataset spatial --suite catalog
python3 -m adapt_lab --dataset tdrive --suite catalog --tdrive data/tdrive-derived.json
```

Seeds are 101, 202, 303, 404, 505. Every method on a seed shares one draw.
Posted-price methods do not see the current bids. Auctions do. The catalog
starts with ADAPT-Q.

## What one ADAPT-Q slot does

1. Draw tasks and workers. The price is computed from the window only.
2. `ε = min(0.08, ε0 + β √(ln(t+1) / n))`. The CUSUM does not widen the ball.
3. The window holds 200 slots. If the Page statistic exceeds τ = 5 the window
   resets to the current slot after the price is posted.
4. Search 61 prices up to the highest available cost in the window, and no
   higher. Score each by the worst-case drift-plus-penalty with V = 20.
   Predicted value is the sum of the highest task values.
5. Match workers with `a = 1` and `c ≤ p`, quadrant first, then spillover.

These four choices were frozen on synthetic seed 101. Seeds 202–505 and the
other two traces were not used to choose them.

## Paired results for the frozen rule

| Trace | ADAPT-Q utility | Coverage | Mean backlog | vs auction utility | vs Lyapunov coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Synthetic | 40577 ± 463 | 0.942 | 430 | −32.9% | +7.8% |
| T-Drive | 47203 ± 400 | 0.912 | 680 | −32.8% | +6.0% |
| Spatial | 121150 ± 5949 | 0.856 | 1435 | −22.8% | +5.3% |

A uniform posted price cannot reach the auction: the auction pays a critical
price, so the infra-marginal workers are not all paid the marginal one.
An offline posted price that knows each regime’s distribution and maximizes
utility reaches about 50435 on synthetic seed 101, against the auction’s 60485.
ADAPT-Q lands at 40460 on that seed.

`libm` can move the last bits of `log`, `exp`, `sin`, and `hypot` relative to
the engine that produced the tables. The algorithm is the same. A short horizon
should land within a fraction of a percent.

Utility is the value of completed tasks minus payments. Unfinished tasks enter
the backlog; they are not subtracted from utility.

T-Drive battery is not in the GPS sample. The proxy is a diurnal sinusoid, the
bin’s real inter-sample gap, and `N(0, 0.15)` noise per worker. Distance is
haversine to the nearest other reservoir fix in the bin.
