# Zone MPC tuning notes

The current implementation is a paper-informed reproduction on the local
`simglucose` environment, not a clinical controller. The main gap versus the
Gondhalekar papers is the complete insulin-on-board history constraint and pump
carry-over logic.

## What needed adjustment

The original day upper bound of `1 U / 5 min` is too aggressive for the mixed
30-patient `simglucose` cohort, especially children. Several child patients hit
the day upper bound after meals, accumulated IOB, then entered prolonged
hypoglycemia even after the controller stopped insulin.

The useful adjustment is to keep the paper structure, but scale the maximum
per-sample insulin by TDI:

```powershell
python evaluate_zone_mpc_30.py --variants adaptive --hours 18 `
  --output-dir output\zone_mpc_30_tune_cap0075 `
  --use-iob-constraint --max-insulin-tdi-fraction 0.0075
```

## 30-patient 18 h results

All runs use the adaptive 2019 penalty, the 2016 observer, unannounced meals,
and the approximate local IOB constraint.

| Setting | Mean BG | <70 | 70-180 | >250 | p95 solve |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline, no TDI cap | 150.3 | 15.6% | 49.3% | 7.6% | 17.3 ms |
| `0.0100*TDI` cap | 176.3 | 3.9% | 53.4% | 12.4% | 17.1 ms |
| `0.0085*TDI` cap | 181.1 | 2.0% | 53.4% | 13.2% | 16.0 ms |
| `0.0075*TDI` cap | 185.6 | 1.2% | 52.2% | 14.3% | 16.0 ms |
| `0.0050*TDI` cap | 202.1 | 0.0% | 44.8% | 18.7% | 17.8 ms |
| `R+*0.5`, `0.0075*TDI` cap | 179.3 | 1.4% | 57.4% | 13.1% | 16.9 ms |

Recommended starting point for the 30-patient mixed cohort:

```powershell
python evaluate_zone_mpc_30.py --variants adaptive --hours 18 `
  --output-dir output\zone_mpc_30_tuned `
  --use-iob-constraint --max-insulin-tdi-fraction 0.0075
```

If hyperglycemia is weighted more heavily and some low-glucose risk is
acceptable for research comparison, use `0.0085`.

## Paper-style adult results

The HyCPAP paper reports Zone-MPC on the UVA/Padova 10-adult cohort. To make
the local `simglucose` comparison closer to that setting, use
`evaluate_zone_mpc_paper.py`.

For the 2019 fixed unannounced protocol with adult#001-010, 24 h, meals
`[50,75,75] g`:

```powershell
python evaluate_zone_mpc_paper.py --scenario fixed_2019 --modes unannounced `
  --output-dir output\zone_mpc_paper_fixed_r05_cap0075 `
  --use-iob-constraint --r-plus-scale 0.5 `
  --max-insulin-tdi-fraction 0.0075 --enable-low-glucose-suspend
```

This gives `TIR=71.1%`, `TBR<70=0.0%`, `Mean BG=154.0 mg/dL`, close to the
2019 abstract values for adaptive Zone-MPC under unannounced meals
(`TIR=70.5%`, `Mean BG=153.8 mg/dL`).

For the HyCPAP paper's randomized Scenario A, adult#001-010, 2 days,
3 repeats, unannounced meals:

```powershell
python evaluate_zone_mpc_paper.py --scenario hybrid_a --modes unannounced `
  --repeats 3 --output-dir output\zone_mpc_paper_hybrid_a_r05_cap0075 `
  --use-iob-constraint --r-plus-scale 0.5 `
  --max-insulin-tdi-fraction 0.0075 --enable-low-glucose-suspend
```

Local result: `TIR=74.3%`, `TBR<70=0.9%`, `Mean BG=156.6 mg/dL`,
`SD BG=52.4 mg/dL`. The HyCPAP paper's Table II reports MPC unannounced
Scenario A as `TIR=72.8%`, `TBR<70=0.5%`, `Mean BG=160.3 mg/dL`,
`SD BG=52.5 mg/dL`.

For randomized Scenario A with announced meals, the local `simglucose`
carbohydrate ratios are too aggressive if the full paper bolus is used. A
local bolus scale of `0.75` plus the same TDI cap gives the closest tradeoff:

```powershell
python evaluate_zone_mpc_paper.py --scenario hybrid_a --modes announced `
  --output-dir output\zone_mpc_paper_hybrid_a_announced_bolus075_cap0065 `
  --use-iob-constraint --meal-bolus-scale 0.75 `
  --max-insulin-tdi-fraction 0.0065 --enable-low-glucose-suspend
```

Local result: `TIR=89.2%`, `TBR<70=0.9%`, `Mean BG=139.4 mg/dL`. The HyCPAP
paper's Table II reports MPC announced Scenario A as `TIR=89.2%`,
`TBR<70=0.1%`, `Mean BG=138.5 mg/dL`. The remaining mismatch is mostly a
small number of low-glucose episodes in adult#006 and adult#009.

## Raspberry Pi 5 feasibility

The optimization size is small (`Nu=5`, `Ny=9`). On this machine the tuned
adaptive controller has p95 solve time around `16 ms`, with all solves
successful in the 30-patient run. Even allowing a several-fold slowdown on a
Raspberry Pi 5, the solve time is far below the 5-minute control period.

For production-grade embedded work, replace SLSQP with a deterministic QP
solver after freezing the velocity/adaptive weights inside each MPC step.
