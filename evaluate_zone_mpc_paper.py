import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.mpc_ctrller import MPCController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj


def adult_patients():
    return [f"adult#{i:03d}" for i in range(1, 11)]


def fixed_2019_scenario(start_time):
    return CustomScenario(
        start_time,
        [
            (timedelta(hours=1), 50),
            (timedelta(hours=5), 75),
            (timedelta(hours=12), 75),
        ],
    )


def hybrid_scenario_a(start_time, days, seed):
    rng = np.random.default_rng(seed)
    scenario = []
    regular_times = [8.0, 13.0, 19.0]
    regular_amounts = [50.0, 75.0, 75.0]
    snack_times = [9.5, 15.0, 21.5]
    snack_amounts = [10.0, 30.0, 20.0]

    for day in range(days):
        for hour, amount in zip(regular_times, regular_amounts):
            if rng.random() <= 0.75:
                scenario.append(_random_meal_time(day, hour, amount, rng))
        for hour, amount in zip(snack_times, snack_amounts):
            if rng.random() <= 0.30:
                scenario.append(_random_meal_time(day, hour, amount, rng))

    scenario.sort(key=lambda item: item[0])
    return CustomScenario(start_time, scenario)


def _random_meal_time(day, hour, amount, rng):
    minute = int(round(rng.normal(hour * 60.0, 60.0)))
    minute = int(np.clip(minute, 0, 24 * 60 - 1))
    carbs = max(float(rng.normal(amount, 0.40 * amount)), 0.0)
    return timedelta(days=day, minutes=minute), carbs


def build_scenario(name, start_time, seed):
    if name == "fixed_2019":
        return fixed_2019_scenario(start_time), timedelta(hours=24)
    if name == "hybrid_a":
        return hybrid_scenario_a(start_time, days=2, seed=seed), timedelta(days=2)
    raise ValueError(f"unknown scenario: {name}")


def summarize(df, patient, mode, repeat, solve_times, solve_successes):
    bg = df.BG.dropna()
    cgm = df.CGM.dropna()
    insulin = df.insulin.dropna()
    solve_times = np.asarray(solve_times, dtype=float)
    solve_successes = np.asarray(solve_successes, dtype=bool)
    return {
        "patient": patient,
        "mode": mode,
        "repeat": repeat,
        "n_steps": int(len(bg)),
        "TBR2_lt54": float((bg < 54).mean() * 100),
        "TBR1_lt70": float((bg < 70).mean() * 100),
        "TITR_70_140": float(((bg >= 70) & (bg <= 140)).mean() * 100),
        "TIR_70_180": float(((bg >= 70) & (bg <= 180)).mean() * 100),
        "TAR1_gt180": float((bg > 180).mean() * 100),
        "TAR2_gt250": float((bg > 250).mean() * 100),
        "min_bg": float(bg.min()),
        "mean_bg": float(bg.mean()),
        "max_bg": float(bg.max()),
        "sd_bg": float(bg.std(ddof=0)),
        "mean_cgm": float(cgm.mean()),
        "total_insulin_u": float(insulin.sum() * 5.0),
        "mean_solve_ms": float(solve_times.mean() * 1000) if len(solve_times) else np.nan,
        "p95_solve_ms": float(np.percentile(solve_times, 95) * 1000) if len(solve_times) else np.nan,
        "max_solve_ms": float(solve_times.max() * 1000) if len(solve_times) else np.nan,
        "solve_success_rate": float(solve_successes.mean() * 100) if len(solve_successes) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Paper-style adult Zone-MPC evaluation.")
    parser.add_argument("--scenario", choices=["fixed_2019", "hybrid_a"], default="fixed_2019")
    parser.add_argument("--modes", nargs="+", choices=["announced", "unannounced"], default=["unannounced"])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sensor", default="GuardianRT")
    parser.add_argument("--pump", default="Insulet")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="output/zone_mpc_paper")
    parser.add_argument("--use-iob-constraint", action="store_true")
    parser.add_argument("--max-insulin-units-per-sample", type=float, default=1.0)
    parser.add_argument("--max-insulin-tdi-fraction", type=float, default=None)
    parser.add_argument("--model-gain-factor", type=float, default=1.0)
    parser.add_argument("--r-plus-scale", type=float, default=1.0)
    parser.add_argument("--meal-bolus-scale", type=float, default=1.0)
    parser.add_argument("--enable-low-glucose-suspend", action="store_true")
    parser.add_argument("--suspend-glucose", type=float, default=80.0)
    parser.add_argument("--predictive-suspend-glucose", type=float, default=100.0)
    parser.add_argument("--predictive-suspend-velocity", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime(2026, 1, 1, 7, 0)
    rows = []

    for mode in args.modes:
        mode_dir = output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        for repeat in range(args.repeats):
            for idx, patient_name in enumerate(adult_patients()):
                run_seed = args.seed + repeat * 1000 + idx
                patient = T1DPatient.withName(patient_name)
                sensor = CGMSensor.withName(args.sensor, seed=run_seed)
                pump = InsulinPump.withName(args.pump)
                scenario, sim_time = build_scenario(args.scenario, start_time, run_seed)
                env = T1DSimEnv(patient, sensor, pump, scenario)
                controller = MPCController(
                    variant="adaptive",
                    announce_meals=mode == "announced",
                    use_iob_constraint=args.use_iob_constraint,
                    max_insulin_units_per_sample=args.max_insulin_units_per_sample,
                    max_insulin_tdi_fraction=args.max_insulin_tdi_fraction,
                    model_gain_factor=args.model_gain_factor,
                    r_plus_scale=args.r_plus_scale,
                    meal_bolus_scale=args.meal_bolus_scale,
                    enable_low_glucose_suspend=args.enable_low_glucose_suspend,
                    suspend_glucose=args.suspend_glucose,
                    predictive_suspend_glucose=args.predictive_suspend_glucose,
                    predictive_suspend_velocity=args.predictive_suspend_velocity,
                )
                sim = SimObj(env, controller, sim_time, animate=False)
                sim.simulate()
                df = sim.results()
                out_name = f"repeat{repeat:02d}_{patient_name}.csv"
                df.to_csv(mode_dir / out_name)
                row = summarize(
                    df,
                    patient_name,
                    mode,
                    repeat,
                    controller.solve_times,
                    controller.solve_successes,
                )
                rows.append(row)
                print(
                    f"{mode:11s} r={repeat:02d} {patient_name:10s} "
                    f"mean={row['mean_bg']:.1f} tir={row['TIR_70_180']:.1f}% "
                    f"titr={row['TITR_70_140']:.1f}% <70={row['TBR1_lt70']:.1f}%"
                )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = (
        summary.groupby("mode")
        .agg(
            runs=("patient", "count"),
            TBR2_lt54=("TBR2_lt54", "mean"),
            TBR1_lt70=("TBR1_lt70", "mean"),
            TITR_70_140=("TITR_70_140", "mean"),
            TIR_70_180=("TIR_70_180", "mean"),
            TAR1_gt180=("TAR1_gt180", "mean"),
            TAR2_gt250=("TAR2_gt250", "mean"),
            mean_bg=("mean_bg", "mean"),
            sd_bg=("sd_bg", "mean"),
            min_bg=("min_bg", "min"),
            worst_max_bg=("max_bg", "max"),
            total_insulin_u=("total_insulin_u", "mean"),
            p95_solve_ms=("p95_solve_ms", "mean"),
            solve_success_rate=("solve_success_rate", "mean"),
        )
        .reset_index()
    )
    aggregate_path = output_dir / "aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)
    print()
    print(aggregate.to_string(index=False))
    print(f"summary: {summary_path}")
    print(f"aggregate: {aggregate_path}")


if __name__ == "__main__":
    main()
