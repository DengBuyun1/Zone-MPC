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


def patient_names():
    return (
        [f"adolescent#{i:03d}" for i in range(1, 11)]
        + [f"adult#{i:03d}" for i in range(1, 11)]
        + [f"child#{i:03d}" for i in range(1, 11)]
    )


def build_scenario(start_time):
    return CustomScenario(
        start_time,
        [
            (timedelta(hours=1), 50),
            (timedelta(hours=6), 75),
            (timedelta(hours=12), 75),
        ],
    )


def summarize(df, patient, variant, solve_times, solve_successes):
    bg = df.BG.dropna()
    cgm = df.CGM.dropna()
    insulin = df.insulin.dropna()
    solve_times = np.asarray(solve_times, dtype=float)
    solve_successes = np.asarray(solve_successes, dtype=bool)
    return {
        "patient": patient,
        "group": patient.split("#", 1)[0],
        "variant": variant,
        "n_steps": int(len(bg)),
        "min_bg": float(bg.min()),
        "mean_bg": float(bg.mean()),
        "max_bg": float(bg.max()),
        "sd_bg": float(bg.std(ddof=0)),
        "time_lt_54": float((bg < 54).mean() * 100),
        "time_lt_70": float((bg < 70).mean() * 100),
        "time_70_180": float(((bg >= 70) & (bg <= 180)).mean() * 100),
        "time_gt_180": float((bg > 180).mean() * 100),
        "time_gt_250": float((bg > 250).mean() * 100),
        "mean_cgm": float(cgm.mean()),
        "total_insulin_u": float(insulin.sum() * 5.0),
        "mean_solve_ms": float(solve_times.mean() * 1000) if len(solve_times) else np.nan,
        "p95_solve_ms": float(np.percentile(solve_times, 95) * 1000) if len(solve_times) else np.nan,
        "max_solve_ms": float(solve_times.max() * 1000) if len(solve_times) else np.nan,
        "solve_success_rate": float(solve_successes.mean() * 100) if len(solve_successes) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate zone MPC on 30 simglucose patients.")
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["adaptive"],
        choices=["previous", "velocity", "adaptive"],
    )
    parser.add_argument("--hours", type=float, default=18.0)
    parser.add_argument("--sensor", default="GuardianRT")
    parser.add_argument("--pump", default="Insulet")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="output/zone_mpc_30")
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

    for variant in args.variants:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        for idx, patient_name in enumerate(patient_names()):
            patient = T1DPatient.withName(patient_name)
            sensor = CGMSensor.withName(args.sensor, seed=args.seed + idx)
            pump = InsulinPump.withName(args.pump)
            scenario = build_scenario(start_time)
            env = T1DSimEnv(patient, sensor, pump, scenario)
            controller = MPCController(
                variant=variant,
                announce_meals=False,
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
            sim = SimObj(env, controller, timedelta(hours=args.hours), animate=False)
            sim.simulate()
            df = sim.results()
            df.to_csv(variant_dir / f"{patient_name}.csv")
            row = summarize(
                df,
                patient_name,
                variant,
                controller.solve_times,
                controller.solve_successes,
            )
            rows.append(row)
            print(
                f"{variant:8s} {patient_name:15s} "
                f"mean={row['mean_bg']:.1f} tir={row['time_70_180']:.1f}% "
                f"<70={row['time_lt_70']:.1f}% p95solve={row['p95_solve_ms']:.1f}ms"
            )

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    aggregate = (
        summary.groupby("variant")
        .agg(
            patients=("patient", "count"),
            mean_bg=("mean_bg", "mean"),
            median_mean_bg=("mean_bg", "median"),
            min_bg=("min_bg", "min"),
            worst_max_bg=("max_bg", "max"),
            time_lt_54=("time_lt_54", "mean"),
            time_lt_70=("time_lt_70", "mean"),
            time_70_180=("time_70_180", "mean"),
            time_gt_180=("time_gt_180", "mean"),
            time_gt_250=("time_gt_250", "mean"),
            total_insulin_u=("total_insulin_u", "mean"),
            mean_solve_ms=("mean_solve_ms", "mean"),
            p95_solve_ms=("p95_solve_ms", "mean"),
            max_solve_ms=("max_solve_ms", "max"),
            solve_success_rate=("solve_success_rate", "mean"),
        )
        .reset_index()
    )
    aggregate_path = output_dir / "aggregate.csv"
    aggregate.to_csv(aggregate_path, index=False)

    group_aggregate = (
        summary.groupby(["variant", "group"])
        .agg(
            patients=("patient", "count"),
            mean_bg=("mean_bg", "mean"),
            min_bg=("min_bg", "min"),
            worst_max_bg=("max_bg", "max"),
            time_lt_54=("time_lt_54", "mean"),
            time_lt_70=("time_lt_70", "mean"),
            time_70_180=("time_70_180", "mean"),
            time_gt_180=("time_gt_180", "mean"),
            time_gt_250=("time_gt_250", "mean"),
            total_insulin_u=("total_insulin_u", "mean"),
            p95_solve_ms=("p95_solve_ms", "mean"),
            solve_success_rate=("solve_success_rate", "mean"),
        )
        .reset_index()
    )
    group_aggregate_path = output_dir / "aggregate_by_group.csv"
    group_aggregate.to_csv(group_aggregate_path, index=False)
    print()
    print(aggregate.to_string(index=False))
    print()
    print(group_aggregate.to_string(index=False))
    print(f"summary: {summary_path}")
    print(f"aggregate: {aggregate_path}")
    print(f"aggregate by group: {group_aggregate_path}")


if __name__ == "__main__":
    main()
