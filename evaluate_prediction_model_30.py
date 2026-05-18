import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from evaluate_zone_mpc_30 import build_scenario, patient_names
from simglucose.actuator.pump import InsulinPump
from simglucose.controller.mpc_ctrller import MPCController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv

def run_patient_simulation(patient_name, args, start_time):
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName(args.sensor, seed=args.seed + patient_names().index(patient_name))
    pump = InsulinPump.withName(args.pump)
    scenario = build_scenario(start_time)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    controller = MPCController(
        variant=args.variant,
        announce_meals=False,
        use_iob_constraint=args.use_iob_constraint,
        max_insulin_units_per_sample=args.max_insulin_units_per_sample,
        max_insulin_tdi_fraction=args.max_insulin_tdi_fraction,
        model_gain_factor=args.model_gain_factor,
        r_plus_scale=args.r_plus_scale,
        meal_bolus_scale=1.0,
        enable_low_glucose_suspend=args.enable_low_glucose_suspend,
        suspend_glucose=args.suspend_glucose,
        predictive_suspend_glucose=args.predictive_suspend_glucose,
        predictive_suspend_velocity=args.predictive_suspend_velocity,
    )

    obs, reward, done, info = env.reset()
    decision_rows = []

    while env.time < start_time + timedelta(hours=args.hours):
        action = controller.policy(obs, reward, done, **info)
        decision_rows.append(
            {
                "patient": patient_name,
                "group": patient_name.split("#", 1)[0],
                "time": info["time"],
                "bg": float(info["bg"]),
                "cgm": float(obs.CGM),
                "u_dev": float(controller.last_u_dev),
                "xhat0": float(controller.xhat[0]),
                "xhat1": float(controller.xhat[1]),
                "xhat2": float(controller.xhat[2]),
            }
        )
        obs, reward, done, info = env.step(action)

    decision_df = pd.DataFrame(decision_rows)
    return {
        "patient": patient_name,
        "decision_df": decision_df,
        "final_bg": float(info["bg"]),
        "A": np.array(controller.A, dtype=float),
        "B": np.array(controller.B, dtype=float),
        "Cy": np.array(controller.Cy, dtype=float),
        "observer_gain": np.array(controller.observer_gain, dtype=float),
        "ys": float(controller.ys),
        "sample_time": float(env.sample_time),
    }


def evaluate_prediction(sim_result, horizon_steps, warmup_steps, measurement_col):
    df = sim_result["decision_df"].copy()
    decision_bg = np.r_[df["bg"].to_numpy(dtype=float), sim_result["final_bg"]]
    u_dev = df["u_dev"].to_numpy(dtype=float)
    A = sim_result["A"]
    B = sim_result["B"]
    Cy = sim_result["Cy"]
    observer_gain = sim_result["observer_gain"]
    ys = sim_result["ys"]
    measurement = df[measurement_col].to_numpy(dtype=float)
    xhat = []
    state = None

    for i in range(len(df)):
        y = measurement[i] - ys
        if state is None:
            state = np.array([y, y, y], dtype=float)
        else:
            x_pred = A @ state + B * u_dev[i - 1]
            innovation = y - float(Cy @ x_pred)
            state = x_pred + observer_gain * innovation
        xhat.append(state.copy())

    xhat = np.asarray(xhat, dtype=float)
    rows = []

    for i in range(warmup_steps, len(df)):
        max_h = min(horizon_steps, len(decision_bg) - 1 - i)
        if max_h <= 0:
            continue
        x = xhat[i].copy()
        current_bg = decision_bg[i]
        base_time = pd.Timestamp(df.iloc[i]["time"])
        for h in range(1, max_h + 1):
            x = A @ x + B * u_dev[i + h - 1]
            predicted_bg = float(Cy @ x + ys)
            actual_bg = float(decision_bg[i + h])
            rows.append(
                {
                    "patient": sim_result["patient"],
                    "group": df.iloc[i]["group"],
                    "time": base_time,
                    "horizon_step": h,
                    "horizon_min": int(h * sim_result["sample_time"]),
                    "current_bg": float(current_bg),
                    "predicted_bg": predicted_bg,
                    "actual_bg": actual_bg,
                    "error": predicted_bg - actual_bg,
                    "abs_error": abs(predicted_bg - actual_bg),
                }
            )
    return pd.DataFrame(rows)


def summarize_predictions(pred_df):
    def r2_score(group):
        y_true = group["actual_bg"].to_numpy(dtype=float)
        y_pred = group["predicted_bg"].to_numpy(dtype=float)
        denom = np.sum((y_true - y_true.mean()) ** 2)
        if denom <= 0:
            return np.nan
        return 1.0 - np.sum((y_true - y_pred) ** 2) / denom

    horizon_rows = []
    for horizon_step, group in pred_df.groupby("horizon_step", sort=True):
        corr = (
            float(
                np.corrcoef(
                    group["actual_bg"].to_numpy(dtype=float),
                    group["predicted_bg"].to_numpy(dtype=float),
                )[0, 1]
            )
            if len(group) > 1
            else np.nan
        )
        horizon_rows.append(
            {
                "horizon_step": int(horizon_step),
                "horizon_min": int(group["horizon_min"].iloc[0]),
                "n_pairs": int(len(group)),
                "rmse": float(np.sqrt(np.mean(group["error"] ** 2))),
                "mae": float(np.mean(group["abs_error"])),
                "bias": float(np.mean(group["error"])),
                "r2": float(r2_score(group)),
                "corr": corr,
            }
        )
    horizon_summary = pd.DataFrame(horizon_rows)

    patient_rows = []
    for (patient, group_name, horizon_step), group in pred_df.groupby(
        ["patient", "group", "horizon_step"], sort=True
    ):
        patient_rows.append(
            {
                "patient": patient,
                "group": group_name,
                "horizon_step": int(horizon_step),
                "horizon_min": int(group["horizon_min"].iloc[0]),
                "n_pairs": int(len(group)),
                "rmse": float(np.sqrt(np.mean(group["error"] ** 2))),
                "mae": float(np.mean(group["abs_error"])),
                "bias": float(np.mean(group["error"])),
            }
        )
    patient_summary = pd.DataFrame(patient_rows)

    return horizon_summary, patient_summary


def plot_poincare(pred_df, horizon_summary, output_path):
    fig, axes = plt.subplots(
        3, 3, figsize=(14, 14), sharex=True, sharey=True, constrained_layout=True
    )
    axes = axes.ravel()
    x_min = float(np.floor(pred_df["current_bg"].min() / 10.0) * 10.0)
    x_max = float(
        np.ceil(
            max(
                pred_df["current_bg"].max(),
                pred_df["actual_bg"].max(),
                pred_df["predicted_bg"].max(),
            )
            / 10.0
        )
        * 10.0
    )

    for idx, horizon_step in enumerate(range(1, 10)):
        ax = axes[idx]
        subset = pred_df.loc[pred_df["horizon_step"] == horizon_step]
        summary = horizon_summary.loc[horizon_summary["horizon_step"] == horizon_step].iloc[0]
        ax.scatter(
            subset["current_bg"],
            subset["actual_bg"],
            s=5,
            alpha=0.12,
            color="#6c757d",
            label="Actual" if idx == 0 else None,
        )
        ax.scatter(
            subset["current_bg"],
            subset["predicted_bg"],
            s=5,
            alpha=0.12,
            color="#1565c0",
            label="Predicted" if idx == 0 else None,
        )
        ax.plot([x_min, x_max], [x_min, x_max], "--", color="#d0d0d0", linewidth=1.0)
        ax.set_title(
            f"+{int(summary['horizon_min'])} min\n"
            f"RMSE={summary['rmse']:.1f}  MAE={summary['mae']:.1f}"
        )
        ax.grid(True, alpha=0.2)

    for ax in axes[6:9]:
        ax.set_xlabel("Current BG (mg/dL)")
    for ax in axes[0:9:3]:
        ax.set_ylabel("Future BG (mg/dL)")

    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Prediction-Model Poincare Plots Across the 45 min Horizon", fontsize=18)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate prediction accuracy of the 3-state Zone-MPC model on 30 simglucose patients."
    )
    parser.add_argument("--variant", default="adaptive", choices=["previous", "velocity", "adaptive"])
    parser.add_argument("--hours", type=float, default=18.0)
    parser.add_argument("--sensor", default="GuardianRT")
    parser.add_argument("--pump", default="Insulet")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="output/prediction_accuracy_30")
    parser.add_argument("--warmup-steps", type=int, default=9)
    parser.add_argument("--measurement", choices=["bg", "cgm"], default="bg")
    parser.add_argument("--use-iob-constraint", action="store_true")
    parser.add_argument("--max-insulin-units-per-sample", type=float, default=1.0)
    parser.add_argument("--max-insulin-tdi-fraction", type=float, default=None)
    parser.add_argument("--model-gain-factor", type=float, default=1.0)
    parser.add_argument("--r-plus-scale", type=float, default=1.0)
    parser.add_argument("--enable-low-glucose-suspend", action="store_true")
    parser.add_argument("--suspend-glucose", type=float, default=80.0)
    parser.add_argument("--predictive-suspend-glucose", type=float, default=100.0)
    parser.add_argument("--predictive-suspend-velocity", type=float, default=0.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = datetime(2026, 1, 1, 7, 0)
    all_pred_rows = []
    sim_overview_rows = []

    for patient_name in patient_names():
        sim_result = run_patient_simulation(patient_name, args, start_time)
        pred_df = evaluate_prediction(
            sim_result,
            horizon_steps=9,
            warmup_steps=max(int(args.warmup_steps), 0),
            measurement_col=args.measurement,
        )
        all_pred_rows.append(pred_df)
        sim_overview_rows.append(
            {
                "patient": patient_name,
                "group": patient_name.split("#", 1)[0],
                "decision_points": int(len(sim_result["decision_df"])),
                "sample_time_min": float(sim_result["sample_time"]),
                "final_bg": float(sim_result["final_bg"]),
            }
        )
        print(
            f"{patient_name:15s} "
            f"points={len(sim_result['decision_df']):3d} "
            f"final_bg={sim_result['final_bg']:.1f}"
        )

    pred_df = pd.concat(all_pred_rows, ignore_index=True)
    horizon_summary, patient_summary = summarize_predictions(pred_df)
    sim_overview = pd.DataFrame(sim_overview_rows)

    pred_path = output_dir / "prediction_pairs.csv"
    horizon_path = output_dir / "horizon_summary.csv"
    patient_path = output_dir / "patient_horizon_summary.csv"
    overview_path = output_dir / "simulation_overview.csv"
    figure_path = output_dir / "poincare_prediction_horizon.png"

    pred_df.to_csv(pred_path, index=False)
    horizon_summary.to_csv(horizon_path, index=False)
    patient_summary.to_csv(patient_path, index=False)
    sim_overview.to_csv(overview_path, index=False)
    plot_poincare(pred_df, horizon_summary, figure_path)

    print()
    print(horizon_summary.to_string(index=False))
    print()
    print(f"prediction pairs: {pred_path}")
    print(f"horizon summary: {horizon_path}")
    print(f"patient summary: {patient_path}")
    print(f"overview: {overview_path}")
    print(f"poincare plot: {figure_path}")


if __name__ == "__main__":
    main()
