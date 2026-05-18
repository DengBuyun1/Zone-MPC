import argparse
from datetime import datetime, timedelta
from pathlib import Path

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.mpc_ctrller import MPCController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj


def build_scenario(start_time):
    return CustomScenario(
        start_time,
        [
            (timedelta(hours=1), 50),
            (timedelta(hours=6), 75),
            (timedelta(hours=12), 75),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="Run a zone-MPC simglucose demo.")
    parser.add_argument("--patient", default="adult#001")
    parser.add_argument("--sensor", default="GuardianRT")
    parser.add_argument("--pump", default="Insulet")
    parser.add_argument(
        "--variant",
        default="adaptive",
        choices=["previous", "velocity", "adaptive"],
        help="previous=plain zone MPC, velocity=2018 MPC, adaptive=2019 MPC.",
    )
    parser.add_argument("--hours", type=float, default=18.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--announce-meals", action="store_true")
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
    parser.add_argument("--output", default="output/zone_mpc_demo.csv")
    args = parser.parse_args()

    start_time = datetime(2026, 1, 1, 7, 0)
    patient = T1DPatient.withName(args.patient)
    sensor = CGMSensor.withName(args.sensor, seed=args.seed)
    pump = InsulinPump.withName(args.pump)
    scenario = build_scenario(start_time)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    controller = MPCController(
        variant=args.variant,
        announce_meals=args.announce_meals,
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
    results = sim.results()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output)

    print(f"saved: {output}")
    print(f"min BG: {results.BG.min():.1f} mg/dL")
    print(f"mean BG: {results.BG.mean():.1f} mg/dL")
    print(f"max BG: {results.BG.max():.1f} mg/dL")
    print(f"time <70: {(results.BG < 70).mean() * 100:.2f}%")
    print(f"time 70-180: {(((results.BG >= 70) & (results.BG <= 180)).mean() * 100):.2f}%")


if __name__ == "__main__":
    main()
