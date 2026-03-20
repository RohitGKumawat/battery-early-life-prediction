from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)



def simulate_cell(cell_id: str, early_cycles: int, points_per_curve: int = 120) -> tuple[pd.DataFrame, dict]:
    latent_quality = RNG.uniform(0.0, 1.0)
    manufacturing_noise = RNG.normal(0.0, 0.03)
    fast_charge_stress = RNG.uniform(0.8, 1.2)

    eol_cycle = int(
        350
        + 1400 * latent_quality
        - 180 * fast_charge_stress
        + 60 * RNG.normal()
    )
    eol_cycle = int(np.clip(eol_cycle, 180, 2200))

    rows = []
    nominal_capacity = 2.8 + 0.35 * latent_quality + RNG.normal(0.0, 0.03)
    base_resistance = 0.030 + 0.020 * (1.0 - latent_quality)

    for cycle in range(1, early_cycles + 1):
        aging_frac = cycle / max(eol_cycle, 1)
        temp_rise = 2.0 + 10.0 * (1 - latent_quality) * aging_frac * 30 + RNG.normal(0, 0.3)
        discharge_capacity = nominal_capacity * (1.0 - 0.10 * aging_frac * 30 + manufacturing_noise / 10)
        discharge_capacity = max(discharge_capacity, nominal_capacity * 0.75)
        internal_resistance = base_resistance * (1.0 + 0.9 * aging_frac * 30 + 0.2 * fast_charge_stress)

        q = np.linspace(0.0, discharge_capacity, points_per_curve)
        t = np.linspace(0.0, 3600.0, points_per_curve)
        soc = q / max(discharge_capacity, 1e-6)

        discharge_voltage = (
            4.18
            - 0.95 * soc
            - 0.08 * np.log1p(4 * soc)
            - 0.18 * internal_resistance * 10
            - 0.04 * (1 - latent_quality) * soc**2
            + RNG.normal(0.0, 0.004, size=points_per_curve)
        )
        discharge_current = -1.5 * fast_charge_stress + RNG.normal(0.0, 0.02, size=points_per_curve)
        discharge_temp = 25.0 + temp_rise + 2.0 * soc + RNG.normal(0.0, 0.2, size=points_per_curve)

        charge_voltage = (
            3.05
            + 1.05 * soc
            + 0.05 * np.sqrt(soc + 1e-6)
            + 0.20 * internal_resistance * 8
            + 0.03 * (1 - latent_quality) * soc**1.5
            + RNG.normal(0.0, 0.004, size=points_per_curve)
        )
        charge_current = 1.4 * fast_charge_stress + RNG.normal(0.0, 0.02, size=points_per_curve)
        charge_temp = 24.5 + temp_rise + 1.5 * soc + RNG.normal(0.0, 0.2, size=points_per_curve)

        for i in range(points_per_curve):
            rows.append(
                {
                    "cell_id": cell_id,
                    "cycle_index": cycle,
                    "step_type": "discharge",
                    "time_s": float(t[i]),
                    "voltage_v": float(discharge_voltage[i]),
                    "current_a": float(discharge_current[i]),
                    "capacity_ah": float(q[i]),
                    "temperature_c": float(discharge_temp[i]),
                    "internal_resistance_ohm": float(internal_resistance),
                    "soc": float(soc[i]),
                }
            )
            rows.append(
                {
                    "cell_id": cell_id,
                    "cycle_index": cycle,
                    "step_type": "charge",
                    "time_s": float(t[i]),
                    "voltage_v": float(charge_voltage[i]),
                    "current_a": float(charge_current[i]),
                    "capacity_ah": float(q[i]),
                    "temperature_c": float(charge_temp[i]),
                    "internal_resistance_ohm": float(internal_resistance),
                    "soc": float(soc[i]),
                }
            )

    label_row = {"cell_id": cell_id, "eol_cycle": eol_cycle}
    return pd.DataFrame(rows), label_row



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a demo battery dataset.")
    parser.add_argument("--n-cells", type=int, default=80)
    parser.add_argument("--early-cycles", type=int, default=40)
    parser.add_argument("--points-per-curve", type=int, default=120)
    parser.add_argument("--output-dir", type=str, default="data/sample")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_curves = []
    labels = []
    for idx in range(args.n_cells):
        curves, label = simulate_cell(
            cell_id=f"CELL_{idx:04d}",
            early_cycles=args.early_cycles,
            points_per_curve=args.points_per_curve,
        )
        all_curves.append(curves)
        labels.append(label)

    curves_df = pd.concat(all_curves, ignore_index=True)
    labels_df = pd.DataFrame(labels)

    curves_df.to_csv(output_dir / "battery_curves.csv", index=False)
    labels_df.to_csv(output_dir / "battery_labels.csv", index=False)
    print(f"Saved curves to {output_dir / 'battery_curves.csv'}")
    print(f"Saved labels to {output_dir / 'battery_labels.csv'}")


if __name__ == "__main__":
    main()
