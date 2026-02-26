

from pathlib import Path
from src.io import prepare_inputs
import src.milp as milp
from datetime import datetime
import argparse
import csv

status_map = {
    1: "LOADED",
    2: "OPTIMAL",
    3: "INFEASIBLE",
    4: "INF_OR_UNBD",
    5: "UNBOUNDED",
    9: "TIME_LIMIT",
}


def _append_csv_row(csv_path: Path, row: dict) -> None:
    """Append one row to a CSV, creating it (with header) if it doesn't exist."""
    csv_path.parent.mkdir(exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def _write_run_summary_txt(path: Path, meta: dict, results: dict) -> None:
    path.parent.mkdir(exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}={v}\n")
        f.write("\n")

        f.write(f"status={status_map.get(results['status'], results['status'])}\n")
        f.write(f"solcount={results['solcount']}\n")
        if results["solcount"] > 0:
            f.write(f"objective={results['objective']}\n")
        f.write(f"runtime_seconds={results['runtime']}\n")


def run_one(instance: str, assignment: str, time_limit: int, use_preassigned_policy: bool) -> dict:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    # Load inputs
    inputs = prepare_inputs(
        input_data_xlsx=root / "data" / "Input_data" / "Input_Data.xlsx",
        n_pm_xlsx=root / "data" / "Input_data" / "n_pm.xlsx",
        l_pm_xlsx=root / "data" / "Input_data" / "l_pm.xlsx",
        n_f_xlsx=root / "data" / "Input_data" / "n_f.xlsx",
        distance_matrix_xlsx=root / "data" / "datasets" / f"distance_matrix_{instance}_storage_to_stations.xlsx",
        part_station_assignment_xlsx=root / "data" / "datasets" / f"part_station_assignment_{assignment}.xlsx",
    )

    # Set global variables
    milp.assignment_parts_stations = inputs["assignment_parts_stations"]
    milp.df = inputs["df"]
    milp.stations = inputs["stations"]
    milp.Fix_S = inputs["Fix_S"]

    # Solve
    results = milp.Model_1_TC(
        inputs["H"], inputs["muy_m"], inputs["v_m"], inputs["co_m"], inputs["Cap_b"], inputs["SC"],
        inputs["mr_rb"], inputs["r_a2"], inputs["r_ia"], inputs["nk_4"], inputs["r_a4"], inputs["m_i"],
        inputs["V_i"], inputs["r_a5"], inputs["nt_5"], inputs["l_pF"], inputs["n_ip"], inputs["n_pm"],
        inputs["demand_i"], inputs["dWS_b"], inputs["l_pm"], inputs["dW_s"], inputs["dS_sb"],
        inputs["station_families"], inputs["n_f"], inputs["demand_f"], inputs["ca"], inputs["ov"],
        inputs["s_ip"], inputs["wd_p"], inputs["cl"], inputs["AL_pb"], inputs["ntr_2"], inputs["ht_ip"],
        inputs["mv"], inputs["ALk_pb"], inputs["q_p"], inputs["families_parts"], inputs["L_p"],
        inputs["L_s"], inputs["h_2"], inputs["w_2"], inputs["BIG_M_route"], inputs["mr1_r"],
        inputs["mr2_rb"], inputs["BIG_M_storage"], inputs["d_p"],
        time_limit=time_limit,
        use_preassigned_policy=use_preassigned_policy,
    )

    # Write outputs
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    meta = {
        "timestamp": timestamp,
        "instance": instance,
        "assignment": assignment,
        "time_limit": time_limit,
        "use_preassigned_policy": use_preassigned_policy,
    }

    summary_file = results_dir / f"run_summary_{instance}_{assignment}_{timestamp}.txt"
    _write_run_summary_txt(summary_file, meta, results)

    row = {
        **meta,
        "status": status_map.get(results["status"], results["status"]),
        "solcount": results["solcount"],
        "objective": results.get("objective", None),
        "runtime_seconds": results.get("runtime", None),
    }
    _append_csv_row(results_dir / "runs.csv", row)

    print("Wrote:", summary_file)
    print("Appended:", results_dir / "runs.csv")
    return results


def main():
    """
    Instance naming convention:

    instance = "10x2"
      → 10 workstations in the assembly line
      → 2 supermarkets available in the system

    assignment = "10_02"
      → 10 workstations
      → "02" indicates the third algorithmically generated
        part-to-station assignment (00, 01, 02, ...)

    All part-to-station assignments for a given number of
    stations (e.g. 10_00, 10_01, 10_02) are compatible with
    any distance matrix of the same station size (e.g. 10x1,
    10x2, 10x3, ...).
    """
    p = argparse.ArgumentParser(description="Run Assembly Line Feeding MILP (single run or batch).")
    p.add_argument("--instance", default="10x2", help='e.g. "10x2"')
    p.add_argument("--assignment", default="10_02", help='e.g. "10_02"')
    p.add_argument("--time-limit", type=int, default=600, help="Gurobi time limit in seconds")
    p.add_argument("--use-preassigned-policy", action="store_true", help="Enforce preassigned feeding policy column")
    p.add_argument("--batch", action="store_true", help="Run multiple assignments")
    p.add_argument("--assignments", default="", help='Comma-separated list, e.g. "10_00,10_01,10_02"')

    args = p.parse_args()

    if args.batch:
        if args.assignments.strip():
            assignments = [a.strip() for a in args.assignments.split(",") if a.strip()]
        else:
            # default fallback: for instance "10x2" -> "10" -> 10_00..10_02
            nstations = args.instance.split("x")[0]
            assignments = [f"{nstations}_{i:02d}" for i in range(3)]

        for a in assignments:
            run_one(args.instance, a, args.time_limit, args.use_preassigned_policy)
    else:
        run_one(args.instance, args.assignment, args.time_limit, args.use_preassigned_policy)


if __name__ == "__main__":
    main()


# from pathlib import Path
# from src.io import prepare_inputs
# import src.milp as milp
# import numpy as np
# import pandas as pd
# from datetime import datetime
#
# status_map = {
#     1: "LOADED",
#     2: "OPTIMAL",
#     3: "INFEASIBLE",
#     4: "INF_OR_UNBD",
#     5: "UNBOUNDED",
#     9: "TIME_LIMIT"
# }
#
# def main():
#
#     '''
#     Instance naming convention:
#
#     instance = "10x2"
#       → 10 workstations in the assembly line
#       → 2 supermarkets available in the system
#
#     assignment = "10_02"
#       → 10 workstations
#       → "02" indicates the third algorithmically generated
#         part-to-station assignment (00, 01, 02, ...)
#
#     All part-to-station assignments for a given number of
#     stations (e.g. 10_00, 10_01, 10_02) are compatible with
#     any distance matrix of the same station size (e.g. 10x1,
#     10x2, 10x3, ...).
#     '''
#
#     # adjust these two to your current test case
#     instance = "10x2"
#     assignment = "10_02"
#
#     root = Path(__file__).resolve().parents[1]
#
#     inputs = prepare_inputs(
#         input_data_xlsx=root / "data" / "Input_data" / "Input_Data.xlsx",
#         n_pm_xlsx=root / "data" / "Input_data" / "n_pm.xlsx",
#         l_pm_xlsx=root / "data" / "Input_data" / "l_pm.xlsx",
#         n_f_xlsx=root / "data" / "Input_data" / "n_f.xlsx",
#         distance_matrix_xlsx=root / "data" / "datasets" / f"distance_matrix_{instance}_storage_to_stations.xlsx",
#         part_station_assignment_xlsx=root / "data" / "datasets" / f"part_station_assignment_{assignment}.xlsx",
#     )
#
#
#
#
#
#
#     # set globals used inside Model_1_TC
#     milp.assignment_parts_stations = inputs["assignment_parts_stations"]
#     milp.df = inputs["df"]
#     milp.stations = inputs["stations"]
#     milp.Fix_S = inputs["Fix_S"]
#
#     # run model
#     results = milp.Model_1_TC(
#         inputs["H"], inputs["muy_m"], inputs["v_m"], inputs["co_m"], inputs["Cap_b"], inputs["SC"],
#         inputs["mr_rb"], inputs["r_a2"], inputs["r_ia"], inputs["nk_4"], inputs["r_a4"], inputs["m_i"],
#         inputs["V_i"], inputs["r_a5"], inputs["nt_5"], inputs["l_pF"], inputs["n_ip"], inputs["n_pm"],
#         inputs["demand_i"], inputs["dWS_b"], inputs["l_pm"], inputs["dW_s"], inputs["dS_sb"],
#         inputs["station_families"], inputs["n_f"], inputs["demand_f"], inputs["ca"], inputs["ov"],
#         inputs["s_ip"], inputs["wd_p"], inputs["cl"], inputs["AL_pb"], inputs["ntr_2"], inputs["ht_ip"],
#         inputs["mv"], inputs["ALk_pb"], inputs["q_p"], inputs["families_parts"], inputs["L_p"],
#         inputs["L_s"], inputs["h_2"], inputs["w_2"], inputs["BIG_M_route"], inputs["mr1_r"],
#         inputs["mr2_rb"], inputs["BIG_M_storage"], inputs["d_p"]
#     )
#
#     summary_dir = Path("results")
#     summary_dir.mkdir(exist_ok=True)
#
#     timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
#
#     summary_file = summary_dir / f"run_summary_{instance}_{assignment}_{timestamp}.txt"
#
#     with open(summary_file, "w") as f:
#         f.write(f"timestamp={timestamp}\n")
#         f.write(f"instance={instance}\n")
#         f.write(f"assignment={assignment}\n\n")
#
#         f.write(f"status={status_map.get(results['status'], results['status'])}\n")
#         f.write(f"solcount={results['solcount']}\n")
#
#         if results["solcount"] > 0:
#             f.write(f"objective={results['objective']}\n")
#             f.write(f"runtime_seconds={results['runtime']}\n")
#
#
#
#
# if __name__ == "__main__":
#     main()
