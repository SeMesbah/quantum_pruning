"""
qubo_hamiltonian.py

Step 3 of the Quantum Pruning project.

Purpose:
Convert block-pruning sensitivity data into a QUBO / Ising Hamiltonian model.

This file does NOT run Qiskit yet.
This file does NOT prune the neural network yet.
This file only builds the mathematical optimization model that Qiskit will use later.

Input:
cost_loss_table.csv

Outputs:
qubo_outputs/selected_candidates.csv
qubo_outputs/qubo_matrix.csv
qubo_outputs/qubo_terms.json
qubo_outputs/hamiltonian_terms.json
qubo_outputs/qubo_metadata.json
qubo_outputs/qubo_energy_check.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert CSV text value to float safely."""
    if value is None:
        return default

    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except ValueError:
        return default


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read CSV as a list of dictionaries."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        rows: List[Dict[str, str]] = []

        for raw_row in reader:
            row = {str(key).strip(): value for key, value in raw_row.items() if key is not None}
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Any) -> None:
    """Write data to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def get_loss_penalty(row: Dict[str, str]) -> float:
    """
    Extract pruning loss penalty from one CSV row.

    Preferred column order:
    1. L_i
    2. loss_increase_raw
    3. accuracy_drop_raw
    4. f1_drop_raw
    """

    for column in ["L_i", "loss_increase_raw", "accuracy_drop_raw", "f1_drop_raw"]:
        if column in row:
            return max(safe_float(row.get(column), default=0.0), 0.0)

    return 0.0


def prepare_candidates(
    rows: List[Dict[str, str]],
    max_candidates: int,
    selection_method: str,
) -> List[Dict[str, Any]]:
    """
    Prepare candidates for the QUBO model.

    Each candidate becomes one binary pruning decision:

        x_i = 0 means keep block i
        x_i = 1 means prune block i

    The loss penalty is normalized to [0, 1].
    The compression value is params / total selected params.

    This means target_compression = 0.30 means:
    try to prune about 30% of the selected candidate parameter mass.
    """

    cleaned: List[Dict[str, Any]] = []

    for row in rows:
        candidate = str(row.get("candidate", "")).strip()
        params = safe_float(row.get("params"), default=0.0)

        if not candidate:
            continue

        if params <= 0:
            continue

        raw_loss = get_loss_penalty(row)

        if "pruning_attractiveness" in row:
            attractiveness = safe_float(row.get("pruning_attractiveness"), default=0.0)
        else:
            attractiveness = params / (raw_loss + 1e-9)

        cleaned.append(
            {
                "candidate": candidate,
                "params": params,
                "raw_loss_penalty": raw_loss,
                "pruning_attractiveness": attractiveness,
                "source_row": row,
            }
        )

    if not cleaned:
        raise ValueError("No valid pruning candidates found in the CSV.")

    losses = np.array([candidate["raw_loss_penalty"] for candidate in cleaned], dtype=float)
    max_loss = float(np.max(losses)) if len(losses) else 0.0

    for candidate in cleaned:
        if max_loss > 0:
            candidate["loss_penalty"] = float(candidate["raw_loss_penalty"] / max_loss)
        else:
            candidate["loss_penalty"] = 0.0

    if selection_method == "attractiveness":
        cleaned.sort(key=lambda candidate: candidate["pruning_attractiveness"], reverse=True)
    elif selection_method == "params":
        cleaned.sort(key=lambda candidate: candidate["params"], reverse=True)
    elif selection_method == "low_loss":
        cleaned.sort(key=lambda candidate: candidate["loss_penalty"])
    else:
        raise ValueError("Unknown selection method.")

    selected = cleaned[:max_candidates]

    total_selected_params = sum(candidate["params"] for candidate in selected)

    if total_selected_params <= 0:
        raise ValueError("Selected candidates have zero total parameters.")

    for index, candidate in enumerate(selected):
        candidate["qubit_index"] = index
        candidate["compression_value"] = float(candidate["params"] / total_selected_params)

    return selected


def build_qubo(
    candidates: List[Dict[str, Any]],
    alpha: float,
    beta: float,
    lambda_constraint: float,
    target_compression: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the QUBO matrix.

    Objective:

        H(x) =
            alpha * sum(L_i * x_i)
            - beta * sum(C_i * x_i)
            + lambda * (sum(C_i * x_i) - target_compression)^2

    Meaning:

        L_i = penalty for pruning an important block
        C_i = compression benefit from pruning a block

    QUBO convention:

        H(x) = constant
               + sum_i Q[i, i] * x_i
               + sum_{i < j} Q[i, j] * x_i * x_j

    The Q matrix is upper-triangular.
    """

    n = len(candidates)
    Q = np.zeros((n, n), dtype=float)

    L = np.array([candidate["loss_penalty"] for candidate in candidates], dtype=float)
    C = np.array([candidate["compression_value"] for candidate in candidates], dtype=float)

    for i in range(n):
        Q[i, i] = (
            alpha * L[i]
            - beta * C[i]
            + lambda_constraint * (C[i] ** 2 - 2.0 * target_compression * C[i])
        )

    for i in range(n):
        for j in range(i + 1, n):
            Q[i, j] = 2.0 * lambda_constraint * C[i] * C[j]

    constant = lambda_constraint * (target_compression ** 2)

    qubo_terms = {
        "constant": constant,
        "linear_terms": [float(Q[i, i]) for i in range(n)],
        "quadratic_terms": [
            {
                "i": i,
                "j": j,
                "coefficient": float(Q[i, j]),
            }
            for i in range(n)
            for j in range(i + 1, n)
            if abs(Q[i, j]) > 1e-15
        ],
        "formula": (
            "H(x) = alpha*sum(L_i*x_i) "
            "- beta*sum(C_i*x_i) "
            "+ lambda*(sum(C_i*x_i)-target_compression)^2"
        ),
    }

    return Q, qubo_terms


def qubo_energy(Q: np.ndarray, x: np.ndarray, constant: float = 0.0) -> float:
    """Evaluate QUBO energy for one binary vector."""
    return float(constant + x @ Q @ x)


def qubo_to_ising_terms(Q: np.ndarray, qubo_constant: float) -> Dict[str, Any]:
    """
    Convert QUBO to Ising Hamiltonian terms.

    Mapping:

        x_i = (1 - Z_i) / 2

    Ising form:

        H = constant + sum_i h_i * Z_i + sum_{i<j} J_ij * Z_i Z_j

    This is the form that the later Qiskit file can convert into Pauli operators.
    """

    n = Q.shape[0]

    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)

    constant = float(qubo_constant)

    for i in range(n):
        a = float(Q[i, i])

        constant += a / 2.0
        h[i] += -a / 2.0

    for i in range(n):
        for j in range(i + 1, n):
            b = float(Q[i, j])

            if abs(b) <= 1e-15:
                continue

            constant += b / 4.0
            h[i] += -b / 4.0
            h[j] += -b / 4.0
            J[i, j] += b / 4.0

    return {
        "constant": float(constant),
        "z_terms": [
            {
                "i": i,
                "coefficient": float(h[i]),
            }
            for i in range(n)
            if abs(h[i]) > 1e-15
        ],
        "zz_terms": [
            {
                "i": i,
                "j": j,
                "coefficient": float(J[i, j]),
            }
            for i in range(n)
            for j in range(i + 1, n)
            if abs(J[i, j]) > 1e-15
        ],
        "mapping": "x_i = (1 - Z_i) / 2",
        "ising_form": "H = constant + sum_i h_i*Z_i + sum_{i<j} J_ij*Z_i*Z_j",
    }


def brute_force_energy_check(
    Q: np.ndarray,
    constant: float,
    candidates: List[Dict[str, Any]],
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Small classical sanity check.

    This is NOT the quantum simulation.

    For 10 variables, there are:

        2^10 = 1024

    possible pruning masks, so it is easy to check the lowest-energy masks.
    """

    n = len(candidates)

    if n > 20:
        raise ValueError("Brute-force check is disabled for more than 20 candidates.")

    results: List[Dict[str, Any]] = []

    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)

        energy = qubo_energy(Q, x, constant)

        compression = float(
            sum(candidates[i]["compression_value"] * bits[i] for i in range(n))
        )

        loss_penalty = float(
            sum(candidates[i]["loss_penalty"] * bits[i] for i in range(n))
        )

        pruned_blocks = [
            candidates[i]["candidate"]
            for i in range(n)
            if bits[i] == 1
        ]

        results.append(
            {
                "bitstring": "".join(str(bit) for bit in bits),
                "energy": energy,
                "compression": compression,
                "loss_penalty": loss_penalty,
                "num_pruned_blocks": int(sum(bits)),
                "pruned_blocks": "; ".join(pruned_blocks),
            }
        )

    results.sort(key=lambda row: row["energy"])

    return results[:max_results]


def export_selected_candidates(outdir: Path, candidates: List[Dict[str, Any]]) -> None:
    rows: List[Dict[str, Any]] = []

    for candidate in candidates:
        rows.append(
            {
                "qubit_index": candidate["qubit_index"],
                "candidate": candidate["candidate"],
                "params": int(candidate["params"]),
                "loss_penalty": candidate["loss_penalty"],
                "compression_value": candidate["compression_value"],
                "raw_loss_penalty": candidate["raw_loss_penalty"],
                "pruning_attractiveness": candidate["pruning_attractiveness"],
            }
        )

    write_csv(
        outdir / "selected_candidates.csv",
        rows,
        [
            "qubit_index",
            "candidate",
            "params",
            "loss_penalty",
            "compression_value",
            "raw_loss_penalty",
            "pruning_attractiveness",
        ],
    )


def export_qubo_matrix(outdir: Path, Q: np.ndarray) -> None:
    n = Q.shape[0]

    columns = ["row"] + [f"q{j}" for j in range(n)]
    rows: List[Dict[str, Any]] = []

    for i in range(n):
        row = {"row": f"q{i}"}

        for j in range(n):
            row[f"q{j}"] = float(Q[i, j])

        rows.append(row)

    write_csv(outdir / "qubo_matrix.csv", rows, columns)


def export_energy_check(outdir: Path, energy_rows: List[Dict[str, Any]]) -> None:
    write_csv(
        outdir / "qubo_energy_check.csv",
        energy_rows,
        [
            "bitstring",
            "energy",
            "compression",
            "loss_penalty",
            "num_pruned_blocks",
            "pruned_blocks",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build QUBO/Hamiltonian files from pruning sensitivity CSV."
    )

    parser.add_argument(
        "--input",
        type=str,
        default="cost_loss_table.csv",
        help="Input pruning sensitivity CSV file.",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="qubo_outputs",
        help="Directory where QUBO/Hamiltonian outputs will be saved.",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Maximum number of pruning candidates/qubits to use.",
    )

    parser.add_argument(
        "--selection-method",
        type=str,
        default="attractiveness",
        choices=["attractiveness", "params", "low_loss"],
        help="How to select candidates for the QUBO subset.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Weight of loss penalty term.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Weight of compression reward term.",
    )

    parser.add_argument(
        "--lambda-constraint",
        type=float,
        default=4.0,
        help="Weight of target-compression constraint term.",
    )

    parser.add_argument(
        "--target-compression",
        type=float,
        default=0.30,
        help="Target compression share among selected candidates. Example: 0.30 = 30%.",
    )

    parser.add_argument(
        "--energy-check-results",
        type=int,
        default=20,
        help="Number of lowest-energy bitstrings to export as a sanity check.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive.")

    if not (0.0 <= args.target_compression <= 1.0):
        raise ValueError("--target-compression should be between 0 and 1.")

    rows = read_csv_rows(input_path)

    candidates = prepare_candidates(
        rows=rows,
        max_candidates=args.max_candidates,
        selection_method=args.selection_method,
    )

    if len(candidates) < args.max_candidates:
        print(
            f"Warning: requested {args.max_candidates} candidates, "
            f"but only {len(candidates)} valid candidates were found."
        )

    Q, qubo_terms = build_qubo(
        candidates=candidates,
        alpha=args.alpha,
        beta=args.beta,
        lambda_constraint=args.lambda_constraint,
        target_compression=args.target_compression,
    )

    hamiltonian_terms = qubo_to_ising_terms(
        Q=Q,
        qubo_constant=qubo_terms["constant"],
    )

    metadata = {
        "input_csv": str(input_path),
        "number_of_candidates": len(candidates),
        "selection_method": args.selection_method,
        "alpha": args.alpha,
        "beta": args.beta,
        "lambda_constraint": args.lambda_constraint,
        "target_compression": args.target_compression,
        "binary_variable_meaning": {
            "0": "keep the candidate block",
            "1": "prune the candidate block",
        },
        "qubo_energy_convention": (
            "H(x) = constant + sum_i Q[i,i]*x_i + sum_{i<j} Q[i,j]*x_i*x_j"
        ),
        "matrix_note": (
            "qubo_matrix.csv is upper-triangular. "
            "Lower-triangular values are zero by design."
        ),
        "next_step": "Use hamiltonian_terms.json in a Qiskit QAOA simulation file.",
    }

    export_selected_candidates(outdir, candidates)
    export_qubo_matrix(outdir, Q)

    write_json(outdir / "qubo_terms.json", qubo_terms)
    write_json(outdir / "hamiltonian_terms.json", hamiltonian_terms)
    write_json(outdir / "qubo_metadata.json", metadata)

    if len(candidates) <= 20 and args.energy_check_results > 0:
        energy_rows = brute_force_energy_check(
            Q=Q,
            constant=qubo_terms["constant"],
            candidates=candidates,
            max_results=args.energy_check_results,
        )

        export_energy_check(outdir, energy_rows)

    print("\nQUBO / Hamiltonian construction completed.")
    print(f"Input CSV              : {input_path}")
    print(f"Output directory       : {outdir}")
    print(f"Candidates / qubits    : {len(candidates)}")
    print(f"Target compression     : {args.target_compression:.2%}")
    print(f"Alpha loss weight      : {args.alpha}")
    print(f"Beta compression weight: {args.beta}")
    print(f"Lambda constraint      : {args.lambda_constraint}")

    print("\nGenerated files:")
    print(f"- {outdir / 'selected_candidates.csv'}")
    print(f"- {outdir / 'qubo_matrix.csv'}")
    print(f"- {outdir / 'qubo_terms.json'}")
    print(f"- {outdir / 'hamiltonian_terms.json'}")
    print(f"- {outdir / 'qubo_metadata.json'}")

    if len(candidates) <= 20 and args.energy_check_results > 0:
        print(f"- {outdir / 'qubo_energy_check.csv'}")

    print("\nImportant:")
    print("This step built the mathematical model only. It did not run Qiskit yet.")


if __name__ == "__main__":
    main()
