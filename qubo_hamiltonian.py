"""
qubo_hamiltonian.py

Step 3 of the Quantum Pruning project.

Purpose:
Read the existing cost_loss_table.csv from the project directory and convert it
into a QUBO / Ising Hamiltonian model.

This file does NOT run Qiskit yet.
This file does NOT prune the neural network yet.
It only builds the mathematical optimization model that Qiskit will use later.

Default command:

    python qubo_hamiltonian.py

Optional command:

    python qubo_hamiltonian.py --input cost_loss_table.csv --outdir qubo_outputs --max-candidates 10
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ============================================================
# Basic file helpers
# ============================================================

def normalize_column_name(name: str) -> str:
    """
    Normalize CSV column names so the script is tolerant of small differences.

    Examples:
        "L_i" -> "l_i"
        "loss increase" -> "loss_increase"
        "accuracy-drop" -> "accuracy_drop"
    """
    return (
        name.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a CSV value to float safely."""
    if value is None:
        return default

    try:
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except ValueError:
        return default


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """
    Read CSV and normalize column names.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"\nInput CSV not found: {path}\n"
            f"Make sure cost_loss_table.csv is in the same project directory "
            f"as qubo_hamiltonian.py, or pass a path with --input.\n"
        )

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError(f"CSV file has no header: {path}")

        rows: List[Dict[str, Any]] = []

        for raw_row in reader:
            row: Dict[str, Any] = {}

            for key, value in raw_row.items():
                if key is None:
                    continue

                normalized_key = normalize_column_name(str(key))
                row[normalized_key] = value

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


# ============================================================
# Candidate extraction
# ============================================================

def first_existing_value(row: Dict[str, Any], possible_keys: List[str], default: Any = None) -> Any:
    """
    Return the first existing value from a list of possible column names.
    """
    for key in possible_keys:
        if key in row:
            return row[key]
    return default


def get_candidate_name(row: Dict[str, Any]) -> str:
    """
    Extract candidate/block/module name from the CSV row.
    """
    value = first_existing_value(
        row,
        ["candidate", "block", "module", "name", "layer"],
        default="",
    )

    return str(value).strip()


def get_params(row: Dict[str, Any]) -> float:
    """
    Extract parameter count from the CSV row.
    """
    value = first_existing_value(
        row,
        ["params", "parameters", "n_params", "num_params", "parameter_count"],
        default=0.0,
    )

    return safe_float(value, default=0.0)


def get_loss_penalty(row: Dict[str, Any]) -> float:
    """
    Extract L_i, the pruning loss penalty.

    Preferred order:
        1. L_i
        2. loss_increase
        3. loss_increase_raw
        4. accuracy_drop
        5. accuracy_drop_raw
        6. f1_drop
        7. f1_drop_raw

    Meaning:
        high L_i = dangerous block to prune
        low L_i  = safer block to prune
    """
    value = first_existing_value(
        row,
        [
            "l_i",
            "loss_increase",
            "loss_increase_raw",
            "accuracy_drop",
            "accuracy_drop_raw",
            "f1_drop",
            "f1_drop_raw",
        ],
        default=0.0,
    )

    return max(safe_float(value, default=0.0), 0.0)


def get_compression_base(row: Dict[str, Any], params: float) -> Tuple[float, str]:
    """
    Extract C_i if it exists, otherwise use params.

    In the Hamiltonian, C_i means compression benefit.

    We later normalize this value over the selected candidates, so the target
    compression remains easy to interpret.
    """
    if "c_i" in row:
        c_value = safe_float(row.get("c_i"), default=0.0)

        if c_value > 0:
            return c_value, "C_i from CSV"

    return params, "params"


def prepare_candidates(
    rows: List[Dict[str, Any]],
    max_candidates: int,
    selection_method: str,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Prepare pruning candidates for the QUBO model.

    Each selected candidate becomes one binary decision:

        x_i = 0 means keep block i
        x_i = 1 means prune block i
    """

    cleaned: List[Dict[str, Any]] = []
    compression_sources_used: List[str] = []

    for row in rows:
        candidate_name = get_candidate_name(row)
        params = get_params(row)

        if not candidate_name:
            continue

        if params <= 0:
            continue

        raw_loss_penalty = get_loss_penalty(row)
        compression_base, compression_source = get_compression_base(row, params)
        compression_sources_used.append(compression_source)

        if compression_base <= 0:
            continue

        pruning_attractiveness = compression_base / (raw_loss_penalty + 1e-9)

        cleaned.append(
            {
                "candidate": candidate_name,
                "params": params,
                "raw_loss_penalty": raw_loss_penalty,
                "compression_base": compression_base,
                "compression_source": compression_source,
                "pruning_attractiveness": pruning_attractiveness,
            }
        )

    if not cleaned:
        raise ValueError(
            "No valid pruning candidates found. "
            "The CSV should contain at least candidate/module name and params."
        )

    # Normalize loss penalty to [0, 1]
    max_loss = max(candidate["raw_loss_penalty"] for candidate in cleaned)

    for candidate in cleaned:
        if max_loss > 0:
            candidate["loss_penalty"] = candidate["raw_loss_penalty"] / max_loss
        else:
            candidate["loss_penalty"] = 0.0

    # Select the subset for Qiskit-sized simulation
    if selection_method == "attractiveness":
        cleaned.sort(key=lambda candidate: candidate["pruning_attractiveness"], reverse=True)
    elif selection_method == "params":
        cleaned.sort(key=lambda candidate: candidate["params"], reverse=True)
    elif selection_method == "low_loss":
        cleaned.sort(key=lambda candidate: candidate["loss_penalty"])
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")

    selected = cleaned[:max_candidates]

    total_compression_base = sum(candidate["compression_base"] for candidate in selected)

    if total_compression_base <= 0:
        raise ValueError("Selected candidates have zero total compression value.")

    # Normalize compression values so sum(C_i) = 1 over selected candidates.
    # This makes target_compression = 0.30 mean about 30% of selected candidate mass.
    for index, candidate in enumerate(selected):
        candidate["qubit_index"] = index
        candidate["compression_value"] = candidate["compression_base"] / total_compression_base

    if all(source == "C_i from CSV" for source in compression_sources_used):
        compression_source_summary = "C_i from CSV"
    elif all(source == "params" for source in compression_sources_used):
        compression_source_summary = "params"
    else:
        compression_source_summary = "mixed: C_i where available, otherwise params"

    return selected, compression_source_summary


# ============================================================
# QUBO construction
# ============================================================

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
            + lambda * (sum(C_i * x_i) - C_target)^2

    Meaning:

        L_i = loss penalty for pruning block i
        C_i = compression benefit for pruning block i

    QUBO form:

        H(x) = constant
               + sum_i Q[i, i] * x_i
               + sum_{i < j} Q[i, j] * x_i * x_j

    The Q matrix is stored as upper-triangular.
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
        "constant": float(constant),
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
    """
    Evaluate the QUBO energy for one binary vector.
    """
    return float(constant + x @ Q @ x)


# ============================================================
# QUBO to Ising Hamiltonian mapping
# ============================================================

def qubo_to_ising_terms(Q: np.ndarray, qubo_constant: float) -> Dict[str, Any]:
    """
    Convert QUBO to Ising Hamiltonian terms.

    Mapping:

        x_i = (1 - Z_i) / 2

    Ising form:

        H = constant + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j

    These terms will be used in the next Qiskit step.
    """

    n = Q.shape[0]

    h = np.zeros(n, dtype=float)
    J = np.zeros((n, n), dtype=float)

    constant = float(qubo_constant)

    # Linear QUBO terms
    for i in range(n):
        a = float(Q[i, i])

        constant += a / 2.0
        h[i] += -a / 2.0

    # Quadratic QUBO terms
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


# ============================================================
# Classical sanity check
# ============================================================

def brute_force_energy_check(
    Q: np.ndarray,
    constant: float,
    candidates: List[Dict[str, Any]],
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Classical sanity check.

    This is NOT the quantum simulation.

    For 10 variables:

        2^10 = 1024

    possible masks, so brute-force checking is still easy.

    Later, Qiskit QAOA should try to find the same or similar low-energy masks.
    """

    n = len(candidates)

    if n > 20:
        raise ValueError("Brute-force check is disabled for more than 20 candidates.")

    results: List[Dict[str, Any]] = []

    for bits in itertools.product([0, 1], repeat=n):
        x = np.array(bits, dtype=float)

        energy = qubo_energy(Q, x, constant)

        compression = sum(
            candidates[i]["compression_value"] * bits[i]
            for i in range(n)
        )

        loss_penalty = sum(
            candidates[i]["loss_penalty"] * bits[i]
            for i in range(n)
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
                "compression": float(compression),
                "loss_penalty": float(loss_penalty),
                "num_pruned_blocks": int(sum(bits)),
                "pruned_blocks": "; ".join(pruned_blocks),
            }
        )

    results.sort(key=lambda row: row["energy"])

    return results[:max_results]


# ============================================================
# Export functions
# ============================================================

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
                "compression_base": candidate["compression_base"],
                "compression_source": candidate["compression_source"],
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
            "compression_base",
            "compression_source",
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


# ============================================================
# Main entry point
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build QUBO/Hamiltonian files from cost_loss_table.csv."
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
        help="Target compression share. Example: 0.30 = 30 percent.",
    )

    parser.add_argument(
        "--energy-check-results",
        type=int,
        default=20,
        help="Number of lowest-energy bitstrings to export as sanity check.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)

    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive.")

    if not (0.0 <= args.target_compression <= 1.0):
        raise ValueError("--target-compression should be between 0 and 1.")

    rows = read_csv_rows(input_path)

    candidates, compression_source_summary = prepare_candidates(
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
        "current_step": "Step 3: CSV -> QUBO -> Ising Hamiltonian",
        "input_csv": str(input_path),
        "output_directory": str(outdir),
        "number_of_candidates": len(candidates),
        "selection_method": args.selection_method,
        "compression_source": compression_source_summary,
        "alpha": args.alpha,
        "beta": args.beta,
        "lambda_constraint": args.lambda_constraint,
        "target_compression": args.target_compression,
        "binary_variable_meaning": {
            "0": "keep the candidate block",
            "1": "prune the candidate block",
        },
        "qubo_formula": (
            "H(x) = alpha*sum(L_i*x_i) "
            "- beta*sum(C_i*x_i) "
            "+ lambda*(sum(C_i*x_i)-target_compression)^2"
        ),
        "qubo_energy_convention": (
            "H(x) = constant + sum_i Q[i,i]*x_i + sum_{i<j} Q[i,j]*x_i*x_j"
        ),
        "ising_mapping": "x_i = (1 - Z_i) / 2",
        "next_step": "Step 4: use hamiltonian_terms.json in Qiskit QAOA simulation.",
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
    print(f"Compression source     : {compression_source_summary}")
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

    print("\nCurrent stage:")
    print("cost_loss_table.csv -> QUBO matrix -> Ising Hamiltonian")

    print("\nImportant:")
    print("This step built the mathematical model only. It did not run Qiskit yet.")


if __name__ == "__main__":
    main()
