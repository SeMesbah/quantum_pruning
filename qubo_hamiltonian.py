"""
qubo_hamiltonian.py

Corrected Step 3 of the Quantum Pruning project.

This version fixes the main modelling problem in the earlier QUBO:

the old objective

    alpha * sum(L_i x_i) - beta * sum(C_i x_i)
    + lambda * (sum(C_i x_i) - C_target)^2

is algebraically valid, but the squared global target creates an all-to-all
Hamiltonian. For n pruning candidates it produces n(n-1)/2 ZZ couplings.
That makes the QAOA cost circuit dense and hardware-unfriendly. In addition,
the -beta compression reward shifts the effective target above C_target.

The corrected default formulation is therefore sparse and topology-aware:

    H(x) = sum_i (alpha L_i - beta C_i) x_i
           + rho * sum_(i,j in E_topology) w_ij x_i x_j
           + mu  * sum_(i,j in E_budget) g_ij x_i x_j

where E_topology contains only meaningful model-structure edges, by default
candidates from the same model stage. E_budget contains only pairs that would
exceed the allowed target band. The target compression is used to tune beta and
to build a sparse budget guard, instead of being encoded as a global squared
all-to-all penalty. This keeps the Hamiltonian much sparser.

Binary meaning:
    x_i = 0  keep candidate block i
    x_i = 1  prune candidate block i

Qiskit / Ising mapping:
    x_i = (1 - Z_i) / 2

Default command:
    python qubo_hamiltonian.py

Recommended explicit command:
    python qubo_hamiltonian.py \
        --input cost_loss_table.csv \
        --outdir qubo_outputs \
        --max-candidates 10 \
        --formulation sparse_topology \
        --interaction-method same_stage \
        --target-compression 0.30
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ============================================================
# Basic file helpers
# ============================================================


def normalize_column_name(name: str) -> str:
    """Normalize CSV column names so the script tolerates small differences."""
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


def resolve_input_path(user_path: str) -> Path:
    """
    Resolve the input CSV robustly.

    Older project zips sometimes contain cost_loss_table.csv only inside
    qubo_outputs/. If the requested path does not exist in the project root,
    this function falls back to qubo_outputs/<filename>.
    """
    path = Path(user_path)

    if path.exists():
        return path

    fallback = Path("qubo_outputs") / path.name
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        f"\nInput CSV not found: {path}\n"
        f"Also checked fallback path: {fallback}\n"
        f"Pass a valid path using --input.\n"
    )


def read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    """Read CSV and normalize column names."""
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
                row[normalize_column_name(str(key))] = value
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
    """Return the first existing value from a list of possible column names."""
    for key in possible_keys:
        if key in row:
            return row[key]
    return default


def get_candidate_name(row: Dict[str, Any]) -> str:
    """Extract candidate/block/module name from the CSV row."""
    value = first_existing_value(
        row,
        ["candidate", "block", "module", "name", "layer"],
        default="",
    )
    return str(value).strip()


def get_params(row: Dict[str, Any]) -> float:
    """Extract parameter count from the CSV row."""
    value = first_existing_value(
        row,
        ["params", "parameters", "n_params", "num_params", "parameter_count", "c_i_raw"],
        default=0.0,
    )
    return safe_float(value, default=0.0)


def get_loss_penalty(row: Dict[str, Any]) -> float:
    """
    Extract L_i, the pruning loss penalty.

    Preferred order:
        1. L_i / normalized loss penalty
        2. loss_increase / loss_increase_raw
        3. accuracy_drop / accuracy_drop_raw
        4. f1_drop / f1_drop_raw

    Meaning:
        high L_i = dangerous block to prune
        low L_i  = safer block to prune
    """
    value = first_existing_value(
        row,
        [
            "l_i",
            "loss_penalty",
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
    Extract a raw compression value.

    Important correction:
    - If C_i in the CSV is already normalized relative to the largest block, it
      is NOT used directly as the final Hamiltonian compression mass.
    - The selected candidates are normalized again so sum(C_i)=1 over the QUBO
      subset. This keeps target_compression = 0.30 interpretable.
    """
    value = first_existing_value(row, ["c_i_raw", "compression_base"], default=None)
    if value is not None:
        c_value = safe_float(value, default=0.0)
        if c_value > 0:
            return c_value, "C_i_raw from CSV"

    value = first_existing_value(row, ["params", "parameters", "n_params", "num_params"], default=None)
    if value is not None:
        c_value = safe_float(value, default=0.0)
        if c_value > 0:
            return c_value, "params"

    value = first_existing_value(row, ["c_i", "compression_value"], default=None)
    if value is not None:
        c_value = safe_float(value, default=0.0)
        if c_value > 0:
            return c_value, "C_i from CSV"

    return params, "params"


def parse_stage_block(candidate_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse names like stages.2.blocks.6 into (stage=2, block=6)."""
    match = re.search(r"stages\.(\d+)\.blocks\.(\d+)", candidate_name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


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

        if not candidate_name or params <= 0:
            continue

        raw_loss_penalty = get_loss_penalty(row)
        compression_base, compression_source = get_compression_base(row, params)
        compression_sources_used.append(compression_source)

        if compression_base <= 0:
            continue

        pruning_attractiveness = compression_base / (raw_loss_penalty + 1e-12)
        stage_index, block_index = parse_stage_block(candidate_name)

        cleaned.append(
            {
                "candidate": candidate_name,
                "params": params,
                "raw_loss_penalty": raw_loss_penalty,
                "compression_base": compression_base,
                "compression_source": compression_source,
                "pruning_attractiveness": pruning_attractiveness,
                "stage_index": stage_index,
                "block_index": block_index,
            }
        )

    if not cleaned:
        raise ValueError(
            "No valid pruning candidates found. "
            "The CSV should contain at least candidate/module name and params."
        )

    # Normalize loss penalty to [0, 1].
    max_loss = max(candidate["raw_loss_penalty"] for candidate in cleaned)

    for candidate in cleaned:
        if max_loss > 0:
            candidate["loss_penalty"] = candidate["raw_loss_penalty"] / max_loss
        else:
            candidate["loss_penalty"] = 0.0

    # Select the subset for Qiskit-sized simulation.
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

    for index, candidate in enumerate(selected):
        candidate["qubit_index"] = index
        candidate["compression_value"] = candidate["compression_base"] / total_compression_base

    if all(source == compression_sources_used[0] for source in compression_sources_used):
        compression_source_summary = compression_sources_used[0]
    else:
        compression_source_summary = "mixed: C_i_raw/params/C_i depending on availability"

    return selected, compression_source_summary


# ============================================================
# Sparse topology-aware interaction graph
# ============================================================


def build_interaction_edges(
    candidates: List[Dict[str, Any]],
    method: str,
    target_compression: float,
    target_tolerance: float,
    budget_guard_weight: float,
) -> List[Dict[str, Any]]:
    """
    Build a sparse set of pairwise interaction edges.

    These edges are the corrected replacement for the old all-to-all target
    square. They represent two ideas:

    1. topology guard:
       pruning two blocks from the same local model region is often riskier
       than the sum of pruning them independently.

    2. budget guard:
       if two candidates together exceed the allowed compression band, add a
       pairwise penalty. This discourages over-pruning without creating a
       complete graph between every pair.

    Supported topology methods:
        none        -> no topology pairwise terms
        adjacent    -> same stage and neighbouring block index
        same_stage  -> same stage, any two selected blocks in that stage
    """
    edges_by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}
    upper_compression = target_compression + target_tolerance

    def add_edge(i: int, j: int, coefficient: float, reason: str) -> None:
        if coefficient <= 0:
            return
        key = (i, j) if i < j else (j, i)
        if key not in edges_by_pair:
            edges_by_pair[key] = {
                "i": key[0],
                "j": key[1],
                "coefficient_base": 0.0,
                "reasons": [],
                "candidate_i": candidates[key[0]]["candidate"],
                "candidate_j": candidates[key[1]]["candidate"],
            }
        edges_by_pair[key]["coefficient_base"] += float(coefficient)
        edges_by_pair[key]["reasons"].append(reason)

    for i in range(len(candidates)):
        ci = candidates[i]
        stage_i = ci.get("stage_index")
        block_i = ci.get("block_index")

        for j in range(i + 1, len(candidates)):
            cj = candidates[j]
            stage_j = cj.get("stage_index")
            block_j = cj.get("block_index")

            c_i = float(ci["compression_value"])
            c_j = float(cj["compression_value"])
            l_i = float(ci["loss_penalty"])
            l_j = float(cj["loss_penalty"])

            # -----------------------------
            # 1) sparse topology guard
            # -----------------------------
            topology_include = False
            topology_reason = ""

            if method != "none":
                if stage_i is not None and stage_j is not None:
                    same_stage = stage_i == stage_j
                    adjacent = (
                        same_stage
                        and block_i is not None
                        and block_j is not None
                        and abs(block_i - block_j) == 1
                    )

                    if method == "adjacent" and adjacent:
                        topology_include = True
                        topology_reason = "topology: adjacent blocks in same stage"
                    elif method == "same_stage" and same_stage:
                        topology_include = True
                        topology_reason = "topology: same model stage"
                    elif method not in {"none", "adjacent", "same_stage"}:
                        raise ValueError(f"Unknown interaction method: {method}")

            if topology_include:
                # sqrt(C_i*C_j) is stronger and more stable than C_i*C_j for
                # small normalized values. The small loss factor raises the
                # penalty when both candidates are individually risky.
                topology_weight = math.sqrt(c_i * c_j) * (1.0 + 0.5 * (l_i + l_j))
                add_edge(i, j, topology_weight, topology_reason)

            # -----------------------------
            # 2) sparse budget guard
            # -----------------------------
            # This is a pairwise approximation of "do not go too far above the
            # target band". Unlike (sum C_i x_i - target)^2, it does not connect
            # every candidate to every other candidate. It adds an edge only
            # when this pair already exceeds the upper allowed compression band.
            pair_compression = c_i + c_j
            excess = pair_compression - upper_compression
            if budget_guard_weight > 0 and excess > 0:
                budget_weight = budget_guard_weight * excess
                add_edge(
                    i,
                    j,
                    budget_weight,
                    f"budget: pair compression {pair_compression:.4f} exceeds upper band {upper_compression:.4f}",
                )

    edges: List[Dict[str, Any]] = []
    for edge in edges_by_pair.values():
        edge["reason"] = " | ".join(edge.pop("reasons"))
        edges.append(edge)

    edges.sort(key=lambda e: (e["i"], e["j"]))
    return edges

# ============================================================
# QUBO construction
# ============================================================


def build_sparse_topology_qubo(
    candidates: List[Dict[str, Any]],
    interaction_edges: List[Dict[str, Any]],
    alpha: float,
    beta: float,
    rho_interaction: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the corrected sparse topology-aware QUBO.

    Objective:
        H(x) = sum_i (alpha L_i - beta C_i) x_i
               + sum_(i,j in E_sparse) q_ij x_i x_j

    The sparse edge set combines topology guards and budget guards. This does
    NOT encode the target as (sum C_i x_i - C_target)^2, because that old term
    creates all-to-all ZZ couplings. Instead, beta is calibrated so the
    low-energy solution lands near the desired compression, while the budget
    guard discourages over-pruning.
    """
    n = len(candidates)
    Q = np.zeros((n, n), dtype=float)

    L = np.array([candidate["loss_penalty"] for candidate in candidates], dtype=float)
    C = np.array([candidate["compression_value"] for candidate in candidates], dtype=float)

    for i in range(n):
        Q[i, i] = alpha * L[i] - beta * C[i]

    for edge in interaction_edges:
        i = int(edge["i"])
        j = int(edge["j"])
        Q[i, j] = rho_interaction * float(edge["coefficient_base"])

    qubo_terms = {
        "constant": 0.0,
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
            "H(x) = sum_i((alpha*L_i - beta*C_i)*x_i) "
            "+ rho*sum_{(i,j) in E}(w_ij*x_i*x_j)"
        ),
        "correction_note": (
            "Target compression is not encoded as a squared global penalty. "
            "This avoids all-to-all ZZ terms and keeps the QAOA cost circuit sparse."
        ),
    }

    return Q, qubo_terms


def build_legacy_target_qubo(
    candidates: List[Dict[str, Any]],
    alpha: float,
    beta: float,
    lambda_constraint: float,
    target_compression: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build the old global target QUBO for comparison only.

    This mode is kept so that old results can be reproduced. It is not the
    recommended default because it produces all-to-all ZZ couplings.
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
            "LEGACY: H(x) = alpha*sum(L_i*x_i) "
            "- beta*sum(C_i*x_i) "
            "+ lambda*(sum(C_i*x_i)-target_compression)^2"
        ),
        "warning": (
            "Legacy target-square formulation creates all-to-all ZZ couplings "
            "and beta shifts the effective compression target. Use only for comparison."
        ),
    }

    return Q, qubo_terms


def qubo_energy(Q: np.ndarray, x: np.ndarray, constant: float = 0.0) -> float:
    """Evaluate the QUBO energy for one binary vector."""
    return float(constant + x @ Q @ x)


# ============================================================
# Beta calibration
# ============================================================


def all_binary_vectors(n: int) -> Iterable[Tuple[int, ...]]:
    return itertools.product([0, 1], repeat=n)


def best_assignment_for_qubo(
    Q: np.ndarray,
    constant: float,
    candidates: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return the best binary assignment for the given QUBO."""
    n = len(candidates)
    best: Optional[Dict[str, Any]] = None

    for bits in all_binary_vectors(n):
        x = np.array(bits, dtype=float)
        energy = qubo_energy(Q, x, constant)
        compression = sum(candidates[i]["compression_value"] * bits[i] for i in range(n))
        loss_penalty = sum(candidates[i]["loss_penalty"] * bits[i] for i in range(n))
        bitstring = "".join(str(bit) for bit in bits)

        row = {
            "bitstring": bitstring,
            "energy": float(energy),
            "compression": float(compression),
            "loss_penalty": float(loss_penalty),
            "num_pruned_blocks": int(sum(bits)),
        }

        if best is None:
            best = row
        else:
            # Primary optimization remains energy. Tie-breakers are deterministic.
            old_key = (best["energy"], best["loss_penalty"], -best["compression"], best["bitstring"])
            new_key = (row["energy"], row["loss_penalty"], -row["compression"], row["bitstring"])
            if new_key < old_key:
                best = row

    if best is None:
        raise RuntimeError("No binary assignment was generated.")

    return best


def calibrate_beta_for_target(
    candidates: List[Dict[str, Any]],
    interaction_edges: List[Dict[str, Any]],
    alpha: float,
    rho_interaction: float,
    target_compression: float,
    beta_min: float,
    beta_max: float,
    beta_steps: int,
) -> Dict[str, Any]:
    """
    Choose beta so that the sparse QUBO's ground state is close to the requested
    target compression.

    This replaces the old global square constraint. It is still a QUBO, but the
    target is handled by tuning a linear Lagrange reward instead of adding a
    dense all-to-all penalty.
    """
    if beta_steps < 2:
        raise ValueError("--beta-steps must be at least 2 for auto calibration.")

    grid = np.linspace(beta_min, beta_max, beta_steps)
    records: List[Dict[str, Any]] = []

    for beta in grid:
        Q, qubo_terms = build_sparse_topology_qubo(
            candidates=candidates,
            interaction_edges=interaction_edges,
            alpha=alpha,
            beta=float(beta),
            rho_interaction=rho_interaction,
        )
        best = best_assignment_for_qubo(Q, qubo_terms["constant"], candidates)
        best["beta"] = float(beta)
        best["target_distance"] = abs(best["compression"] - target_compression)
        records.append(best)

    # First choose the best bitstring family: closest to target, then lowest loss,
    # then higher compression, then fewer pruned blocks.
    best_record = min(
        records,
        key=lambda row: (
            row["target_distance"],
            row["loss_penalty"],
            -row["compression"],
            row["num_pruned_blocks"],
            row["bitstring"],
        ),
    )

    target_bitstring = best_record["bitstring"]
    interval_records = [row for row in records if row["bitstring"] == target_bitstring]

    # Pick the middle of the beta interval where this same bitstring is the
    # optimum. This avoids choosing a beta exactly on a decision boundary.
    beta_values = [row["beta"] for row in interval_records]
    calibrated_beta = float((min(beta_values) + max(beta_values)) / 2.0)

    return {
        "beta": calibrated_beta,
        "selected_bitstring_during_calibration": target_bitstring,
        "selected_compression_during_calibration": best_record["compression"],
        "selected_loss_penalty_during_calibration": best_record["loss_penalty"],
        "target_compression": target_compression,
        "target_distance": best_record["target_distance"],
        "beta_interval_for_selected_bitstring": [float(min(beta_values)), float(max(beta_values))],
        "beta_grid_min": beta_min,
        "beta_grid_max": beta_max,
        "beta_grid_steps": beta_steps,
        "explanation": (
            "Beta was calibrated so the sparse QUBO ground state lands close to "
            "the requested target compression without adding an all-to-all "
            "squared target penalty."
        ),
    }


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
            {"i": i, "coefficient": float(h[i])}
            for i in range(n)
            if abs(h[i]) > 1e-15
        ],
        "zz_terms": [
            {"i": i, "j": j, "coefficient": float(J[i, j])}
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
    """Classical sanity check for small QUBOs."""
    n = len(candidates)

    if n > 20:
        raise ValueError("Brute-force check is disabled for more than 20 candidates.")

    results: List[Dict[str, Any]] = []

    for bits in all_binary_vectors(n):
        x = np.array(bits, dtype=float)
        energy = qubo_energy(Q, x, constant)

        compression = sum(candidates[i]["compression_value"] * bits[i] for i in range(n))
        loss_penalty = sum(candidates[i]["loss_penalty"] * bits[i] for i in range(n))
        pairwise_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_penalty += Q[i, j] * bits[i] * bits[j]

        pruned_blocks = [candidates[i]["candidate"] for i in range(n) if bits[i] == 1]

        results.append(
            {
                "bitstring": "".join(str(bit) for bit in bits),
                "energy": float(energy),
                "compression": float(compression),
                "loss_penalty": float(loss_penalty),
                "pairwise_penalty": float(pairwise_penalty),
                "num_pruned_blocks": int(sum(bits)),
                "pruned_blocks": "; ".join(pruned_blocks),
            }
        )

    results.sort(key=lambda row: (row["energy"], row["loss_penalty"], -row["compression"], row["bitstring"]))
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
                "stage_index": "" if candidate.get("stage_index") is None else candidate["stage_index"],
                "block_index": "" if candidate.get("block_index") is None else candidate["block_index"],
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
            "stage_index",
            "block_index",
            "params",
            "loss_penalty",
            "compression_value",
            "raw_loss_penalty",
            "compression_base",
            "compression_source",
            "pruning_attractiveness",
        ],
    )


def export_interaction_edges(outdir: Path, edges: List[Dict[str, Any]], rho_interaction: float) -> None:
    rows: List[Dict[str, Any]] = []
    for edge in edges:
        rows.append(
            {
                "i": edge["i"],
                "j": edge["j"],
                "candidate_i": edge["candidate_i"],
                "candidate_j": edge["candidate_j"],
                "coefficient_base": edge["coefficient_base"],
                "rho_interaction": rho_interaction,
                "qubo_coefficient": rho_interaction * edge["coefficient_base"],
                "reason": edge["reason"],
            }
        )

    write_csv(
        outdir / "interaction_edges.csv",
        rows,
        [
            "i",
            "j",
            "candidate_i",
            "candidate_j",
            "coefficient_base",
            "rho_interaction",
            "qubo_coefficient",
            "reason",
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
            "pairwise_penalty",
            "num_pruned_blocks",
            "pruned_blocks",
        ],
    )


def build_circuit_complexity_report(
    n: int,
    formulation: str,
    hamiltonian_terms: Dict[str, Any],
    interaction_edges: List[Dict[str, Any]],
) -> Dict[str, Any]:
    old_all_to_all_zz = n * (n - 1) // 2
    new_zz = len(hamiltonian_terms.get("zz_terms", []))

    if old_all_to_all_zz > 0:
        reduction = 1.0 - (new_zz / old_all_to_all_zz)
    else:
        reduction = 0.0

    return {
        "formulation": formulation,
        "number_of_qubits": n,
        "z_terms": len(hamiltonian_terms.get("z_terms", [])),
        "zz_terms": new_zz,
        "legacy_all_to_all_zz_terms_for_same_n": old_all_to_all_zz,
        "zz_reduction_vs_legacy_all_to_all": reduction,
        "interaction_edges": len(interaction_edges),
        "why_this_matters": (
            "Each ZZ term becomes a two-qubit cost interaction in QAOA. "
            "The corrected sparse formulation avoids the complete graph produced "
            "by the old squared global target constraint."
        ),
    }


# ============================================================
# Main entry point
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build corrected QUBO/Hamiltonian files from cost_loss_table.csv."
    )

    parser.add_argument("--input", type=str, default="cost_loss_table.csv", help="Input pruning sensitivity CSV file.")
    parser.add_argument("--outdir", type=str, default="qubo_outputs", help="Directory where outputs will be saved.")
    parser.add_argument("--max-candidates", type=int, default=10, help="Maximum number of pruning candidates/qubits to use.")
    parser.add_argument(
        "--selection-method",
        type=str,
        default="attractiveness",
        choices=["attractiveness", "params", "low_loss"],
        help="How to select candidates for the QUBO subset.",
    )
    parser.add_argument(
        "--formulation",
        type=str,
        default="sparse_topology",
        choices=["sparse_topology", "legacy_target_square"],
        help="QUBO formulation. Default is corrected sparse topology-aware model.",
    )
    parser.add_argument("--alpha", type=float, default=1.0, help="Weight of loss penalty term.")
    parser.add_argument("--beta", type=float, default=0.08, help="Compression reward weight. Ignored if --auto-beta is active.")
    parser.add_argument("--no-auto-beta", action="store_true", help="Disable beta calibration and use --beta directly.")
    parser.add_argument("--beta-min", type=float, default=0.0, help="Minimum beta for auto calibration.")
    parser.add_argument("--beta-max", type=float, default=2.0, help="Maximum beta for auto calibration.")
    parser.add_argument("--beta-steps", type=int, default=2001, help="Number of beta grid points for auto calibration.")
    parser.add_argument("--rho-interaction", type=float, default=1.0, help="Weight multiplier for sparse pairwise pruning interactions.")
    parser.add_argument("--target-tolerance", type=float, default=0.06, help="Allowed compression band above target before sparse budget guard activates.")
    parser.add_argument("--budget-guard-weight", type=float, default=4.0, help="Weight of sparse pairwise budget guard terms.")
    parser.add_argument(
        "--interaction-method",
        type=str,
        default="same_stage",
        choices=["none", "adjacent", "same_stage"],
        help="How to build sparse pairwise interaction edges.",
    )
    parser.add_argument("--lambda-constraint", type=float, default=4.0, help="Legacy target-square constraint weight.")
    parser.add_argument("--target-compression", type=float, default=0.30, help="Requested compression share, e.g. 0.30 = 30 percent.")
    parser.add_argument("--energy-check-results", type=int, default=20, help="Number of lowest-energy bitstrings to export.")

    args = parser.parse_args()

    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive.")

    if not (0.0 <= args.target_compression <= 1.0):
        raise ValueError("--target-compression should be between 0 and 1.")

    input_path = resolve_input_path(args.input)
    outdir = Path(args.outdir)

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

    interaction_edges = build_interaction_edges(
        candidates,
        method=args.interaction_method,
        target_compression=args.target_compression,
        target_tolerance=args.target_tolerance,
        budget_guard_weight=args.budget_guard_weight,
    )
    beta_calibration: Optional[Dict[str, Any]] = None
    beta_used = args.beta

    if args.formulation == "sparse_topology":
        if not args.no_auto_beta:
            beta_calibration = calibrate_beta_for_target(
                candidates=candidates,
                interaction_edges=interaction_edges,
                alpha=args.alpha,
                rho_interaction=args.rho_interaction,
                target_compression=args.target_compression,
                beta_min=args.beta_min,
                beta_max=args.beta_max,
                beta_steps=args.beta_steps,
            )
            beta_used = float(beta_calibration["beta"])

        Q, qubo_terms = build_sparse_topology_qubo(
            candidates=candidates,
            interaction_edges=interaction_edges,
            alpha=args.alpha,
            beta=beta_used,
            rho_interaction=args.rho_interaction,
        )
    else:
        # Legacy mode is intentionally explicit. Auto-beta is not used because
        # beta inside the old equation shifts the target.
        beta_used = args.beta
        Q, qubo_terms = build_legacy_target_qubo(
            candidates=candidates,
            alpha=args.alpha,
            beta=beta_used,
            lambda_constraint=args.lambda_constraint,
            target_compression=args.target_compression,
        )

    hamiltonian_terms = qubo_to_ising_terms(Q=Q, qubo_constant=qubo_terms["constant"])
    complexity_report = build_circuit_complexity_report(
        n=len(candidates),
        formulation=args.formulation,
        hamiltonian_terms=hamiltonian_terms,
        interaction_edges=interaction_edges,
    )

    metadata = {
        "current_step": "Step 3 corrected: CSV -> sparse QUBO -> Ising Hamiltonian",
        "input_csv": str(input_path),
        "output_directory": str(outdir),
        "number_of_candidates": len(candidates),
        "selection_method": args.selection_method,
        "compression_source": compression_source_summary,
        "formulation": args.formulation,
        "alpha": args.alpha,
        "beta": beta_used,
        "beta_auto_calibrated": bool(beta_calibration is not None),
        "beta_calibration": beta_calibration,
        "rho_interaction": args.rho_interaction,
        "interaction_method": args.interaction_method,
        "target_tolerance": args.target_tolerance,
        "budget_guard_weight": args.budget_guard_weight,
        "upper_compression_band": args.target_compression + args.target_tolerance,
        "interaction_edges": len(interaction_edges),
        "lambda_constraint": args.lambda_constraint if args.formulation == "legacy_target_square" else None,
        "target_compression": args.target_compression,
        "binary_variable_meaning": {"0": "keep the candidate block", "1": "prune the candidate block"},
        "qubo_formula": qubo_terms["formula"],
        "qubo_energy_convention": "H(x) = constant + sum_i Q[i,i]*x_i + sum_{i<j} Q[i,j]*x_i*x_j",
        "matrix_note": "qubo_matrix.csv is upper-triangular. Lower-triangular values are zero by design.",
        "ising_mapping": "x_i = (1 - Z_i) / 2",
        "correction_summary": {
            "old_problem_1": "The old squared target term produced all-to-all ZZ couplings.",
            "old_problem_2": "The old -beta compression reward shifted the effective target above C_target.",
            "new_solution": "Use sparse topology-aware interactions, a sparse pairwise budget guard, and calibrated beta to reach the target compression region.",
        },
        "next_step": "Re-run QAOA with the new hamiltonian_terms.json. Old qaoa_result.json is no longer valid.",
    }

    export_selected_candidates(outdir, candidates)
    export_interaction_edges(outdir, interaction_edges, args.rho_interaction)
    export_qubo_matrix(outdir, Q)

    write_json(outdir / "qubo_terms.json", qubo_terms)
    write_json(outdir / "hamiltonian_terms.json", hamiltonian_terms)
    write_json(outdir / "qubo_metadata.json", metadata)
    write_json(outdir / "circuit_complexity_report.json", complexity_report)

    # Remove stale QAOA result from old Hamiltonian, if present.
    stale_qaoa_result = outdir / "qaoa_result.json"
    if stale_qaoa_result.exists():
        stale_qaoa_result.unlink()

    if len(candidates) <= 20 and args.energy_check_results > 0:
        energy_rows = brute_force_energy_check(
            Q=Q,
            constant=qubo_terms["constant"],
            candidates=candidates,
            max_results=args.energy_check_results,
        )
        export_energy_check(outdir, energy_rows)
    else:
        energy_rows = []

    print("\nCorrected QUBO / Hamiltonian construction completed.")
    print(f"Input CSV              : {input_path}")
    print(f"Output directory       : {outdir}")
    print(f"Candidates / qubits    : {len(candidates)}")
    print(f"Formulation            : {args.formulation}")
    print(f"Compression source     : {compression_source_summary}")
    print(f"Requested target       : {args.target_compression:.2%}")
    print(f"Alpha loss weight      : {args.alpha}")
    print(f"Beta compression weight: {beta_used}")
    print(f"Rho interaction weight : {args.rho_interaction}")
    print(f"Interaction method     : {args.interaction_method}")
    print(f"Upper target band      : {args.target_compression + args.target_tolerance:.2%}")
    print(f"Budget guard weight    : {args.budget_guard_weight}")
    print(f"Sparse ZZ terms        : {complexity_report['zz_terms']}")
    print(f"Legacy all-to-all ZZ   : {complexity_report['legacy_all_to_all_zz_terms_for_same_n']}")

    if beta_calibration:
        print("\nBeta calibration:")
        print(f"- selected bitstring   : {beta_calibration['selected_bitstring_during_calibration']}")
        print(f"- compression          : {beta_calibration['selected_compression_during_calibration']:.4f}")
        print(f"- beta interval        : {beta_calibration['beta_interval_for_selected_bitstring']}")

    if energy_rows:
        best = energy_rows[0]
        print("\nClassical ground-state sanity check:")
        print(f"- bitstring            : {best['bitstring']}")
        print(f"- energy               : {best['energy']:.6f}")
        print(f"- compression          : {best['compression']:.4f}")
        print(f"- loss penalty         : {best['loss_penalty']:.6f}")
        print(f"- pruned blocks        : {best['pruned_blocks']}")

    print("\nGenerated files:")
    print(f"- {outdir / 'selected_candidates.csv'}")
    print(f"- {outdir / 'interaction_edges.csv'}")
    print(f"- {outdir / 'qubo_matrix.csv'}")
    print(f"- {outdir / 'qubo_terms.json'}")
    print(f"- {outdir / 'hamiltonian_terms.json'}")
    print(f"- {outdir / 'qubo_metadata.json'}")
    print(f"- {outdir / 'circuit_complexity_report.json'}")

    if len(candidates) <= 20 and args.energy_check_results > 0:
        print(f"- {outdir / 'qubo_energy_check.csv'}")

    print("\nImportant:")
    print("This step rebuilt the mathematical model. Re-run QAOA after this change.")


if __name__ == "__main__":
    main()
