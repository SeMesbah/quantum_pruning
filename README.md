# Quantum Pruning

This project formulates block-level neural-network pruning as a small QUBO / Ising Hamiltonian problem and prepares the Hamiltonian for Qiskit QAOA experiments.

The current model uses a ConvNeXT-tiny vision transformer fine-tuned for Canadian street-view city classification. Each candidate block becomes one binary pruning decision:

```text
x_i = 0  keep block i
x_i = 1  prune block i
```

---

## Corrected QUBO Formulation

The earlier formulation was algebraically valid, but not a good QAOA formulation:

```text
H_old(x) = alpha * sum(L_i x_i)
           - beta * sum(C_i x_i)
           + lambda * (sum(C_i x_i) - C_target)^2
```

The problem is the squared global target term. After expansion it connects every candidate to every other candidate. With 8 candidates this creates 28 ZZ terms. With more candidates it grows as:

```text
n(n-1)/2
```

That makes the QAOA cost circuit dense. The `- beta * sum(C_i x_i)` term also shifts the effective compression optimum above the stated target.

The corrected default formulation is sparse and topology-aware:

```text
H(x) = sum_i (alpha * L_i - beta * C_i) * x_i
       + sum_(i,j in E_sparse) q_ij * x_i * x_j
```

where:

| Symbol | Meaning |
|---|---|
| `L_i` | normalized loss penalty if candidate block `i` is pruned |
| `C_i` | normalized compression value of candidate block `i` |
| `alpha` | weight of model-quality damage |
| `beta` | compression reward, auto-calibrated by default |
| `E_sparse` | sparse interaction edges, not all possible pairs |
| `q_ij` | pairwise penalty for risky joint pruning |

The sparse interaction set contains two kinds of edges:

1. **Topology guard** — penalizes pruning multiple blocks from the same model stage.
2. **Budget guard** — penalizes pairs that exceed the allowed compression band above the target.

The target compression is therefore handled by:

```text
beta calibration + sparse budget guard
```

instead of one dense all-to-all squared penalty.

---

## Current Corrected Result

Using the current 8 valid candidates and target compression `30%`, the corrected script produces:

```text
best bitstring: 10000000
compression:   35.03%
loss penalty:  0.027674
pruned block:  stages.3.blocks.2
```

This is much closer to the intended target than the old result:

```text
old result: 11000000
old compression: 43.87%
```

The circuit is also less dense:

```text
old all-to-all ZZ terms: 28
new sparse ZZ terms:    13
```

The result is not exactly 30% because the available candidates are discrete. The closest useful low-damage option is one large block with approximately 35% of the selected candidate mass.

---

## Workflow

```text
trained image-classification model
        ↓
block sensitivity analysis  (qnn_and_pruning.ipynb → cost_loss_table.csv)
        ↓
corrected sparse QUBO model  (qubo_hamiltonian.py → qubo_outputs/)
        ↓
Ising Hamiltonian representation  (hamiltonian_terms.json)
        ↓
standard Qiskit QAOA  (qaoa_experiment.ipynb → best bitstring)
        ↓
genuine structural block removal → smaller model
        ↓
evaluate pruned model accuracy, loss and F1 score
        ↓
upload pruned model to Hugging Face
```

### Two QAOA paths

| Notebook | Role | QAOA engine |
|---|---|---|
| **`qaoa_experiment.ipynb`** | **Main pipeline** — QUBO → QAOA → structural prune → publish to HF | official `qiskit_algorithms.QAOA` (library-standard, hardware-ready) |
| `qaoa_rank_masks.ipynb` + `top_k_mask_evaluation.ipynb` | Reference path — ranks/evaluates masks | hand-rolled exact statevector |
| `classical_pruning.ipynb` + `quantum_vs_classical_comparison.ipynb` | Classical greedy baseline + head-to-head comparison | — |

Both QAOA notebooks read the **same** Step-3 QUBO outputs (`hamiltonian_terms.json`,
`selected_candidates.csv`, `qubo_energy_check.csv`).

---

## Step 3: Build the Corrected QUBO / Hamiltonian

Run:

```bash
python qubo_hamiltonian.py \
  --input cost_loss_table.csv \
  --outdir qubo_outputs \
  --max-candidates 10 \
  --target-compression 0.30
```

The default mode is now:

```text
--formulation sparse_topology
--interaction-method same_stage
--target-tolerance 0.06
--budget-guard-weight 4.0
```

This means the optimizer targets approximately 30% compression, allows a small upper band, and discourages pairwise over-pruning without creating a complete graph.

---

## Generated Files

| File | Meaning |
|---|---|
| `cost_loss_table.csv` | sensitivity table used as input |
| `qubo_outputs/selected_candidates.csv` | selected pruning blocks and qubit indices |
| `qubo_outputs/interaction_edges.csv` | sparse topology/budget edges used in the QUBO |
| `qubo_outputs/qubo_matrix.csv` | upper-triangular QUBO matrix |
| `qubo_outputs/qubo_terms.json` | QUBO formula terms |
| `qubo_outputs/hamiltonian_terms.json` | Ising Hamiltonian terms for Qiskit |
| `qubo_outputs/qubo_metadata.json` | formulation settings and beta calibration details |
| `qubo_outputs/circuit_complexity_report.json` | old-vs-new circuit density comparison |
| `qubo_outputs/qubo_energy_check.csv` | brute-force sanity check for the small 8-qubit case |

`qaoa_result.json` from the old Hamiltonian was intentionally removed. Re-run the QAOA notebook/script after rebuilding the Hamiltonian.

---

## QUBO to Ising Mapping

The Qiskit Hamiltonian uses the standard binary-to-Pauli mapping:

```text
x_i = (1 - Z_i) / 2
```

The Ising form is:

```text
H = constant + sum_i h_i Z_i + sum_(i<j) J_ij Z_i Z_j
```

This part of the mathematics was already correct. The correction is in the pruning objective design, not in the Pauli mapping.

---

## Legacy Formulation

The old target-square model is still available only for comparison:

```bash
python qubo_hamiltonian.py \
  --formulation legacy_target_square \
  --beta 1.0 \
  --lambda-constraint 4.0
```

Use this only to reproduce the previous result. It is not the recommended QAOA formulation.

---

## Setup

```bash
pip install -r requirements.txt
```

Requirements:

```text
python >= 3.9
numpy
qiskit
qiskit-algorithms
matplotlib
pandas
```
