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

The interaction edges (`E_sparse`) come from up to three sources, controlled by `--interaction-method`:

1. **Topology guard** — penalizes pruning multiple blocks from the same model stage.
2. **Budget guard** — penalizes pairs that exceed the allowed compression band above the target.
3. **`all_pairs` (default)** — every candidate pair gets an edge (45 for 10 candidates), so the formula estimate above can be blended with real measured data for every pair, not just a hand-picked subset.

The target compression is therefore handled by `beta` calibration + the pairwise guards, instead of one dense all-to-all squared penalty.

### Measured pairwise interaction data (default: on)

`q_ij` doesn't have to be a pure formula guess. `pairwise_sensitivity.ipynb` actually bypasses **two** blocks at once (all 45 pairs among the 10 candidates) and measures the real joint damage on the trained model — the same technique the single-block sensitivity step uses, extended to pairs. `qubo_hamiltonian.py` blends this real measurement into each edge's weight by default (`--measured-pairwise-csv qubo_outputs/pairwise_sensitivity.csv`), averaging it with the formula estimate. Negative measured gaps (a pair that turned out safer together than the naive sum predicted) are kept as real, not clipped to zero.

### Loss metric (`--loss-metric`, default: `accuracy`)

`L_i` — and, when blending, the measured half of `q_ij` — can be built from validation-loss increase (`loss`), real accuracy drop (`accuracy`), or real macro-F1 drop (`f1`). The two must share a basis to stay unit-consistent (`qubo_hamiltonian.py` enforces this automatically). Empirically, building the whole equation on **accuracy** and blending with measured pairwise accuracy-drop data gives the best real-model result of every variant tried — see below.

---

## Current Result (default settings: `all_pairs`, `--loss-metric accuracy`, measured blend, `target-compression 0.40`)

| variant | blocks pruned | real accuracy drop | real F1 drop | real param reduction |
|---|---|---|---|---|
| formula-only, sparse (`same_stage`, 12 edges) | 3 blocks | 4.00% | 4.48% | 22.5% |
| formula-only, dense (`all_pairs`, 45 edges) | 2 blocks | 3.17% | 3.37% | 21.4% |
| measured blend, loss-basis | 2 blocks (different pair) | 3.83% | 4.06% | 21.4% |
| **measured blend, accuracy-basis (current default)** | `stages.2.blocks.2` + `stages.3.blocks.2` | **3.17%** | **3.37%** | **21.4%** |

The accuracy-basis blend was chosen as the default because it consistently reproduces the best real-model result across independent runs, while being grounded in actual measured pairwise damage rather than a formula guess alone. QAOA (`qiskit_algorithms.QAOA`, Aer-backed) verifiably finds this Hamiltonian's true global optimum (`qaoa_result.json → validation.found_global_optimum`) in a few seconds.

The result is not exactly 40% because the candidates are discrete blocks of fixed size — see `qubo_outputs/qubo_energy_check.csv` for the full ranked list of reachable compressions.

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
