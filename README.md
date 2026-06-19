# Quantum Pruning

Applying pruning and compression techniques to a SwinV2 Vision Transformer fine-tuned on Canadian street-view imagery for city classification.

The current project direction is to formulate **block-level neural-network pruning** as a **QUBO / Hamiltonian optimization problem**, and later solve a small version of this problem using **Qiskit quantum simulation**.

## Model

- **Architecture:** SwinV2-Base (window 12, 192×192 input)
- **Source:** `canada-guesser/canadian_streetview_cities_models` on Hugging Face
- **Task:** Classify street-view images into one of 15 Canadian cities

### Supported Cities

Calgary, Charlottetown, Edmonton, Halifax, Hamilton, Kitchener-Waterloo, Montreal, Ottawa-Gatineau, Quebec City, Saskatoon, St. John's, Toronto, Vancouver, Winnipeg, Victoria

---

## Current Project Workflow

The intended software and data-processing workflow is:

```text
trained image-classification model
        ↓
block sensitivity analysis
        ↓
cost_loss_table.csv
        ↓
QUBO mathematical model
        ↓
Ising Hamiltonian representation
        ↓
Qiskit QAOA quantum simulation
        ↓
best measured bitstring / pruning mask
        ↓
apply pruning mask to the model
        ↓
evaluate pruned model accuracy, loss and F1 score
```

### Current Step

The project is currently at this stage:

```text
cost_loss_table.csv
        ↓
QUBO mathematical model
        ↓
Ising Hamiltonian representation
```

This step is implemented in:

```text
qubo_hamiltonian.py
```

This file **does not run Qiskit yet** and **does not prune the model yet**. Its purpose is only to build the mathematical optimization problem that the later Qiskit simulation will use.

---

## Why QUBO / Hamiltonian Pruning?

Pruning means removing parts of the neural network to make it smaller and cheaper to run. The problem is that not every block is equally safe to remove.

Some blocks may contain many parameters and are attractive to prune. Other blocks may be very important for accuracy, so pruning them causes a large performance drop.

The pruning decision can therefore be written as an optimization problem:

> choose which blocks to prune so that the model becomes smaller, but the prediction quality does not collapse.

This is suitable for a QUBO / Hamiltonian formulation because each pruning decision is binary.

---

## Binary Pruning Variables

For every candidate block, we define one binary variable:

```text
x_i = 0  keep block i
x_i = 1  prune block i
```

For example, if the optimizer returns:

```text
01010011
```

then this means:

```text
block 1: keep
block 2: prune
block 3: keep
block 4: prune
block 5: keep
block 6: keep
block 7: prune
block 8: prune
```

So the final bitstring is the pruning mask.

---

## Input Data

The file:

```text
cost_loss_table.csv
```

contains candidate pruning blocks. Important columns are:

| Column | Meaning |
|---|---|
| `candidate` | name of the model block |
| `params` | number of parameters in the block |
| `loss_increase_raw` | how much the loss increased when the block was pruned alone |
| `accuracy_drop_raw` | how much accuracy dropped when the block was pruned alone |
| `f1_drop_raw` | how much macro-F1 dropped when the block was pruned alone |
| `L_i` | normalized loss penalty |
| `C_i` | normalized compression value |
| `pruning_attractiveness` | heuristic score: high compression and low loss is attractive |

In the QUBO model, we mainly use:

```text
L_i = loss penalty of pruning block i
C_i = compression value of pruning block i
```

---

## Mathematical Model

The QUBO objective is written as an energy function:

```text
H(x) = α · Σ(L_i · x_i)
       - β · Σ(C_i · x_i)
       + λ · (Σ(C_i · x_i) - C_target)^2
```

where:

| Symbol | Meaning |
|---|---|
| `x_i` | binary pruning decision for block `i` |
| `L_i` | loss penalty if block `i` is pruned |
| `C_i` | compression benefit if block `i` is pruned |
| `α` | weight of the loss penalty |
| `β` | weight of the compression reward |
| `λ` | weight of the target-compression constraint |
| `C_target` | desired compression level among the selected candidates |

The goal is to find the bitstring with the lowest energy.

Low energy means:

```text
small accuracy/loss damage
+ useful parameter reduction
+ compression close to the target
```

---

## Meaning of Each Formula Term

### 1. Loss Penalty Term

```text
α · Σ(L_i · x_i)
```

This term punishes dangerous pruning decisions.

If `L_i` is high, then pruning block `i` is risky. When `x_i = 1`, this term increases the energy. Since the optimizer minimizes energy, it avoids pruning blocks with high loss penalty.

In simple words:

```text
Do not prune blocks that damage the model too much.
```

---

### 2. Compression Reward Term

```text
- β · Σ(C_i · x_i)
```

This term rewards pruning useful blocks.

If `C_i` is high, then pruning block `i` saves many parameters. The minus sign is important because we minimize energy. A larger compression value lowers the energy and makes that pruning decision more attractive.

In simple words:

```text
Prefer pruning blocks that save many parameters.
```

---

### 3. Target Compression Constraint

```text
λ · (Σ(C_i · x_i) - C_target)^2
```

This term prevents the optimizer from pruning too little or too much.

If the selected pruning mask gives compression close to `C_target`, this term is small. If the selected pruning mask is far away from the target, this term becomes large.

In simple words:

```text
Try to prune around the desired compression level.
```

Example:

```text
C_target = 0.30
```

means:

```text
try to prune about 30% of the selected candidate parameter mass.
```

---

## Why This Is a QUBO

QUBO means:

```text
Quadratic Unconstrained Binary Optimization
```

The problem is binary because every `x_i` is either `0` or `1`.

The problem becomes quadratic because the target-compression term is squared:

```text
(Σ(C_i · x_i) - C_target)^2
```

When expanded, this creates terms like:

```text
x_i · x_j
```

These terms describe interactions between pruning choices.

For example:

```text
pruning block A alone may be acceptable
pruning block B alone may be acceptable
pruning A and B together may exceed the compression target
```

This is why the QUBO is more meaningful than simply sorting the CSV by attractiveness.

---

## QUBO Matrix Form

The same energy can be written as:

```text
H(x) = constant + Σ Q_ii · x_i + Σ Q_ij · x_i · x_j
```

where:

```text
Q_ii = αL_i - βC_i + λ(C_i^2 - 2C_targetC_i)
```

and for `i < j`:

```text
Q_ij = 2λC_iC_j
```

The constant term is:

```text
constant = λC_target^2
```

This matrix is exported by `qubo_hamiltonian.py` as:

```text
qubo_outputs/qubo_matrix.csv
```

---

## From QUBO to Hamiltonian

For Qiskit, the binary variables are later converted into Pauli-Z operators using:

```text
x_i = (1 - Z_i) / 2
```

This transforms the QUBO into an Ising Hamiltonian:

```text
H = constant + Σ h_i Z_i + Σ J_ij Z_i Z_j
```

where:

| Term | Meaning |
|---|---|
| `Z_i` | Pauli-Z operator acting on qubit `i` |
| `h_i` | single-qubit coefficient |
| `J_ij` | two-qubit interaction coefficient |

This Hamiltonian is exported as:

```text
qubo_outputs/hamiltonian_terms.json
```

This is the file that the next Qiskit simulation step should read.

---

## Step 3: Build QUBO / Hamiltonian Files

Run:

```bash
python qubo_hamiltonian.py --input cost_loss_table.csv --outdir qubo_outputs --max-candidates 10
```

Current note: the present `cost_loss_table.csv` contains 8 valid block candidates, so the script will use 8 candidates for now. When the sensitivity analysis is extended to 10 candidates, the same command will use 10.

Generated files:

```text
qubo_outputs/selected_candidates.csv
qubo_outputs/qubo_matrix.csv
qubo_outputs/qubo_terms.json
qubo_outputs/hamiltonian_terms.json
qubo_outputs/qubo_metadata.json
qubo_outputs/qubo_energy_check.csv
```

### Output Meaning

| File | Meaning |
|---|---|
| `selected_candidates.csv` | selected pruning blocks and their qubit indices |
| `qubo_matrix.csv` | QUBO coefficients in matrix form |
| `qubo_terms.json` | QUBO formula terms |
| `hamiltonian_terms.json` | Ising Hamiltonian terms for later Qiskit use |
| `qubo_metadata.json` | settings such as `α`, `β`, `λ`, and `C_target` |
| `qubo_energy_check.csv` | small classical sanity check of low-energy bitstrings |

The energy check is not the final quantum result. It is only a sanity check to verify that the mathematical model behaves correctly before moving to Qiskit.

---

## Next Planned Step: Qiskit Simulation

The next file should be separate, for example:

```text
run_qaoa_qiskit.py
```

Its job will be:

```text
read hamiltonian_terms.json
        ↓
construct Qiskit Hamiltonian operator
        ↓
run QAOA using Qiskit Aer simulator
        ↓
measure candidate bitstrings
        ↓
select the lowest-energy bitstring
        ↓
export pruning_mask.json
```

This is the step where the project becomes an actual Qiskit quantum simulation.

---

## Legacy Weight-Level Quantum-Biased Pruning Prototype

The repository also contains an earlier pruning idea in `run_pruning.py` and `quantum_pruning.ipynb`.

That pipeline uses a small entangled quantum circuit simulated via **Qiskit Aer** to produce a quantum-biased importance score for individual weights. The circuit has three stages:

1. **Ry amplitude encoding** — maps each weight magnitude to a qubit excitation probability: `θᵢ = 2·arcsin(√wᵢ_norm)`
2. **Linear CX entanglement** — correlates adjacent qubits so each weight's importance is influenced by its neighbours
3. **Grover diffusion step** (`2|ψ⟩⟨ψ| − I`) — amplifies the gap between high- and low-magnitude weights

Measured per-qubit probabilities `Q` modulate the classical L1-magnitude score:

```text
importance_i = |w_i| · (1 + λ · Q_i)
mask         = importance_i >= percentile(sparsity)
```

This part is useful as an experimental prototype, but the clearer and more defendable project direction is the QUBO/Hamiltonian block-pruning workflow described above.

---

## Sparsity Schedule Used by the Legacy Prototype

| Layer type | Target sparsity | Rationale |
|------------|----------------|-----------|
| `attn` (Q/K/V + proj) | 15 % | Information bottleneck — prune conservatively |
| `mlp` (fc1 / fc2) | 40 % | High redundancy in feed-forward blocks |
| `head` (classifier) | 10 % | Final decision layer — minimal pruning |
| everything else | 30 % | Default |

---

## Existing Legacy Prototype Results

Pruned with `quantum_bias = 0.25`, `n_qubits = 6`, `shots = 512`.

| Metric | Value |
|--------|-------|
| Total parameters | 86,909,191 |
| Zeroed parameters | 27,382,295 |
| Overall sparsity (all params) | 31.5 % |
| Linear-layer sparsity | 31.6 % |

### Inference comparison: Vancouver street-view image

| Rank | Original | Pruned |
|------|----------|--------|
| 1 | Vancouver 100.0 % | Vancouver 100.0 % |
| 2 | Toronto   0.0 %  | Toronto   0.0 %  |
| 3 | Montreal  0.0 %  | Montreal  0.0 %  |

The top-1 prediction is preserved at full confidence after removing approximately 31% of all weights in this single-image test.

---

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Step 3: QUBO / Hamiltonian construction

```bash
python qubo_hamiltonian.py --input cost_loss_table.csv --outdir qubo_outputs --max-candidates 10
```

### Legacy standalone script

```bash
python run_pruning.py
```

### Legacy notebook

Run the four cells of `quantum_pruning.ipynb` in order:

1. Install dependencies
2. Load the SwinV2 model and run baseline inference
3. Define `QuantumPruner` with Qiskit circuit and mask logic
4. Apply sparsity and compare original vs. pruned predictions

Point `Image.open(...)` in cell 2 to your own image before running.

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt)
