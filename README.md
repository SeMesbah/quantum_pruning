# Quantum Pruning

Applying pruning and compression techniques to a SwinV2 Vision Transformer fine-tuned on Canadian street-view imagery for city classification.

## Model

- **Architecture:** SwinV2-Base (window 12, 192×192 input)
- **Source:** `canada-guesser/canadian_streetview_cities_models` on Hugging Face
- **Task:** Classify street-view images into one of 15 Canadian cities

### Supported Cities

Calgary, Charlottetown, Edmonton, Halifax, Hamilton, Kitchener-Waterloo, Montreal, Ottawa-Gatineau, Quebec City, Saskatoon, St. John's, Toronto, Vancouver, Victoria, Winnipeg

## Quantum Pruning

The pruning pipeline uses a small entangled quantum circuit simulated via **Qiskit Aer** to produce a quantum-biased importance score for each weight. The circuit has three stages:

1. **Ry amplitude encoding** — maps each weight magnitude to a qubit excitation probability: `θᵢ = 2·arcsin(√wᵢ_norm)`
2. **Linear CX entanglement** — correlates adjacent qubits so each weight's importance is influenced by its neighbours (something classical L1/L2 scoring cannot express)
3. **Grover diffusion step** (`2|ψ⟩⟨ψ| − I`) — amplifies the gap between high- and low-magnitude weights, making the pruning threshold more decisive

Measured per-qubit probabilities `Q` modulate the classical L1-magnitude score:

```
importance_i = |w_i| · (1 + λ · Q_i)
mask         = importance_i >= percentile(sparsity)
```

Large layers use **stratified magnitude sampling** (up to `max_circuits` representative chunks drawn across the magnitude range) followed by linear interpolation, so circuit count stays bounded regardless of layer size.

### Circuit (6-qubit example)

```
       ┌────────────┐     ┌───┐┌───┐                         ┌───┐┌───┐      ░ ┌─┐
  q_0: ┤ Ry(2.6222) ├──■──┤ H ├┤ X ├──── ··· ───────────■───┤ X ├┤ H ├──────░─┤M├
       ┤ Ry(0.7433) ├┤X ├──■──┤ H ├┤ X ├─ ··· ──────────■───┤ X ├┤ H ├──────░──╫─┤M├
       ...
```

Depth 14 · 6 qubits · gates: Ry×6, CX×5, H×14, X×12, MCX×1

## Sparsity Schedule

| Layer type | Target sparsity | Rationale |
|------------|----------------|-----------|
| `attn` (Q/K/V + proj) | 15 % | Information bottleneck — prune conservatively |
| `mlp` (fc1 / fc2) | 40 % | High redundancy in feed-forward blocks |
| `head` (classifier) | 10 % | Final decision layer — minimal pruning |
| everything else | 30 % | Default |

## Results

Pruned with `quantum_bias = 0.25`, `n_qubits = 6`, `shots = 512`.

| Metric | Value |
|--------|-------|
| Total parameters | 86,909,191 |
| Zeroed parameters | 27,382,295 |
| Overall sparsity (all params) | 31.5 % |
| Linear-layer sparsity | 31.6 % |

### Inference comparison (Vancouver street-view image)

| Rank | Original | Pruned |
|------|----------|--------|
| 1 | Vancouver 100.0 % | Vancouver 100.0 % |
| 2 | Toronto   0.0 %  | Toronto   0.0 %  |
| 3 | Montreal  0.0 %  | Montreal  0.0 %  |

The top-1 prediction is preserved at full confidence after removing ~31 % of all weights.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Notebook (interactive)

Run the four cells of `quantum_pruning.ipynb` in order:

1. Install dependencies
2. Load the SwinV2 model and run baseline inference
3. Define `QuantumPruner` (Qiskit circuit + mask logic)
4. Apply 30 % sparsity and compare original vs. pruned predictions

Point `Image.open(...)` in cell 2 to your own image before running.

### Standalone script

```bash
python run_pruning.py
```

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt)
