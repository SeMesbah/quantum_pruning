# Quantum Pruning

Applying pruning and compression techniques to a SwinV2 Vision Transformer fine-tuned on Canadian street-view imagery for city classification.

## Model

- **Architecture:** SwinV2-Base (window 12, 192×192 input)
- **Source:** `canada-guesser/canadian_streetview_cities_models` on Hugging Face
- **Task:** Classify street-view images into one of 15 Canadian cities

### Supported Cities

Calgary, Charlottetown, Edmonton, Halifax, Hamilton, Kitchener-Waterloo, Montreal, Ottawa-Gatineau, Quebec City, Saskatoon, St. John's, Toronto, Vancouver, Victoria, Winnipeg

## Quantum Pruning

The pruning pipeline uses a small entangled quantum circuit (Hadamard + CNOT chain) simulated via **Qiskit Aer** to produce a quantum-biased importance score for each weight. These scores modulate the classical L1-magnitude importance, shifting which weights fall below the pruning threshold. The result is a sparse model whose pruning mask has a quantum-random component rather than being purely magnitude-driven.

```
importance_combined = |w| * (1 + 0.2 * q_bias)
mask = importance_combined >= percentile(sparsity)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run the four cells of `quantum_pruning.ipynb` in order:

1. Install dependencies
2. Load the SwinV2 model and run baseline inference
3. Define `apply_quantum_pruning()` (Qiskit circuit + mask logic)
4. Apply 30 % sparsity and compare original vs. pruned predictions

Point `Image.open(...)` in cell 2 to your own image before running.

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt)
