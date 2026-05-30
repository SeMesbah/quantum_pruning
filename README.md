# Quantum Pruning

Applying pruning and compression techniques to a SwinV2 Vision Transformer fine-tuned on Canadian street-view imagery for city classification.

## Model

- **Architecture:** SwinV2-Base (window 12, 192×192 input)
- **Source:** `canada-guesser/canadian_streetview_cities_models` on Hugging Face
- **Task:** Classify street-view images into one of 15 Canadian cities

### Supported Cities

Calgary, Charlottetown, Edmonton, Halifax, Hamilton, Kitchener-Waterloo, Montreal, Ottawa-Gatineau, Quebec City, Saskatoon, St. John's, Toronto, Vancouver, Victoria, Winnipeg

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Open and run `quantum_pruning.ipynb` in Jupyter. Point `Image.open(...)` to your target image before running inference.

## Requirements

- Python 3.9+
- See [requirements.txt](requirements.txt)
