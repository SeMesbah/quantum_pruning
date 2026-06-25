"""
top_k_mask_evaluation.py

Evaluate the top-k QUBO/QAOA pruning masks on the real model.

Purpose:
    The QUBO/Hamiltonian gives a ranked list of pruning masks.
    This script checks whether the top-ranked masks actually perform well
    when applied to the neural network.

Default workflow:
    1. Read qubo_outputs/qubo_energy_check.csv
    2. Take top 5 masks by lowest QUBO energy
    3. For each mask:
        - load a fresh baseline model
        - apply only this mask
        - evaluate on validation/test subset
    4. Save comparison table

Run:
    python top_k_mask_evaluation.py

Example:
    python top_k_mask_evaluation.py \
        --top-k 5 \
        --model convnext \
        --split test \
        --max-samples 600 \
        --batch-size 16

Important:
    This script uses the same block-bypass pruning simulation used in the
    sensitivity analysis. It validates the mask behavior, but it does not yet
    permanently export a smaller compressed model file.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm

import timm
from datasets import load_dataset
from huggingface_hub import hf_hub_download


# ============================================================
# Constants
# ============================================================

CLASS_NAMES = [
    "Calgary", "Charlottetown", "Edmonton", "Halifax", "Hamilton",
    "Kitchener-Waterloo", "Montreal", "Ottawa-Gatineau", "Quebec City",
    "Saskatoon", "St Johns", "Toronto", "Vancouver", "Victoria", "Winnipeg",
]


# ============================================================
# Basic helpers
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV file is empty: {path}")

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


# ============================================================
# Image preprocessing
# ============================================================

def resize_and_pad(img: Image.Image, target_size=(320, 320)) -> Image.Image:
    """
    Same preprocessing style as the original ConvNeXT inference pipeline.
    """
    img = img.copy()
    img.thumbnail(target_size, Image.Resampling.LANCZOS)

    new_img = Image.new("RGB", target_size, (0, 0, 0))
    left = (target_size[0] - img.size[0]) // 2
    top = (target_size[1] - img.size[1]) // 2
    new_img.paste(img, (left, top))

    return new_img


def get_transform(model_name: str):
    if model_name == "convnext":
        return v2.Compose([
            v2.Lambda(lambda img: resize_and_pad(img)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    if model_name == "swinv2":
        return transforms.Compose([
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5),
            ),
        ])

    raise ValueError(f"Unknown model name: {model_name}")


# ============================================================
# Dataset
# ============================================================

class StreetViewSubset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.data = list(hf_dataset)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]

        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)

        img = img.convert("RGB")
        x = self.transform(img)
        y = int(row["label"])

        return x, y


def build_dataloader(
    model_name: str,
    split: str,
    max_samples: int,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    transform = get_transform(model_name)

    if max_samples > 0:
        split_expr = f"{split}[:{max_samples}]"
    else:
        split_expr = split

    ds = load_dataset(
        "canada-guesser/Canadian-streetview-cities",
        split=split_expr,
    )

    wrapped = StreetViewSubset(ds, transform)

    return DataLoader(
        wrapped,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# Model loading
# ============================================================

def torch_load_compatible(path: str, device: torch.device):
    """
    Handles both newer and older PyTorch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_finetuned_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "convnext":
        path = hf_hub_download(
            repo_id="canada-guesser/canadian_streetview_cities_models",
            filename="cnn_model/convnext_tiny_set_3_final.bin",
        )

        model = timm.create_model(
            "convnext_tiny",
            pretrained=False,
            num_classes=len(CLASS_NAMES),
        )

        checkpoint = torch_load_compatible(path, device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)

    elif model_name == "swinv2":
        path = hf_hub_download(
            repo_id="canada-guesser/canadian_streetview_cities_models",
            filename="vit_model/swinv2_base_window12_192_0_finetuned_canadian_streetview.bin",
        )

        model = timm.create_model(
            "swinv2_base_window12_192",
            pretrained=False,
            num_classes=len(CLASS_NAMES),
        )

        state_dict = torch_load_compatible(path, device)
        model.load_state_dict(state_dict)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.to(device)
    model.eval()

    return model


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()

    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    total = 0
    preds: List[int] = []
    labels: List[int] = []

    for x, y in tqdm(loader, desc="evaluate", leave=False):
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        if hasattr(logits, "logits"):
            logits = logits.logits

        loss = criterion(logits, y)

        total_loss += float(loss.item())
        total += int(y.numel())

        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        labels.extend(y.detach().cpu().tolist())

    return {
        "loss": total_loss / max(total, 1),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "n_samples": total,
    }


def count_trainable_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ============================================================
# Mask / pruning helpers
# ============================================================

class CandidateBypassWrapper(nn.Module):
    """
    Simulates structured block pruning by bypassing a block.

    bypass behavior:
        output = input

    This works for ConvNeXT blocks where input and output shapes match.
    It is the same idea as used during the sensitivity analysis.
    """

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        return x


def get_parent_and_child(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")

    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    child_key = parts[-1]
    return parent, child_key


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    parts = module_name.split(".")

    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)

    return module


def replace_module(model: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, child_key = get_parent_and_child(model, module_name)

    if child_key.isdigit():
        parent[int(child_key)] = new_module
    else:
        setattr(parent, child_key, new_module)


def apply_pruning_mask(model: nn.Module, pruned_blocks: List[str]) -> None:
    """
    Applies one mask to the model by replacing each selected block
    with a bypass wrapper.

    This modifies the model in-place.
    """
    for block_name in pruned_blocks:
        block_name = block_name.strip()
        if not block_name:
            continue

        original_module = get_module_by_name(model, block_name)
        replace_module(model, block_name, CandidateBypassWrapper(original_module))


def parse_pruned_blocks(text: str) -> List[str]:
    if text is None:
        return []

    text = str(text).strip()

    if text == "":
        return []

    return [part.strip() for part in text.split(";") if part.strip()]


# ============================================================
# QUBO result loading
# ============================================================

def load_selected_candidate_params(path: Path) -> Dict[str, int]:
    """
    Reads selected_candidates.csv and returns:
        candidate name -> params
    """
    rows = read_csv(path)

    params_by_candidate: Dict[str, int] = {}

    for row in rows:
        name = str(row.get("candidate", "")).strip()
        params = safe_int(row.get("params", 0), default=0)

        if name:
            params_by_candidate[name] = params

    return params_by_candidate


def load_top_k_masks(
    energy_path: Path,
    top_k: int,
    exclude_empty_mask: bool,
) -> List[Dict[str, Any]]:
    """
    Loads masks from qubo_energy_check.csv and returns top-k by energy.
    """
    rows = read_csv(energy_path)

    cleaned: List[Dict[str, Any]] = []

    for row in rows:
        mask = str(row.get("bitstring", "")).strip()
        pruned_blocks = parse_pruned_blocks(str(row.get("pruned_blocks", "")))

        if exclude_empty_mask and len(pruned_blocks) == 0:
            continue

        cleaned.append({
            "bitstring": mask,
            "energy": safe_float(row.get("energy", 0.0)),
            "predicted_compression": safe_float(row.get("compression", 0.0)),
            "predicted_loss_penalty": safe_float(row.get("loss_penalty", 0.0)),
            "pairwise_penalty": safe_float(row.get("pairwise_penalty", 0.0)),
            "num_pruned_blocks": safe_int(row.get("num_pruned_blocks", len(pruned_blocks))),
            "pruned_blocks": pruned_blocks,
            "pruned_blocks_text": "; ".join(pruned_blocks),
        })

    cleaned = sorted(cleaned, key=lambda r: r["energy"])

    return cleaned[:top_k]


# ============================================================
# Recommendation logic
# ============================================================

def add_recommendations(
    rows: List[Dict[str, Any]],
    target_compression: float,
    compression_penalty_weight: float,
) -> Dict[str, Any]:
    """
    Adds final decision labels.

    We keep two recommendations:
        1. best actual F1
        2. best trade-off score

    Trade-off score:
        actual_macro_f1 - compression_penalty_weight * |predicted_compression - target|

    This avoids pretending that QUBO rank alone is final proof.
    """
    if not rows:
        return {}

    for row in rows:
        predicted_compression = safe_float(row["predicted_compression"])
        actual_macro_f1 = safe_float(row["actual_macro_f1"])

        row["tradeoff_score"] = (
            actual_macro_f1
            - compression_penalty_weight * abs(predicted_compression - target_compression)
        )
        row["final_decision"] = ""

    best_f1_row = max(rows, key=lambda r: safe_float(r["actual_macro_f1"]))
    best_tradeoff_row = max(rows, key=lambda r: safe_float(r["tradeoff_score"]))

    best_f1_row["final_decision"] = "best_actual_f1"

    if best_tradeoff_row is best_f1_row:
        best_tradeoff_row["final_decision"] = "best_actual_f1_and_best_tradeoff"
    else:
        best_tradeoff_row["final_decision"] = "best_tradeoff"

    return {
        "best_actual_f1": {
            "qubo_rank": best_f1_row["qubo_rank"],
            "mask": best_f1_row["mask"],
            "pruned_blocks": best_f1_row["pruned_blocks"],
            "actual_accuracy": best_f1_row["actual_accuracy"],
            "actual_macro_f1": best_f1_row["actual_macro_f1"],
            "actual_validation_loss": best_f1_row["actual_validation_loss"],
            "accuracy_drop": best_f1_row["accuracy_drop"],
            "f1_drop": best_f1_row["f1_drop"],
        },
        "best_tradeoff": {
            "qubo_rank": best_tradeoff_row["qubo_rank"],
            "mask": best_tradeoff_row["mask"],
            "pruned_blocks": best_tradeoff_row["pruned_blocks"],
            "predicted_compression": best_tradeoff_row["predicted_compression"],
            "actual_accuracy": best_tradeoff_row["actual_accuracy"],
            "actual_macro_f1": best_tradeoff_row["actual_macro_f1"],
            "tradeoff_score": best_tradeoff_row["tradeoff_score"],
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["convnext", "swinv2"], default="convnext")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--target-compression", type=float, default=0.30)
    parser.add_argument("--compression-penalty-weight", type=float, default=0.25)

    parser.add_argument(
        "--energy-csv",
        default="qubo_outputs/qubo_energy_check.csv",
        help="Input file with ranked masks.",
    )

    parser.add_argument(
        "--selected-candidates-csv",
        default="qubo_outputs/selected_candidates.csv",
        help="Input file mapping QUBO candidates to parameter counts.",
    )

    parser.add_argument(
        "--output-csv",
        default="qubo_outputs/top_5_mask_evaluation.csv",
    )

    parser.add_argument(
        "--recommendation-json",
        default="qubo_outputs/top_5_mask_recommendation.json",
    )

    parser.add_argument(
        "--exclude-empty-mask",
        action="store_true",
        help="Exclude the all-zero keep-everything mask if it appears.",
    )

    args, unknown = parser.parse_known_args()

    if unknown:
        print("Ignored unknown arguments:", unknown)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    energy_path = Path(args.energy_csv)
    selected_candidates_path = Path(args.selected_candidates_csv)
    output_csv_path = Path(args.output_csv)
    recommendation_json_path = Path(args.recommendation_json)

    print("\nLoading top masks...")
    top_masks = load_top_k_masks(
        energy_path=energy_path,
        top_k=args.top_k,
        exclude_empty_mask=args.exclude_empty_mask,
    )

    print(f"Loaded {len(top_masks)} masks from: {energy_path}")

    for i, mask_row in enumerate(top_masks, start=1):
        print(
            f"Rank {i}: mask={mask_row['bitstring']}, "
            f"energy={mask_row['energy']:.6f}, "
            f"blocks={mask_row['pruned_blocks_text']}"
        )

    print("\nLoading selected candidate parameter map...")
    params_by_candidate = load_selected_candidate_params(selected_candidates_path)

    print("\nLoading validation/test subset...")
    loader = build_dataloader(
        model_name=args.model,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("\nEvaluating fresh baseline model...")
    baseline_model = load_finetuned_model(args.model, device)
    total_trainable_params = count_trainable_params(baseline_model)

    baseline_metrics = evaluate(baseline_model, loader, device)

    print(
        f"Baseline: "
        f"loss={baseline_metrics['loss']:.6f}, "
        f"acc={baseline_metrics['accuracy']:.4f}, "
        f"f1={baseline_metrics['macro_f1']:.4f}, "
        f"params={total_trainable_params:,}"
    )

    del baseline_model
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result_rows: List[Dict[str, Any]] = []

    print("\nEvaluating top masks separately...")

    for qubo_rank, mask_row in enumerate(top_masks, start=1):
        mask = mask_row["bitstring"]
        pruned_blocks = mask_row["pruned_blocks"]

        print("\n" + "=" * 80)
        print(f"Evaluating QUBO rank {qubo_rank}")
        print(f"Mask: {mask}")
        print(f"Pruned blocks: {mask_row['pruned_blocks_text']}")

        model = load_finetuned_model(args.model, device)

        apply_pruning_mask(model, pruned_blocks)

        metrics = evaluate(model, loader, device)

        pruned_params = sum(
            params_by_candidate.get(block_name, 0)
            for block_name in pruned_blocks
        )

        actual_parameter_reduction = (
            pruned_params / total_trainable_params
            if total_trainable_params > 0
            else 0.0
        )

        accuracy_drop = baseline_metrics["accuracy"] - metrics["accuracy"]
        f1_drop = baseline_metrics["macro_f1"] - metrics["macro_f1"]
        loss_increase = metrics["loss"] - baseline_metrics["loss"]

        result_row = {
            "qubo_rank": qubo_rank,
            "mask": mask,
            "pruned_blocks": mask_row["pruned_blocks_text"],
            "qubo_energy": mask_row["energy"],
            "predicted_compression": mask_row["predicted_compression"],
            "predicted_loss_penalty": mask_row["predicted_loss_penalty"],
            "pairwise_penalty": mask_row["pairwise_penalty"],
            "num_pruned_blocks": len(pruned_blocks),

            "baseline_accuracy": baseline_metrics["accuracy"],
            "baseline_macro_f1": baseline_metrics["macro_f1"],
            "baseline_validation_loss": baseline_metrics["loss"],

            "actual_accuracy": metrics["accuracy"],
            "actual_macro_f1": metrics["macro_f1"],
            "actual_validation_loss": metrics["loss"],

            "accuracy_drop": accuracy_drop,
            "f1_drop": f1_drop,
            "loss_increase": loss_increase,

            "pruned_parameter_count": pruned_params,
            "total_trainable_params": total_trainable_params,
            "actual_parameter_reduction": actual_parameter_reduction,

            "n_samples": metrics["n_samples"],
            "tradeoff_score": "",
            "final_decision": "",
        }

        result_rows.append(result_row)

        print(
            f"Result: "
            f"loss={metrics['loss']:.6f}, "
            f"acc={metrics['accuracy']:.4f}, "
            f"f1={metrics['macro_f1']:.4f}"
        )
        print(
            f"Drops: "
            f"Δloss={loss_increase:.6f}, "
            f"Δacc={accuracy_drop:.4f}, "
            f"Δf1={f1_drop:.4f}"
        )
        print(
            f"Parameter reduction estimate: "
            f"{pruned_params:,} / {total_trainable_params:,} "
            f"= {actual_parameter_reduction:.4%}"
        )

        del model
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    recommendation = add_recommendations(
        rows=result_rows,
        target_compression=args.target_compression,
        compression_penalty_weight=args.compression_penalty_weight,
    )

    fieldnames = [
        "qubo_rank",
        "mask",
        "pruned_blocks",
        "qubo_energy",
        "predicted_compression",
        "predicted_loss_penalty",
        "pairwise_penalty",
        "num_pruned_blocks",

        "baseline_accuracy",
        "baseline_macro_f1",
        "baseline_validation_loss",

        "actual_accuracy",
        "actual_macro_f1",
        "actual_validation_loss",

        "accuracy_drop",
        "f1_drop",
        "loss_increase",

        "pruned_parameter_count",
        "total_trainable_params",
        "actual_parameter_reduction",

        "n_samples",
        "tradeoff_score",
        "final_decision",
    ]

    write_csv(output_csv_path, result_rows, fieldnames)

    summary = {
        "top_k": args.top_k,
        "model": args.model,
        "split": args.split,
        "max_samples": args.max_samples,
        "target_compression": args.target_compression,
        "baseline": baseline_metrics,
        "total_trainable_params": total_trainable_params,
        "recommendation": recommendation,
        "note": (
            "Masks were evaluated separately from a fresh baseline model. "
            "Pruning is simulated by bypassing selected structured blocks."
        ),
    }

    write_json(recommendation_json_path, summary)

    print("\n" + "=" * 80)
    print(f"Saved evaluation table to: {output_csv_path}")
    print(f"Saved recommendation summary to: {recommendation_json_path}")

    print("\nFinal recommendation:")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
