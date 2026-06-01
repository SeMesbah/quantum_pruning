"""Standalone script that mirrors quantum_pruning.ipynb cells 2-4."""

# ── Cell 2: Load model ────────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download
import torch, timm
from PIL import Image
from torchvision import transforms

vit_path = hf_hub_download(
    repo_id="canada-guesser/canadian_streetview_cities_models",
    filename="vit_model/swinv2_base_window12_192_0_finetuned_canadian_streetview.bin"
)

model = timm.create_model("swinv2_base_window12_192", pretrained=False, num_classes=15)
model.load_state_dict(torch.load(vit_path, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((192, 192)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

class_names = [
    "Calgary", "Charlottetown", "Edmonton", "Halifax", "Hamilton",
    "Kitchener-Waterloo", "Montreal", "Ottawa-Gatineau", "Quebec City", "Saskatoon",
    "St Johns", "Toronto", "Vancouver", "Victoria", "Winnipeg",
]

img = Image.open(r"C:\_projects\qiskit-examples\image.png").convert("RGB")
x = transform(img).unsqueeze(0)

with torch.no_grad():
    pred = model(x)
print("Baseline city:", class_names[pred.argmax().item()])

# ── Cell 3: QuantumPruner class ───────────────────────────────────────────────
import numpy as np
import torch.nn as nn
import copy
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


class QuantumPruner:
    def __init__(self, n_qubits=6, shots=512, max_circuits_per_layer=32, seed=42):
        self.n_qubits = n_qubits
        self.shots = shots
        self.max_circuits = max_circuits_per_layer
        self.rng = np.random.default_rng(seed)
        self.sim = AerSimulator()

    def build_circuit(self, chunk_norm):
        n = len(chunk_norm)
        qc = QuantumCircuit(n)
        for i, w in enumerate(chunk_norm):
            theta = 2.0 * np.arcsin(np.sqrt(float(np.clip(w, 1e-9, 1.0 - 1e-9))))
            qc.ry(theta, i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        if n >= 2:
            qc.h(range(n))
            qc.x(range(n))
            qc.h(n - 1)
            qc.mcx(list(range(n - 1)), n - 1)
            qc.h(n - 1)
            qc.x(range(n))
            qc.h(range(n))
        qc.measure_all()
        return qc

    def _run_circuit(self, chunk_norm):
        n = len(chunk_norm)
        qc = self.build_circuit(chunk_norm)
        compiled = transpile(qc, self.sim, optimization_level=1)
        counts = self.sim.run(compiled, shots=self.shots).result().get_counts()
        probs = np.zeros(n)
        for bitstring, cnt in counts.items():
            for i, bit in enumerate(reversed(bitstring)):
                if i < n:
                    probs[i] += int(bit) * cnt
        return probs / self.shots

    def quantum_scores(self, flat_abs):
        n = len(flat_abs)
        w_max = flat_abs.max()
        if w_max == 0.0:
            return np.zeros(n)
        w_norm = flat_abs / w_max
        step = self.n_qubits
        n_chunks = (n + step - 1) // step
        if n_chunks <= self.max_circuits:
            scores = np.empty(n)
            for s in range(0, n, step):
                e = min(s + step, n)
                scores[s:e] = self._run_circuit(w_norm[s:e])
        else:
            sorted_idx = np.argsort(w_norm)
            bucket = max(1, n // self.max_circuits)
            sample_means, sample_ranks = [], []
            for b in range(self.max_circuits):
                s = b * bucket
                e = min(s + step, n)
                if e > n:
                    break
                q = self._run_circuit(w_norm[sorted_idx[s:e]])
                sample_means.append(np.mean(q))
                sample_ranks.append(s + (e - s) / 2.0)
            all_ranks = np.arange(n, dtype=float)
            q_by_rank = np.interp(all_ranks, sample_ranks, sample_means)
            scores = np.empty(n)
            scores[sorted_idx] = q_by_rank
        return scores

    def prune_tensor(self, weight, sparsity, quantum_bias=0.25):
        flat = weight.detach().cpu().numpy().ravel()
        abs_w = np.abs(flat)
        q = self.quantum_scores(abs_w)
        importance = abs_w * (1.0 + quantum_bias * q)
        threshold = np.percentile(importance, sparsity * 100.0)
        mask = (importance >= threshold).reshape(weight.shape)
        pruned_w = torch.tensor(weight.detach().cpu().numpy() * mask, dtype=weight.dtype)
        return pruned_w, mask

    def apply(self, model, sparsity_config, quantum_bias=0.25, verbose=True):
        model = copy.deepcopy(model)
        total = pruned_count = 0
        layer_stats = []

        def resolve(name):
            for k, v in sparsity_config.items():
                if k != 'default' and k in name:
                    return v
            return sparsity_config.get('default', 0.30)

        for name, mod in model.named_modules():
            if not isinstance(mod, nn.Linear):
                continue
            sp = resolve(name)
            new_w, mask = self.prune_tensor(mod.weight, sp, quantum_bias)
            mod.weight = nn.Parameter(new_w)
            n_pruned = int((~mask).sum())
            n_total  = int(mask.size)
            pruned_count += n_pruned
            total        += n_total
            layer_stats.append(dict(name=name, shape=list(mod.weight.shape),
                                    target=sp, actual=n_pruned / n_total))
            if verbose:
                print(f"  [{sp:.0%}->{n_pruned/n_total:.1%}]  {name}  {list(mod.weight.shape)}")

        overall = pruned_count / total if total else 0.0
        if verbose:
            print(f"\n  Overall sparsity: {overall:.1%}  ({pruned_count:,} / {total:,} weights zeroed)")
        return model, dict(total=total, pruned=pruned_count,
                           overall_sparsity=overall, layers=layer_stats)


# ── Cell 4: Apply pruning & compare ──────────────────────────────────────────
import time
import torch.nn.functional as F

SPARSITY_CONFIG = {
    'attn':    0.15,
    'mlp':     0.40,
    'head':    0.10,
    'default': 0.30,
}

pruner = QuantumPruner(n_qubits=6, shots=512, max_circuits_per_layer=32)

print("\nApplying quantum-assisted pruning …\n")
t0 = time.time()
model_pruned, stats = pruner.apply(model, SPARSITY_CONFIG, quantum_bias=0.25)
print(f"\nCompleted in {time.time() - t0:.1f}s")

model_pruned.eval()
with torch.no_grad():
    logits_orig   = model(x)
    logits_pruned = model_pruned(x)

probs_orig   = F.softmax(logits_orig,   dim=1)[0].numpy()
probs_pruned = F.softmax(logits_pruned, dim=1)[0].numpy()
top3_orig    = np.argsort(probs_orig)[::-1][:3]
top3_pruned  = np.argsort(probs_pruned)[::-1][:3]

print("\n┌────────────────────────────────────────────────────────────┐")
print("│                   Inference Comparison                     │")
print("├──────────────────────────────┬─────────────────────────────┤")
print("│  Original                    │  Pruned                     │")
print("├──────────────────────────────┼─────────────────────────────┤")
for rank, (io, ip) in enumerate(zip(top3_orig, top3_pruned), 1):
    print(f"│  {rank}. {class_names[io]:<24s} {probs_orig[io]:5.1%}  │"
          f"  {rank}. {class_names[ip]:<24s} {probs_pruned[ip]:5.1%}  │")
print("└──────────────────────────────┴─────────────────────────────┘")

total_all  = sum(p.numel() for p in model.parameters())
pruned_all = sum((p == 0).sum().item() for p in model_pruned.parameters())
print(f"\n  Total parameters  : {total_all:,}")
print(f"  Zeroed parameters : {pruned_all:,}  ({pruned_all/total_all:.1%} of all params)")
print(f"  Linear sparsity   : {stats['overall_sparsity']:.1%}")

# ── Quantum circuit introspection ─────────────────────────────────────────────
example_weights = np.array([0.85, 0.12, 0.67, 0.34, 0.91, 0.05])
w_norm = example_weights / example_weights.max()
qc_demo = pruner.build_circuit(w_norm)
print("\nQuantum importance circuit for a 6-weight chunk")
print("=" * 65)
print(qc_demo.draw(output='text', fold=120))
print(f"  Depth : {qc_demo.depth()}   Gates : {dict(qc_demo.count_ops())}")
measured = pruner._run_circuit(w_norm)
print(f"  |w| normalised        : {np.round(w_norm, 3)}")
print(f"  P(|1⟩) — full circuit : {np.round(measured, 3)}")
