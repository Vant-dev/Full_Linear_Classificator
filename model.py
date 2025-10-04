# prepare_hf_model_full.py
import os
import json
import pickle
import traceback
import torch

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "best_model.pt")            # твой чекпоинт
VEC_PATH  = os.path.join(BASE_DIR, "vectorizer.pkl")          # опционально
SVD_PATH  = os.path.join(BASE_DIR, "svd.pkl")                 # опционально
LE_PATH   = os.path.join(BASE_DIR, "label_encoder.pkl")       # опционально

OUTPUT_DIR = os.path.join(BASE_DIR, "my_model_hf")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- model.py content (запишем, если не существует) ----------------
MODEL_PY = os.path.join(OUTPUT_DIR, "model.py")
model_py_text = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, p_dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(p_dropout)
        self.linear2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.linear_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.gelu(self.ln1(self.linear1(x)))
        x = self.dropout(x)
        x = F.gelu(self.ln2(self.linear2(x)))
        x = self.dropout(x)
        return self.linear_out(x)
'''
# запишем model.py если он отсутствует (или перезапишем по желанию)
if not os.path.exists(MODEL_PY):
    with open(MODEL_PY, "w", encoding="utf-8") as f:
        f.write(model_py_text)
    print("Wrote model.py ->", MODEL_PY)
else:
    print("model.py already exists in output dir; not overwritten:", MODEL_PY)

# ---------------- helper: safe load pickle ----------------
def try_load_pickle(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Failed to load pickle {path}: {e}")
        return None

vectorizer = try_load_pickle(VEC_PATH)
svd = try_load_pickle(SVD_PATH)
le = try_load_pickle(LE_PATH)

# ---------------- load checkpoint robustly ----------------
if not os.path.exists(CKPT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
ckpt_raw = torch.load(CKPT_PATH, map_location="cpu")

# если чекпоинт — словарь с ключом state_dict
if isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw:
    sd = ckpt_raw["state_dict"]
else:
    sd = ckpt_raw if isinstance(ckpt_raw, dict) else ckpt_raw

# если sd is a model state_dict tensor mapping, ok; else try to handle
if not isinstance(sd, dict):
    raise ValueError("Can't interpret checkpoint format: expected dict-like state_dict.")

# strip common prefixes like 'module.' etc.
def strip_prefix(state_dict):
    # detect a common prefix
    keys = list(state_dict.keys())
    if not keys:
        return state_dict
    sample = keys[0]
    # common prefixes: 'module.' or 'model.' etc - we'll remove if present consistently
    prefixes = []
    for p in ("module.", "model.", "model_state_dict."):
        if any(k.startswith(p) for k in keys):
            prefixes.append(p)
    if not prefixes:
        return state_dict
    # remove the first matching prefix
    prefix = prefixes[0]
    new = {k[len(prefix):] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    return new

sd = strip_prefix(sd)

# ---------------- infer input_dim and num_classes ----------------
input_dim = None
num_classes = None

# 1) try from svd / vectorizer
try:
    if svd is not None and hasattr(svd, "n_components"):
        input_dim = int(svd.n_components)
    elif vectorizer is not None:
        # try common attributes
        for attr in ("max_features", "vocabulary_", "n_features_in_"):
            if hasattr(vectorizer, attr):
                val = getattr(vectorizer, attr)
                # if vocabulary_ -> use its length
                if attr == "vocabulary_" and isinstance(val, dict):
                    input_dim = int(len(val))
                else:
                    try:
                        input_dim = int(val)
                    except Exception:
                        pass
                break
except Exception:
    input_dim = None

# 2) try from label encoder
try:
    if le is not None and hasattr(le, "classes_"):
        num_classes = int(len(le.classes_))
        classes = [str(c) for c in le.classes_]
    else:
        classes = None
except Exception:
    num_classes = None
    classes = None

# 3) fallback: infer from state_dict shapes
# find first linear weight that likely corresponds to linear1
if input_dim is None:
    candidate = None
    for k, v in sd.items():
        if k.endswith("linear1.weight") or k.endswith("linear1.weight".replace("linear1","linear1")) or ("linear1.weight" in k):
            candidate = v
            break
    if candidate is None:
        # try to find any weight with 2 dims and assume shape = (out, in)
        for k, v in sd.items():
            if hasattr(v, "shape") and len(getattr(v, "shape", ())) == 2:
                candidate = v
                break
    if candidate is not None:
        try:
            input_dim = int(list(candidate.shape)[1])
            print("Inferred input_dim from checkpoint weight shape:", input_dim)
        except Exception:
            input_dim = None

if num_classes is None:
    # try to infer from final layer weight
    candidate = None
    for k, v in sd.items():
        if k.endswith("linear_out.weight") or "linear_out.weight" in k or k.endswith("linear_out.weight".replace("linear_out","linear_out")):
            candidate = v
            break
    if candidate is None:
        # fallback: try last linear-like layer by name
        for k, v in reversed(list(sd.items())):
            if hasattr(v, "shape") and len(getattr(v, "shape", ())) == 2:
                candidate = v
                break
    if candidate is not None:
        try:
            num_classes = int(list(candidate.shape)[0])
            print("Inferred num_classes from checkpoint weight shape:", num_classes)
        except Exception:
            num_classes = None

# final safety defaults
if input_dim is None:
    raise RuntimeError("Failed to determine input_dim. Provide svd/vectorizer or ensure checkpoint contains recognizable linear1.weight.")
if num_classes is None:
    raise RuntimeError("Failed to determine num_classes. Provide label_encoder or ensure checkpoint contains recognizable linear_out.weight.")

print("Determined input_dim:", input_dim, "num_classes:", num_classes)

# ---------------- create model and load weights ----------------
# import the model class from the model.py we wrote into OUTPUT_DIR
import importlib.util
spec = importlib.util.spec_from_file_location("hf_model_module", MODEL_PY)
hf_model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hf_model_module)
SimpleClassifier = getattr(hf_model_module, "SimpleClassifier")

model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes, p_dropout=0.3)

# try direct load_state_dict, else match by names & shapes
try:
    model.load_state_dict(sd)
    print("Direct load_state_dict() successful.")
except Exception as e:
    print("Direct load failed, trying shape-matching load:", e)
    model_state = model.state_dict()
    # make a stripped copy (remove any extra prefixes)
    stripped = {k.split('.', 1)[-1]: v for k, v in sd.items()}
    to_load = {}
    for k_model, v_model in model_state.items():
        if k_model in sd and getattr(sd[k_model], "shape", None) == getattr(v_model, "shape", None):
            to_load[k_model] = sd[k_model]
        elif k_model in stripped and getattr(stripped[k_model], "shape", None) == getattr(v_model, "shape", None):
            to_load[k_model] = stripped[k_model]
        else:
            # try find any sd tensor with same shape
            for k_sd, v_sd in sd.items():
                if getattr(v_sd, "shape", None) == getattr(v_model, "shape", None):
                    to_load[k_model] = v_sd
                    break
    model.load_state_dict(to_load, strict=False)
    print(f"Loaded {len(to_load)} parameters by shape matching (strict=False).")

# ---------------- save weights ----------------
weights_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
torch.save(model.state_dict(), weights_path)
print("Saved weights ->", weights_path)

# ---------------- save config.json (safe sanitize) ----------------
cfg = {
    "model_type": "simple_classifier",
    "input_dim": int(input_dim),
    "num_classes": int(num_classes),
    "p_dropout": float(0.3),
    "classes": [str(c) for c in (classes if classes is not None else [i for i in range(num_classes)])],
    "transformers_version": None
}

# try to import transformers to fill version (optional)
try:
    import transformers
    cfg["transformers_version"] = getattr(transformers, "__version__", None)
except Exception:
    cfg["transformers_version"] = None

def sanitize(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # try convert numpy etc
    try:
        return int(obj)
    except Exception:
        try:
            return float(obj)
        except Exception:
            return str(obj)

cfg_safe = sanitize(cfg)
config_file = os.path.join(OUTPUT_DIR, "config.json")
with open(config_file, "w", encoding="utf-8") as f:
    json.dump(cfg_safe, f, ensure_ascii=False, indent=2)
print("Saved config ->", config_file)

# ---------------- copy also pickles (optional) ----------------
for path in (VEC_PATH, SVD_PATH, LE_PATH):
    if os.path.exists(path):
        try:
            import shutil
            shutil.copy(path, OUTPUT_DIR)
            print("Copied", path, "->", OUTPUT_DIR)
        except Exception as e:
            print("Failed to copy", path, ":", e)

print("Done. OUTPUT_DIR listing:", os.listdir(OUTPUT_DIR))
