import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from model import SimpleClassifier

# -------------------- Настройки --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(BASE_DIR, "best_model.pt")
VEC_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")
SVD_PATH = os.path.join(BASE_DIR, "svd.pkl")
LE_PATH  = os.path.join(BASE_DIR, "label_encoder.pkl")

NUM_THREADS = 8
torch.set_num_threads(NUM_THREADS)

# -------------------- Загрузка препроцессоров --------------------
def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)

vectorizer = load_pickle(VEC_PATH)
svd = load_pickle(SVD_PATH) if os.path.exists(SVD_PATH) else None
le = load_pickle(LE_PATH)

# Определяем input_dim для модели
input_dim = svd.n_components if svd else getattr(vectorizer, "max_features", None)
print(f"Using input_dim = {input_dim}, svd used: {svd is not None}")

# -------------------- Загрузка модели --------------------
def load_model(checkpoint_path, input_dim, num_classes):
    ckpt_raw = torch.load(checkpoint_path, map_location="cpu")
    sd = ckpt_raw.get("state_dict", ckpt_raw) if isinstance(ckpt_raw, dict) else ckpt_raw

    # detect LayerNorm usage (иногда важно для некоторых моделей)
    used_ln = any(k.startswith("ln") or ".ln1" in k or "ln1." in k for k in sd.keys())
    print("Checkpoint keys sample:", list(sd.keys())[:10], "Used LayerNorm:", used_ln)

    # Определяем количество классов
    possible_out_keys = [k for k in sd.keys() if k.endswith("linear_out.weight") or k.endswith("linear_out.weight".replace("linear_out","linear3"))]
    num_classes_ckpt = sd[possible_out_keys[0]].shape[0] if possible_out_keys else len(le.classes_)
    print("num_classes =", num_classes_ckpt)

    model = SimpleClassifier(input_dim, num_classes_ckpt, p_dropout=0.3)

    # Попытка загрузить state_dict, обработка префиксов
    try:
        model.load_state_dict(sd)
        print("Loaded full state_dict successfully.")
    except Exception:
        stripped = {k.split('.', 1)[-1]: v for k, v in sd.items()}
        model_state = model.state_dict()
        to_load = {k: v for k, v in stripped.items() if k in model_state and v.shape == model_state[k].shape}
        model.load_state_dict(to_load, strict=False)
        print(f"Loaded {len(to_load)} params by shape match (strict=False).")

    return model

model = load_model(CKPT_PATH, input_dim, len(le.classes_))

# -------------------- Ускорения (CPU) --------------------
print("Applying dynamic quantization...")
model_q = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Попытка JIT-трейса
scripted = None
try:
    example = torch.randn(1, input_dim)
    scripted = torch.jit.trace(model_q, example)
    print("JIT trace succeeded, using scripted model.")
except Exception as e:
    print("JIT trace failed or not applicable, using quantized model only.", e)

runtime_model = scripted if scripted else model_q
runtime_model.eval()

# -------------------- Предобработка и предсказание --------------------
def preprocess_text(text: str) -> np.ndarray:
    X_vec = vectorizer.transform([text])
    if svd:
        X_vec = svd.transform(X_vec)
    return X_vec.astype(np.float32)

def predict(text: str) -> str:
    X = preprocess_text(text)
    xb = torch.from_numpy(X)
    with torch.inference_mode():
        logits = runtime_model(xb)
        pred = torch.argmax(logits, dim=1).cpu().numpy()[0]
    return le.inverse_transform([pred])[0]

# -------------------- Пример использования --------------------
if __name__ == "__main__":
    prompt = input("Введите запрос: ")
    result = predict(prompt)

    print("Запрос:", prompt)
    if result == 0:
        print("Предсказание: Поиск в базе.")
    elif result == 1:
        print("Предсказание: Поиск в сети.")
    else:
        print("Предсказание: Прямой запрос.")
