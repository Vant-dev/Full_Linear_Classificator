import os
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from model import SimpleClassifier

# -------------------- Настройки --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "test3_final2.xlsx")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 19
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
LABEL_SMOOTHING = 0.05

# -------------------- Функции --------------------
def load_data(path):
    """Чтение Excel и подготовка текстов и меток"""
    df = pd.read_excel(path)
    texts = df['text'].astype(str).fillna('').tolist()
    raw_labels = df['label'].tolist()
    return texts, raw_labels

def preprocess(texts, labels, max_features=3200, ngram_range=(1,3), min_df=2, svd_components=200):
    """TF-IDF + SVD + LabelEncoder"""
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    X_text = vectorizer.fit_transform(texts)

    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_reduced = svd.fit_transform(X_text)

    le = LabelEncoder()
    y = le.fit_transform(labels)
    return X_reduced, y, vectorizer, svd, le

def split_data(X, y, test_size=0.2, val_size=0.5, random_state=42):
    """Разделение на train/val/test"""
    X_train_full, X_temp, y_train_full, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=random_state, stratify=y_temp
    )
    return X_train_full, X_val, X_test, y_train_full, y_val, y_test

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=BATCH_SIZE):
    """Создание TensorDataset и DataLoader"""
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(DEVICE)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor),
                              batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor),
                            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor),
                             batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler, epochs=EPOCHS, patience=PATIENCE):
    """Цикл обучения с ранней остановкой"""
    best_val_loss = float('inf')
    stalled = 0

    for epoch in range(1, epochs+1):
        # ----- Train -----
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            out = model(X_batch)
            loss = loss_fn(out, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * X_batch.size(0)
            correct_train += (out.argmax(1) == y_batch).sum().item()
            total_train += y_batch.size(0)
        train_loss /= total_train
        train_acc = correct_train / total_train

        # ----- Validation -----
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                out = model(X_batch)
                loss = loss_fn(out, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                correct_val += (out.argmax(1) == y_batch).sum().item()
                total_val += y_batch.size(0)
        val_loss /= total_val
        val_acc = correct_val / total_val

        scheduler.step(val_loss)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

        # ----- Early stopping -----
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            stalled = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            stalled += 1
            if stalled >= patience:
                print("Early stopping triggered")
                break

def evaluate_model(model, test_loader, loss_fn, le):
    """Тестирование, отчет и матрица ошибок"""
    model.load_state_dict(torch.load("best_model.pt"))
    model.eval()

    y_true, y_pred = [], []
    test_loss, correct_test, total_test = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            out = model(X_batch)
            loss = loss_fn(out, y_batch)
            test_loss += loss.item() * X_batch.size(0)
            preds = out.argmax(1)
            correct_test += (preds == y_batch).sum().item()
            total_test += y_batch.size(0)

            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_loss /= total_test
    test_acc = correct_test / total_test
    print(f"\nFinal Test | Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")

    labels_present = sorted(set(y_true) | set(y_pred))
    target_names = [str(le.inverse_transform([lab])[0]) for lab in labels_present]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, labels=labels_present, target_names=target_names))

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred, labels=labels_present))

# -------------------- Основной скрипт --------------------
if __name__ == "__main__":
    texts, raw_labels = load_data(DATA_PATH)
    X_reduced, y, vectorizer, svd, le = preprocess(texts, raw_labels)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_reduced, y)
    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)

    input_dim = X_train.shape[1]
    num_classes = len(le.classes_)
    model = SimpleClassifier(input_dim=input_dim, num_classes=num_classes, p_dropout=DROPOUT).to(DEVICE)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    train_model(model, train_loader, val_loader, loss_fn, optimizer, scheduler)

    evaluate_model(model, test_loader, loss_fn, le)

    # -------------------- Сохраняем препроцессоры --------------------
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)

    print("\nArtifacts saved: vectorizer.pkl, svd.pkl, label_encoder.pkl")
