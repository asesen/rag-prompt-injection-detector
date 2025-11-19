# detector.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from .vectorstore import VectorStore


# Предполагается, что parquet-файлы содержат столбцы:
# 'embedding' (iterable/ndarray длины 1024) и 'jailbreak' (0/1)


# -----------------------
# Утилиты: метрики
# -----------------------
def compute_metrics(y_true, y_pred, prefix=""):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    print(f"{prefix}acc={acc:.4f}, prec={prec:.4f}, rec={rec:.4f}, f1={f1:.4f}")
    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

# -----------------------
# Dataset wrapper
# -----------------------
class EmbeddingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -----------------------
# Primary model 1024 -> 8 -> 2
# -----------------------
class PrimaryNet(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=8, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        # инициализация
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.fc1(x)
        h = self.act(h)
        logits = self.fc2(h)
        return logits, h

# -----------------------
# Detector (device = cpu)
# -----------------------
class Detector:
    def __init__(self, vector_store: VectorStore, device: str = "cpu"):
        self.vector_store = vector_store
        self.device = device
        self.primary = None
        self.secondary = None
        self.eps = 1e-6
        self.exact_match_threshold = 1e-4  # если distance < threshold -> exact match

    # -------------------
    # Тренировка primary
    # -------------------
    def train_primary(self,
                      epochs: int,
                      batch_size: int,
                      lr: float,
                      weight_decay: float,
                      save_path="model/primary_cpu.pt"):
        train_df = self.vector_store.give_train_data()
        val_df = self.vector_store.give_val_data()

        X_train = np.vstack(train_df['embedding'].values).astype(np.float32)
        y_train = train_df['jailbreak'].astype(int).values.astype(np.int64)
        X_val = np.vstack(val_df['embedding'].values).astype(np.float32)
        y_val = val_df['jailbreak'].astype(int).values.astype(np.int64)

        self.primary = PrimaryNet(input_dim=X_train.shape[1], hidden_dim=8, num_classes=2).to(self.device)

        train_loader = DataLoader(EmbeddingDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(EmbeddingDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

        # weighted loss: [weight_for_0, weight_for_1] => ratio 1:10 (1 for class 0, 10 for class 1)
        class_weights = torch.tensor([1.0, 10.0], dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = AdamW(self.primary.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, epochs + 1):
            self.primary.train()
            all_preds = []
            all_labels = []
            for xb, yb in train_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                optimizer.zero_grad()
                logits, _ = self.primary(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(yb.cpu().numpy())

            train_pred = np.concatenate(all_preds)
            train_true = np.concatenate(all_labels)

            # validation
            self.primary.eval()
            v_preds = []
            v_labels = []


            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(self.device)
                    yb = yb.to(self.device)
                    logits, _ = self.primary(xb)
                    preds = logits.argmax(dim=1).cpu().numpy()
                    v_preds.append(preds)
                    v_labels.append(yb.cpu().numpy())

            val_pred = np.concatenate(v_preds)
            val_true = np.concatenate(v_labels)

            print(f"Primary epoch {epoch}/{epochs}")
            compute_metrics(train_true, train_pred, prefix="Train: ")
            compute_metrics(val_true, val_pred, prefix="Val:   ")
            print("-" * 60)

        # save
        torch.save({
            'state_dict': self.primary.state_dict(),
            'input_dim': X_train.shape[1],
            'hidden_dim': 8
        }, save_path)
        print(f"Primary saved to {save_path}")

    def load_primary(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.primary = PrimaryNet(input_dim=ckpt['input_dim'], hidden_dim=ckpt['hidden_dim'], num_classes=2)
        self.primary.load_state_dict(ckpt['state_dict'])
        self.primary.to(self.device)
        self.primary.eval()
        print(f"Primary loaded from {path}")

    def predict_primary_intermediate(self, emb: np.ndarray):
        """
        Возвращает: logits (2,), intermediate (8,)
        emb: 1D np.ndarray
        """
        assert self.primary is not None, "Primary model not loaded"
        x = torch.tensor(emb.reshape(1, -1).astype(np.float32)).to(self.device)
        with torch.no_grad():
            logits, inter = self.primary(x)
        return logits.cpu().numpy()[0], inter.cpu().numpy()[0]

    def detect(self, emb: list,context_dist = None, context_idx = None):
        """
        emb: список или 1D np.ndarray с embedding
        context: пока заглушка
        Возвращает: вероятность класса 'jailbreak' (float 0..1)
        """
        emb = np.array(emb, dtype=np.float32).reshape(-1)  # гарантируем 1D
        logits, _ = self.predict_primary_intermediate(emb)  # logits shape (2,)

        # преобразуем в вероятности через softmax
        probs = F.softmax(torch.tensor(logits), dim=0).numpy()  # shape (2,)
        jailbreak_prob = probs[1]

        return jailbreak_prob


# -----------------------
# Пример использования (псевдокод, запускать в среде с parquet и faiss или без faiss)
# -----------------------
    # Пример (не запускается автоматически, используй в своей среде)
    # vector_store = VectorStore(...)
    # detector = Detector(vector_store=vector_store, device="cpu")
    #
    # detector.train_primary(epochs=10, batch_size=128, lr=1e-3, weight_decay=1e-4, save_path="primary_cpu.pt")
    # detector.load_primary("primary_cpu.pt")
    # detector.train_secondary(epochs=8, batch_size=128, lr=5e-4, weight_decay=1e-4, small_train_fraction=0.05, neighbor_k=5, save_path="secondary_cpu.pt")
    # detector.load_secondary("secondary_cpu.pt")
    #
    # # пример инференса на первых 10 тестовых примерах:
    # test_df = vector_store.give_test_data()
    # embs = np.vstack(test_df['embedding'].values)[:10]
