import os
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


DATA_DIR = "data"       
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]
WINDOW = 20                
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 50
VAL_RATIO = 0.15         
PATIENCE = 7               

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(path)
    if "Price" in df.columns and df.iloc[0, 0] == "Ticker" and df.iloc[1, 0] == "Date":
        df = df.iloc[2:].reset_index(drop=True)

        df = df.rename(columns={"Price": "Date"})
        df["Date"] = pd.to_datetime(df["Date"])

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

        df["Adj Close"] = df["Close"]

    else:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        keep_cols = ["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        missing = [c for c in keep_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing expected columns {missing} in {path}. "
                f"Available columns: {list(df.columns)}"
            )
        df = df[keep_cols]
    df = df.sort_values("Date").reset_index(drop=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_10"] = df["Close"].rolling(window=10, min_periods=10).mean()
    df["SMA_20"] = df["Close"].rolling(window=20, min_periods=20).mean()
    df["Ret_1"] = df["Close"].pct_change(1)
    df["Vol_10"] = df["Ret_1"].rolling(window=10, min_periods=10).std()

    df = df.dropna().reset_index(drop=True)
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.iloc[:-1].reset_index(drop=True)
    return df


def build_windows(
    df: pd.DataFrame, feature_cols: List[str], window: int
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    values = df[feature_cols].values
    targets = df["Target"].values

    for t in range(window, len(df)):
        X_t = values[t - window:t]
        y_t = targets[t]
        X.append(X_t)
        y.append(y_t)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)



class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, model_type: str = "lstm"):
        if model_type == "mlp":
            num_samples, T, F = X.shape
            X = X.reshape(num_samples, T * F)

        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.model_type = model_type

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        logits = self.net(x).squeeze(-1)
        return logits


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)  
        h_last = h_n[-1]           
        logits = self.fc(h_last).squeeze(-1)
        return logits



def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    patience: int = 7,
) -> nn.Module:
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for Xb, yb in train_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.float().to(DEVICE)

            optimizer.zero_grad()
            logits = model(Xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * Xb.size(0)

        train_loss = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb = Xb.to(DEVICE)
                yb = yb.float().to(DEVICE)
                logits = model(Xb)
                loss = criterion(logits, yb)
                val_loss += loss.item() * Xb.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader):
    model.eval()
    preds = []
    probs = []
    labels = []

    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(Xb)
            p = torch.sigmoid(logits)

            probs.extend(p.cpu().numpy().tolist())
            preds.extend((p >= 0.5).long().cpu().numpy().tolist())
            labels.extend(yb.cpu().numpy().tolist())

    acc = accuracy_score(labels, preds)
    bacc = balanced_accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")  

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"ROC-AUC: {auc:.4f}")
    return acc, bacc, auc



def run_experiment(model_type: str = "lstm"):
    all_X = []
    all_y = []

    for ticker in TICKERS:
        print(f"Processing {ticker}...")
        df = load_data(ticker)
        df = add_indicators(df)   
        df = create_target(df)  

        feature_cols = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "SMA_10",
            "SMA_20",
            "Ret_1",
            "Vol_10",
        ]

        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])


        X, y = build_windows(df, feature_cols, WINDOW)
        all_X.append(X)
        all_y.append(y)

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=VAL_RATIO * 2, shuffle=False
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, shuffle=False
    )

    train_ds = WindowDataset(X_train, y_train, model_type=model_type)
    val_ds = WindowDataset(X_val, y_val, model_type=model_type)
    test_ds = WindowDataset(X_test, y_test, model_type=model_type)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    if model_type == "mlp":
        input_dim = train_ds.X.shape[1]
        model = MLP(input_dim=input_dim)
    else:
        _, T, F = X_train.shape
        model = LSTMClassifier(input_size=F)

    model = train_model(model, train_loader, val_loader,
                        epochs=EPOCHS, lr=LR, patience=PATIENCE)
    acc, bacc, auc = evaluate_model(model, test_loader)
    return model, (acc, bacc, auc)

def predict_next_day_for_ticker(model: nn.Module, ticker: str, model_type: str = "lstm") -> float:
    df = load_data(ticker)
    df = add_indicators(df)
    df = create_target(df)

    feature_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "SMA_10",
        "SMA_20",
        "Ret_1",
        "Vol_10",
    ]

    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    last_window = df[feature_cols].values[-WINDOW:]  

    x = torch.tensor(last_window, dtype=torch.float32)

    if model_type == "lstm":
        x = x.unsqueeze(0)
    else:
        x = x.view(1, -1)

    x = x.to(DEVICE)

    model.eval()
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()

    return prob



if __name__ == "__main__":
    model, metrics = run_experiment(model_type="lstm")

    print("\nNext-day movement predictions (P(up)):")
    for t in TICKERS:
        prob_up = predict_next_day_for_ticker(model, t, model_type="lstm")
        direction = "UP" if prob_up >= 0.5 else "DOWN"
        print(f"{t}: P(up) = {prob_up:.3f} -> {direction}")