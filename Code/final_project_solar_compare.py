
"""
final_project_solar_compare.py
------------------------------------------------------------
Final project helper script for short-term solar power forecasting.

What it does
------------
- Reads all daily gpg_*.csv logs from a folder named "logs"
- Cleans and aligns the data onto a 1-minute grid
- Builds three main model sections:
    1) FTDNN
    2) NARX
    3) LSTM
- For FTDNN and NARX, trains both:
    - linear version
    - nonlinear version
  so you can show whether the nonlinear model really beats a linear predictor
- Compares every model against a persistence baseline
- Saves numeric metrics and comparison plots

Default goal
------------
Use the last LOOKBACK minutes to predict PV_Power_W HORIZON minutes ahead.

Typical use
-----------
python final_project_solar_compare.py
python final_project_solar_compare.py --lookback 120 --horizon 60 --epochs 50
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# Constants
# ============================================================

TARGET = "PV_Power_W"

WEATHER_7 = [
    "Air_Temp_C",
    "Humidity_pct",
    "Air_Pressure_inHg",
    "Wind_Speed_ms",
    "Wind_Direction_deg",
    "Solar_Irradiance_Wm2",
    "UV_Index",
]

EXOG_BASE_COLS = [
    "PV_Voltage_V",
    "PV_Current_A",
    "Battery_Voltage_V",
    "Battery_Current_A",
    "Battery_Power_W",
    "Avg_Power_to_Battery_W",
    "Battery_Temp_C",
    "Air_Temp_C",
    "Humidity_pct",
    "Air_Pressure_inHg",
    "Wind_Speed_ms",
    "Wind_Direction_deg",
    "Solar_Irradiance_Wm2",
    "UV_Index",
    "Est_Efficiency",
]

TIME_COLS = ["mod_sin", "mod_cos", "doy_sin", "doy_cos"]


# ============================================================
# Small helpers
# ============================================================

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    except Exception:
        pass


def mae_rmse_r2(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        r2 = float("nan")
    else:
        r2 = float(1.0 - np.sum(err ** 2) / denom)
    return mae, rmse, r2


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    safe_std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / safe_std


def clamp_power(x: np.ndarray | float, lo: float = 0.0, hi: float | None = None):
    if hi is None:
        return np.maximum(x, lo)
    return np.clip(x, lo, hi)


# ============================================================
# Log loading / preprocessing
# ============================================================

def fix_legacy_weather_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    If older logs had short headers but rows had 7 extra weather columns,
    pandas may load them as Unnamed columns.
    """
    df = df.copy()

    if all(c in df.columns for c in WEATHER_7):
        return df

    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if ("Est_Efficiency" in df.columns) and (len(unnamed) >= 7):
        unnamed_sorted = sorted(unnamed, key=lambda x: int(str(x).split(":")[1]))
        for weather_col, unnamed_col in zip(WEATHER_7, unnamed_sorted[:7]):
            df[weather_col] = df[unnamed_col]

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"], errors="coerce")

    mod = ts.dt.hour * 60 + ts.dt.minute
    df["mod_sin"] = np.sin(2 * np.pi * mod / 1440.0)
    df["mod_cos"] = np.cos(2 * np.pi * mod / 1440.0)

    doy = ts.dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    return df


def to_1min_grid_merge_asof(df: pd.DataFrame, cols: list[str], max_gap="2min") -> pd.DataFrame:
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").copy()

    tmin = pd.to_datetime(df["timestamp"].min())
    tmax = pd.to_datetime(df["timestamp"].max())
    grid = pd.DataFrame({"timestamp": pd.date_range(tmin, tmax, freq="1min")})

    out = pd.merge_asof(
        grid,
        df[["timestamp"] + cols].sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(max_gap),
    )
    return out


def load_logs(logs_dir: Path) -> pd.DataFrame:
    csvs = sorted(logs_dir.glob("gpg_*.csv"))
    if not csvs:
        raise SystemExit(f"No gpg_*.csv files found in: {logs_dir.resolve()}")

    parts = []
    for path in csvs:
        try:
            part = pd.read_csv(path)
            if not part.empty:
                parts.append(part)
        except Exception as exc:
            print(f"Skipping {path.name}: {exc}")

    if not parts:
        raise SystemExit("No readable CSV files found.")

    df = pd.concat(parts, ignore_index=True)

    if "timestamp" not in df.columns:
        raise SystemExit("Missing 'timestamp' column in logs.")

    df = fix_legacy_weather_columns(df)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    numeric_candidates = [
        "PV_Voltage_V",
        "PV_Current_A",
        "PV_Power_W",
        "Battery_Voltage_V",
        "Battery_Current_A",
        "Battery_Power_W",
        "Avg_Power_to_Battery_W",
        "Daily_Energy_kWh",
        "Daily_Charge_Ah",
        "History_Max_Input_Voltage_V",
        "Lifetime_Energy_Gen_kWh",
        "Lifetime_Charge_Ah",
        "Battery_Temp_C",
        "Air_Temp_C",
        "Humidity_pct",
        "Air_Pressure_inHg",
        "Wind_Speed_ms",
        "Wind_Direction_deg",
        "Solar_Irradiance_Wm2",
        "UV_Index",
        "Est_Efficiency",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Battery_Power_W" not in df.columns and {"Battery_Voltage_V", "Battery_Current_A"}.issubset(df.columns):
        df["Battery_Power_W"] = df["Battery_Voltage_V"] * df["Battery_Current_A"]

    if "Avg_Power_to_Battery_W" not in df.columns and "Battery_Power_W" in df.columns:
        df["Avg_Power_to_Battery_W"] = df["Battery_Power_W"]

    df = add_time_features(df)
    return df


def prepare_dataframe(logs_dir: Path, max_gap: str) -> tuple[pd.DataFrame, list[str]]:
    raw = load_logs(logs_dir)

    available_exog = [c for c in EXOG_BASE_COLS if c in raw.columns]
    if TARGET not in raw.columns:
        raise SystemExit(f"Missing target column: {TARGET}")

    feature_cols = available_exog + TIME_COLS
    needed = ["timestamp", TARGET] + feature_cols
    work = raw[needed].copy()

    grid_cols = [TARGET] + feature_cols
    df_1m = to_1min_grid_merge_asof(work, cols=grid_cols, max_gap=max_gap)

    df_1m = df_1m.dropna(subset=[TARGET]).reset_index(drop=True)
    df_1m = df_1m.dropna(subset=feature_cols + [TARGET]).reset_index(drop=True)

    if len(df_1m) < 200:
        raise SystemExit(f"Not enough clean 1-minute rows after preprocessing: {len(df_1m)}")

    return df_1m, feature_cols


# ============================================================
# Split and window builders
# ============================================================

def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    return train_df, val_df, test_df


def build_ftdnn_windows(df: pd.DataFrame, exog_cols: list[str], lookback: int, horizon: int):
    exog = df[exog_cols].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    ts = pd.to_datetime(df["timestamp"]).to_numpy()

    X_rows = []
    resid_rows = []
    base_rows = []
    actual_rows = []
    time_rows = []

    max_start = len(df) - lookback - horizon + 1
    for i in range(max_start):
        end_idx = i + lookback
        base_idx = end_idx - 1
        target_idx = base_idx + horizon

        x_hist = exog[i:end_idx].reshape(-1)
        base_power = y[base_idx]
        actual_future = y[target_idx]
        resid = actual_future - base_power

        X_rows.append(x_hist)
        resid_rows.append(resid)
        base_rows.append(base_power)
        actual_rows.append(actual_future)
        time_rows.append(ts[target_idx])

    return {
        "X": np.asarray(X_rows, dtype=np.float32),
        "resid": np.asarray(resid_rows, dtype=np.float32),
        "base_power": np.asarray(base_rows, dtype=np.float32),
        "actual_future": np.asarray(actual_rows, dtype=np.float32),
        "target_time": np.asarray(time_rows),
    }


def build_narx_windows(df: pd.DataFrame, exog_cols: list[str], lookback_in: int, lookback_out: int, horizon: int):
    exog = df[exog_cols].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    ts = pd.to_datetime(df["timestamp"]).to_numpy()

    lookback = max(lookback_in, lookback_out)

    X_rows = []
    resid_rows = []
    base_rows = []
    actual_rows = []
    time_rows = []

    max_start = len(df) - lookback - horizon + 1
    for i in range(max_start):
        end_idx = i + lookback
        base_idx = end_idx - 1
        target_idx = base_idx + horizon

        exog_hist = exog[i + lookback - lookback_in:end_idx].reshape(-1)
        y_hist = y[i + lookback - lookback_out:end_idx].reshape(-1)

        x_row = np.concatenate([exog_hist, y_hist], axis=0)

        base_power = y[base_idx]
        actual_future = y[target_idx]
        resid = actual_future - base_power

        X_rows.append(x_row)
        resid_rows.append(resid)
        base_rows.append(base_power)
        actual_rows.append(actual_future)
        time_rows.append(ts[target_idx])

    return {
        "X": np.asarray(X_rows, dtype=np.float32),
        "resid": np.asarray(resid_rows, dtype=np.float32),
        "base_power": np.asarray(base_rows, dtype=np.float32),
        "actual_future": np.asarray(actual_rows, dtype=np.float32),
        "target_time": np.asarray(time_rows),
    }


def build_lstm_windows(df: pd.DataFrame, exog_cols: list[str], lookback: int, horizon: int):
    seq_cols = exog_cols + [TARGET]
    X_seq = df[seq_cols].to_numpy(np.float32)
    y = df[TARGET].to_numpy(np.float32)
    ts = pd.to_datetime(df["timestamp"]).to_numpy()

    X_rows = []
    resid_rows = []
    base_rows = []
    actual_rows = []
    time_rows = []

    max_start = len(df) - lookback - horizon + 1
    for i in range(max_start):
        end_idx = i + lookback
        base_idx = end_idx - 1
        target_idx = base_idx + horizon

        seq_hist = X_seq[i:end_idx]
        base_power = y[base_idx]
        actual_future = y[target_idx]
        resid = actual_future - base_power

        X_rows.append(seq_hist)
        resid_rows.append(resid)
        base_rows.append(base_power)
        actual_rows.append(actual_future)
        time_rows.append(ts[target_idx])

    return {
        "X": np.asarray(X_rows, dtype=np.float32),
        "resid": np.asarray(resid_rows, dtype=np.float32),
        "base_power": np.asarray(base_rows, dtype=np.float32),
        "actual_future": np.asarray(actual_rows, dtype=np.float32),
        "target_time": np.asarray(time_rows),
    }


# ============================================================
# Datasets
# ============================================================

class FlatDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32).reshape(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


# ============================================================
# Models
# ============================================================

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int]):
        super().__init__()

        layers = []
        prev = input_dim

        if len(hidden_sizes) == 0:
            layers.append(nn.Linear(prev, 1))
        else:
            for hidden in hidden_sizes:
                layers.append(nn.Linear(prev, hidden))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.10))
                prev = hidden
            layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden_size=96, num_layers=2, dropout=0.15):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


# ============================================================
# Training / evaluation
# ============================================================

def fit_standardizer(X_train: np.ndarray):
    if X_train.ndim == 2:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
    elif X_train.ndim == 3:
        mean = X_train.mean(axis=(0, 1))
        std = X_train.std(axis=(0, 1))
    else:
        raise ValueError("X_train must be 2D or 3D")
    std = np.where(std < 1e-8, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def apply_standardizer(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    if X.ndim == 2:
        return standardize(X, mean, std).astype(np.float32)
    if X.ndim == 3:
        return ((X - mean[None, None, :]) / std[None, None, :]).astype(np.float32)
    raise ValueError("X must be 2D or 3D")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(2, patience // 2)
    )

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if bad_epochs >= patience:
            print("Early stopping.")
            break

    model.load_state_dict(best_state)
    return history, best_val


def predict_residuals(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    preds = []

    ds = SequenceDataset(X, np.zeros(len(X), dtype=np.float32)) if X.ndim == 3 else FlatDataset(X, np.zeros(len(X), dtype=np.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for xb, _ in dl:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy().reshape(-1)
            preds.append(pred)

    if not preds:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(preds).astype(np.float32)


def run_supervised_experiment(
    model_name: str,
    model: nn.Module,
    train_data: dict,
    val_data: dict,
    test_data: dict,
    out_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
):
    X_train_raw = train_data["X"]
    X_val_raw = val_data["X"]
    X_test_raw = test_data["X"]

    y_train_resid = train_data["resid"].astype(np.float32)
    y_val_resid = val_data["resid"].astype(np.float32)
    y_test_resid = test_data["resid"].astype(np.float32)

    x_mean, x_std = fit_standardizer(X_train_raw)
    X_train = apply_standardizer(X_train_raw, x_mean, x_std)
    X_val = apply_standardizer(X_val_raw, x_mean, x_std)
    X_test = apply_standardizer(X_test_raw, x_mean, x_std)

    y_mean = float(y_train_resid.mean())
    y_std = float(y_train_resid.std() if y_train_resid.std() > 1e-6 else 1.0)

    y_train_s = ((y_train_resid - y_mean) / y_std).astype(np.float32)
    y_val_s = ((y_val_resid - y_mean) / y_std).astype(np.float32)

    train_ds = SequenceDataset(X_train, y_train_s) if X_train.ndim == 3 else FlatDataset(X_train, y_train_s)
    val_ds = SequenceDataset(X_val, y_val_s) if X_val.ndim == 3 else FlatDataset(X_val, y_val_s)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = model.to(device)
    history, best_val = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        lr=lr,
        patience=patience,
    )

    pred_resid_test_s = predict_residuals(model, X_test, device=device, batch_size=batch_size)
    pred_resid_test = pred_resid_test_s * y_std + y_mean

    pred_power_test = test_data["base_power"] + pred_resid_test
    power_hi = float(np.nanmax(train_data["actual_future"])) * 1.25
    pred_power_test = clamp_power(pred_power_test, lo=0.0, hi=power_hi)

    actual_test = test_data["actual_future"]
    persistence_test = test_data["base_power"]

    mae, rmse, r2 = mae_rmse_r2(actual_test, pred_power_test)
    p_mae, p_rmse, p_r2 = mae_rmse_r2(actual_test, persistence_test)

    metrics = {
        "model": model_name,
        "test_mae_W": mae,
        "test_rmse_W": rmse,
        "test_r2": r2,
        "persistence_mae_W": p_mae,
        "persistence_rmse_W": p_rmse,
        "persistence_r2": p_r2,
        "mae_improvement_vs_persistence_W": p_mae - mae,
        "rmse_improvement_vs_persistence_W": p_rmse - rmse,
        "best_val_loss_scaled": best_val,
        "n_test": int(len(actual_test)),
    }

    # save model bundle
    model_path = out_dir / f"{model_name}.pt"
    bundle_path = out_dir / f"{model_name}_bundle.npz"
    torch.save(model.state_dict(), model_path)
    np.savez(
        bundle_path,
        x_mean=x_mean,
        x_std=x_std,
        y_mean=np.array([y_mean], dtype=np.float32),
        y_std=np.array([y_std], dtype=np.float32),
    )

    pred_df = pd.DataFrame({
        "target_time": pd.to_datetime(test_data["target_time"]),
        "actual_power_W": actual_test,
        "persistence_power_W": persistence_test,
        f"{model_name}_power_W": pred_power_test,
        f"{model_name}_residual_pred_W": pred_resid_test,
    })

    return {
        "metrics": metrics,
        "history": history,
        "pred_df": pred_df,
        "model_path": str(model_path),
        "bundle_path": str(bundle_path),
    }


# ============================================================
# Plot helpers
# ============================================================

def plot_training_curves(histories: dict[str, dict], out_path: Path):
    plt.figure(figsize=(12, 6))
    for name, hist in histories.items():
        plt.plot(hist["val_loss"], label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss (scaled MSE)")
    plt.title("Validation curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_last_k(predictions: pd.DataFrame, out_path: Path, title: str, k: int = 200):
    if predictions.empty:
        return

    view = predictions.tail(k).copy()

    plt.figure(figsize=(14, 5))
    plt.plot(view["target_time"], view["actual_power_W"], label="Actual", linewidth=2)
    plt.plot(view["target_time"], view["persistence_power_W"], label="Persistence", linewidth=2)

    for col in view.columns:
        if col.endswith("_power_W") and col not in ("actual_power_W", "persistence_power_W"):
            plt.plot(view["target_time"], view[col], label=col.replace("_power_W", ""), linewidth=1.8)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("PV_Power_W")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_metric_bars(metrics_df: pd.DataFrame, metric_col: str, out_path: Path, title: str):
    plot_df = metrics_df.copy()
    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["model"], plot_df[metric_col])
    plt.ylabel(metric_col)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-dir", default="logs")
    ap.add_argument("--out-dir", default="final_project_outputs")
    ap.add_argument("--max-gap", default="2min")
    ap.add_argument("--lookback", type=int, default=120, help="Past minutes used as input history")
    ap.add_argument("--narx-output-lookback", type=int, default=120, help="Past output history for NARX")
    ap.add_argument("--horizon", type=int, default=60, help="Forecast horizon in minutes")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)

    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --------------------------------------------------------
    # Load and preprocess
    # --------------------------------------------------------
    df_1m, exog_cols = prepare_dataframe(logs_dir=logs_dir, max_gap=args.max_gap)
    train_df, val_df, test_df = chronological_split(df_1m)

    min_needed = max(args.lookback, args.narx_output_lookback) + args.horizon + 20
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        if len(split_df) < min_needed:
            raise SystemExit(
                f"{name} split too small after preprocessing ({len(split_df)} rows). "
                f"Need more than about {min_needed} rows. Reduce lookback/horizon or add more logs."
            )

    summary = {
        "rows_after_preprocessing": int(len(df_1m)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "feature_cols": exog_cols,
        "lookback": int(args.lookback),
        "narx_output_lookback": int(args.narx_output_lookback),
        "horizon": int(args.horizon),
        "max_gap": args.max_gap,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    all_metrics = []
    all_histories = {}
    pred_table = None

    # ========================================================
    # SECTION 1 - FTDNN
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 1 - FTDNN")
    print("=" * 70)

    ftdnn_train = build_ftdnn_windows(train_df, exog_cols=exog_cols, lookback=args.lookback, horizon=args.horizon)
    ftdnn_val = build_ftdnn_windows(val_df, exog_cols=exog_cols, lookback=args.lookback, horizon=args.horizon)
    ftdnn_test = build_ftdnn_windows(test_df, exog_cols=exog_cols, lookback=args.lookback, horizon=args.horizon)

    ftdnn_linear = run_supervised_experiment(
        model_name="ftdnn_linear",
        model=MLPRegressor(input_dim=ftdnn_train["X"].shape[1], hidden_sizes=[]),
        train_data=ftdnn_train,
        val_data=ftdnn_val,
        test_data=ftdnn_test,
        out_dir=out_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    all_metrics.append(ftdnn_linear["metrics"])
    all_histories["ftdnn_linear"] = ftdnn_linear["history"]

    ftdnn_nonlinear = run_supervised_experiment(
        model_name="ftdnn_nonlinear",
        model=MLPRegressor(input_dim=ftdnn_train["X"].shape[1], hidden_sizes=[128, 64]),
        train_data=ftdnn_train,
        val_data=ftdnn_val,
        test_data=ftdnn_test,
        out_dir=out_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    all_metrics.append(ftdnn_nonlinear["metrics"])
    all_histories["ftdnn_nonlinear"] = ftdnn_nonlinear["history"]

    # ========================================================
    # SECTION 2 - NARX
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 2 - NARX")
    print("=" * 70)

    narx_train = build_narx_windows(
        train_df,
        exog_cols=exog_cols,
        lookback_in=args.lookback,
        lookback_out=args.narx_output_lookback,
        horizon=args.horizon,
    )
    narx_val = build_narx_windows(
        val_df,
        exog_cols=exog_cols,
        lookback_in=args.lookback,
        lookback_out=args.narx_output_lookback,
        horizon=args.horizon,
    )
    narx_test = build_narx_windows(
        test_df,
        exog_cols=exog_cols,
        lookback_in=args.lookback,
        lookback_out=args.narx_output_lookback,
        horizon=args.horizon,
    )

    narx_linear = run_supervised_experiment(
        model_name="narx_linear",
        model=MLPRegressor(input_dim=narx_train["X"].shape[1], hidden_sizes=[]),
        train_data=narx_train,
        val_data=narx_val,
        test_data=narx_test,
        out_dir=out_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    all_metrics.append(narx_linear["metrics"])
    all_histories["narx_linear"] = narx_linear["history"]

    narx_nonlinear = run_supervised_experiment(
        model_name="narx_nonlinear",
        model=MLPRegressor(input_dim=narx_train["X"].shape[1], hidden_sizes=[128, 64]),
        train_data=narx_train,
        val_data=narx_val,
        test_data=narx_test,
        out_dir=out_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    all_metrics.append(narx_nonlinear["metrics"])
    all_histories["narx_nonlinear"] = narx_nonlinear["history"]

    # ========================================================
    # SECTION 3 - LSTM
    # ========================================================
    print("\n" + "=" * 70)
    print("SECTION 3 - LSTM")
    print("=" * 70)

    lstm_train = build_lstm_windows(train_df, exog_cols=exog_cols, lookback=args.lookback, horizon=args.horizon)
    lstm_val = build_lstm_windows(val_df, exog_cols=exog_cols, lookback=args.lookback, horizon=args.horizon)
    lstm_test = build_lstm_windows(test_df, exog_cols=exog_cols, lookback=args.lookback, horizon=args.horizon)

    lstm_out = run_supervised_experiment(
        model_name="lstm",
        model=LSTMRegressor(n_features=lstm_train["X"].shape[2], hidden_size=48, num_layers=1, dropout=0.10),
        train_data=lstm_train,
        val_data=lstm_val,
        test_data=lstm_test,
        out_dir=out_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
    )
    all_metrics.append(lstm_out["metrics"])
    all_histories["lstm"] = lstm_out["history"]

    # --------------------------------------------------------
    # Combine predictions onto the same test timeline
    # --------------------------------------------------------
    base_pred = pd.DataFrame({
        "target_time": pd.to_datetime(lstm_test["target_time"]),
        "actual_power_W": lstm_test["actual_future"],
        "persistence_power_W": lstm_test["base_power"],
    })

    for result in [ftdnn_linear, ftdnn_nonlinear, narx_linear, narx_nonlinear, lstm_out]:
        add_cols = [
            c for c in result["pred_df"].columns
            if c not in ("target_time", "actual_power_W", "persistence_power_W")
        ]
        merge_df = result["pred_df"][["target_time"] + add_cols].copy()
        merge_df = merge_df.drop_duplicates(subset="target_time", keep="last")
        base_pred = base_pred.merge(
            merge_df,
            on="target_time",
            how="left",
        )

    pred_table = base_pred.sort_values("target_time").reset_index(drop=True)
    pred_table.to_csv(out_dir / "comparison_predictions.csv", index=False)

    # --------------------------------------------------------
    # Metrics table
    # --------------------------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values("test_mae_W").reset_index(drop=True)
    metrics_df.to_csv(out_dir / "comparison_metrics.csv", index=False)

    with open(out_dir / "comparison_metrics.txt", "w", encoding="utf-8") as f:
        f.write(f"Forecast horizon: +{args.horizon} minute(s)\n")
        f.write(f"Lookback: {args.lookback} minute(s)\n")
        f.write(f"Rows after preprocessing: {len(df_1m)}\n")
        f.write(f"Feature columns: {exog_cols}\n\n")
        for _, row in metrics_df.iterrows():
            f.write(
                f"{row['model']:<18} "
                f"MAE={row['test_mae_W']:.3f} W  "
                f"RMSE={row['test_rmse_W']:.3f} W  "
                f"R^2={row['test_r2']:.4f}  "
                f"MAE improvement vs persistence={row['mae_improvement_vs_persistence_W']:.3f} W\n"
            )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------
    plot_training_curves(all_histories, out_dir / "training_curves.png")
    plot_last_k(
        pred_table,
        out_path=out_dir / "comparison_last200.png",
        title=f"Solar forecasting comparison (last ~200 test points, +{args.horizon} min)",
        k=200,
    )
    plot_metric_bars(
        metrics_df,
        metric_col="test_mae_W",
        out_path=out_dir / "metric_mae.png",
        title=f"Model comparison by MAE (+{args.horizon} min)",
    )
    plot_metric_bars(
        metrics_df,
        metric_col="test_rmse_W",
        out_path=out_dir / "metric_rmse.png",
        title=f"Model comparison by RMSE (+{args.horizon} min)",
    )

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    for _, row in metrics_df.iterrows():
        print(
            f"{row['model']:<18} "
            f"MAE={row['test_mae_W']:.3f}  "
            f"RMSE={row['test_rmse_W']:.3f}  "
            f"R^2={row['test_r2']:.4f}  "
            f"MAE_vs_persist={row['mae_improvement_vs_persistence_W']:.3f}"
        )

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
