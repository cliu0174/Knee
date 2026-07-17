from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover
    raise RuntimeError("xgboost is required for this script") from exc

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyTorch is required for frontier baselines") from exc

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_feature_ablation import feature_bucket  # noqa: E402
from train_knee_baselines import META_COLUMNS, detect_window_prefix  # noqa: E402


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mape_percent": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0),
        "r2": float(r2_score(y_true, y_pred)) if y_true.size > 1 else float("nan"),
    }


def select_columns(df: pd.DataFrame, prefix: str, include_local_deltaq: bool) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in META_COLUMNS or col == f"{prefix}_window_complete":
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) <= 1:
            continue
        if feature_bucket(col, prefix) in {"raw", "deltaq"}:
            cols.append(col)
            continue
        if include_local_deltaq and col.startswith(f"{prefix}_ldq_"):
            cols.append(col)
    return cols


def local_deltaq_columns(df: pd.DataFrame, prefix: str) -> list[str]:
    cols = [c for c in df.columns if c.startswith(f"{prefix}_ldq_") and pd.api.types.is_numeric_dtype(df[c])]
    return sorted([c for c in cols if df[c].nunique(dropna=True) > 1])


def make_xgb(params: dict[str, object], seed: int) -> XGBRegressor:
    return XGBRegressor(
        n_estimators=int(float(params["n_estimators"])),
        max_depth=int(float(params["max_depth"])),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        reg_lambda=float(params["reg_lambda"]),
        reg_alpha=float(params["reg_alpha"]),
        min_child_weight=float(params["min_child_weight"]),
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=4,
    )


def cv_score_model(model, x: pd.DataFrame, y: np.ndarray, seed: int, n_splits: int, n_repeats: int) -> dict[str, float]:
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    maes: list[float] = []
    for fold, (tr, va) in enumerate(splitter.split(x)):
        estimator = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            estimator.fit(x.iloc[tr], y[tr])
        pred = np.asarray(estimator.predict(x.iloc[va]), dtype=float)
        maes.append(float(mean_absolute_error(y[va], pred)))
    return {
        "dev_cv_mae_mean": float(np.mean(maes)),
        "dev_cv_mae_std": float(np.std(maes)),
        "dev_cv_mae_max": float(np.max(maes)),
        "dev_selection_score": float(np.mean(maes) + 0.10 * np.std(maes) + 0.10 * np.max(maes)),
    }


def cv_select_xgb(
    params_df: pd.DataFrame,
    x: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    n_splits: int,
    n_repeats: int,
    max_candidates: int,
) -> tuple[str, dict[str, object], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    subset = params_df.head(max_candidates).copy()
    splitter = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for _, row in subset.iterrows():
        candidate_id = str(row["candidate_id"])
        params = {k: row[k] for k in row.index if k != "candidate_id"}
        maes: list[float] = []
        for fold, (tr, va) in enumerate(splitter.split(x)):
            model = make_xgb(params, seed + fold)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(x.iloc[tr], y[tr])
            pred = np.asarray(model.predict(x.iloc[va]), dtype=float)
            maes.append(float(mean_absolute_error(y[va], pred)))
        rows.append(
            {
                "candidate_id": candidate_id,
                "dev_cv_mae_mean": float(np.mean(maes)),
                "dev_cv_mae_std": float(np.std(maes)),
                "dev_cv_mae_max": float(np.max(maes)),
                "dev_selection_score": float(np.mean(maes) + 0.10 * np.std(maes) + 0.10 * np.max(maes)),
                **params,
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["dev_selection_score", "dev_cv_mae_mean"]).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    selected_params = {k: selected[k] for k in subset.columns if k != "candidate_id"}
    return str(selected["candidate_id"]), selected_params, ranking


class CNNEncoderRegressor(nn.Module):
    def __init__(self, n_features: int, channels: int = 16, embedding_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embedding = nn.Sequential(nn.Flatten(), nn.Linear(channels, embedding_dim), nn.ReLU())
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        z = self.encoder(x.unsqueeze(1))
        emb = self.embedding(z)
        if return_embedding:
            return emb
        return self.head(emb).squeeze(-1)


class TcnEcaRegressor(nn.Module):
    def __init__(self, n_features: int, channels: int = 24):
        super().__init__()
        self.in_proj = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        self.block1 = nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.block2 = nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.relu(self.in_proj(x.unsqueeze(1)))
        z = torch.relu(self.block1(z)[..., : x.shape[1]]) + z
        z = torch.relu(self.block2(z)[..., : x.shape[1]]) + z
        z = z * self.attn(z)
        return self.head(z).squeeze(-1)


def train_torch_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    epochs: int,
    lr: float,
) -> tuple[nn.Module, float]:
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = model.to(device)
    y_mean = float(y_train.mean())
    y_std = float(y_train.std() + 1e-9)
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor((y_train - y_mean) / y_std, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=min(16, len(train_ds)), shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.MSELoss()
    best_state = None
    best_mae = float("inf")
    patience = 35
    stale = 0
    xv = torch.tensor(x_valid, dtype=torch.float32).to(device)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xv).cpu().numpy() * y_std + y_mean
        mae = float(mean_absolute_error(y_valid, pred))
        if mae < best_mae:
            best_mae = mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model._target_mean = y_mean  # type: ignore[attr-defined]
    model._target_std = y_std  # type: ignore[attr-defined]
    return model, best_mae


def torch_predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()
    return pred * float(model._target_std) + float(model._target_mean)  # type: ignore[attr-defined]


def add_result(
    rows: list[dict[str, object]],
    predictions: list[dict[str, object]],
    method: str,
    family: str,
    input_type: str,
    selection_note: str,
    df: pd.DataFrame,
    dev_mask: np.ndarray,
    external_mask: np.ndarray,
    pred_all: np.ndarray,
) -> None:
    for split_name, mask in [("development_train_primary", dev_mask), ("external_secondary", external_mask)]:
        rows.append(
            {
                "method": method,
                "family": family,
                "input_type": input_type,
                "selection_note": selection_note,
                "split": split_name,
                "n": int(mask.sum()),
                **regression_metrics(df.loc[mask, "knee_cycle"].to_numpy(dtype=float), pred_all[mask]),
            }
        )
    for row_index, row in df.iterrows():
        predictions.append(
            {
                "method": method,
                "family": family,
                "input_type": input_type,
                "cell_id": row["cell_id"],
                "batch": row["batch"],
                "official_split": row["official_split"],
                "observed_knee_cycle": float(row["knee_cycle"]),
                "predicted_knee_cycle": float(pred_all[row_index]),
                "abs_error": float(abs(float(row["knee_cycle"]) - float(pred_all[row_index]))),
            }
        )


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.feature_path)
    df = df[df["knee_after_window"].astype(bool)].copy().reset_index(drop=True)
    prefix = detect_window_prefix(df)
    raw_cols = select_columns(df, prefix, include_local_deltaq=False)
    local_cols = select_columns(df, prefix, include_local_deltaq=True)
    ldq_cols = local_deltaq_columns(df, prefix)
    y = df["knee_cycle"].to_numpy(dtype=float)
    dev_mask = df["official_split"].isin(["train", "primary_test"]).to_numpy()
    external_mask = (df["official_split"] == "secondary_test").to_numpy()
    dev_idx = np.flatnonzero(dev_mask)
    tr_idx, va_idx = train_test_split(dev_idx, test_size=0.22, random_state=args.random_state, shuffle=True)

    results: list[dict[str, object]] = []
    predictions: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    params_df = pd.read_csv(args.xgb_params_path)

    model_specs = [
        ("Ridge", "linear", Pipeline([("scale", StandardScaler()), ("model", RidgeCV(alphas=np.logspace(-3, 4, 30)))])),
        (
            "ElasticNet",
            "linear",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.8, 0.95], alphas=np.logspace(-3, 2, 20), cv=5, max_iter=5000)),
                ]
            ),
        ),
        ("SVR-RBF", "kernel", Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", C=20.0, gamma="scale", epsilon=15.0))])),
        ("kNN", "distance", Pipeline([("scale", StandardScaler()), ("model", KNeighborsRegressor(n_neighbors=5, weights="distance"))])),
        ("RandomForest", "tree", RandomForestRegressor(n_estimators=700, min_samples_leaf=2, max_features="sqrt", random_state=args.random_state, n_jobs=-1)),
        ("ExtraTrees", "tree", ExtraTreesRegressor(n_estimators=700, min_samples_leaf=2, max_features="sqrt", random_state=args.random_state, n_jobs=-1)),
        (
            "GaussianProcess",
            "probabilistic",
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        GaussianProcessRegressor(
                            kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
                            alpha=1e-6,
                            normalize_y=True,
                            random_state=args.random_state,
                        ),
                    ),
                ]
            ),
        ),
    ]

    for input_name, cols in [("raw_deltaq", raw_cols), ("local_deltaq", local_cols)]:
        x = df[cols]
        x_dev = x.loc[dev_mask].reset_index(drop=True)
        y_dev = y[dev_mask]
        for model_name, family, model in model_specs:
            score = cv_score_model(model, x_dev, y_dev, args.random_state, args.n_splits, args.n_repeats)
            selection_rows.append({"method": model_name, "input_type": input_name, **score})
            estimator = clone(model)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                estimator.fit(x.loc[dev_mask], y[dev_mask])
            pred_all = np.asarray(estimator.predict(x), dtype=float)
            add_result(results, predictions, model_name, family, input_name, "fixed estimator; repeated CV scored on development", df, dev_mask, external_mask, pred_all)

        xgb_id, xgb_params, ranking = cv_select_xgb(params_df, x_dev, y_dev, args.random_state, args.n_splits, args.n_repeats, args.max_xgb_candidates)
        ranking.insert(0, "input_type", input_name)
        ranking.to_csv(output_dir / f"xgb_cv_selection_{input_name}.csv", index=False)
        xgb = make_xgb(xgb_params, args.random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            xgb.fit(x.loc[dev_mask], y[dev_mask])
        pred_all = np.asarray(xgb.predict(x), dtype=float)
        add_result(results, predictions, "XGBoost", "boosting", input_name, f"development CV selected {xgb_id}", df, dev_mask, external_mask, pred_all)

    # Frontier baseline 1: adapted CNN-XGBoost.
    x_scaler = StandardScaler()
    x_ldq = x_scaler.fit_transform(df[ldq_cols].to_numpy(dtype=float))
    cnn = CNNEncoderRegressor(n_features=x_ldq.shape[1], channels=16, embedding_dim=16)
    cnn, cnn_val_mae = train_torch_model(cnn, x_ldq[tr_idx], y[tr_idx], x_ldq[va_idx], y[va_idx], args.random_state, args.epochs, lr=5e-3)
    with torch.no_grad():
        emb_all = cnn(torch.tensor(x_ldq, dtype=torch.float32), return_embedding=True).cpu().numpy()
    emb_cols = [f"cnn_emb_{i}" for i in range(emb_all.shape[1])]
    x_cnn = pd.concat([df[raw_cols].reset_index(drop=True), pd.DataFrame(emb_all, columns=emb_cols)], axis=1)
    xgb_id, xgb_params, ranking = cv_select_xgb(params_df, x_cnn.loc[dev_mask].reset_index(drop=True), y[dev_mask], args.random_state, args.n_splits, args.n_repeats, args.max_xgb_candidates)
    ranking.to_csv(output_dir / "cnn_xgboost_cv_selection.csv", index=False)
    cnn_xgb = make_xgb(xgb_params, args.random_state)
    cnn_xgb.fit(x_cnn.loc[dev_mask], y[dev_mask])
    pred_all = np.asarray(cnn_xgb.predict(x_cnn), dtype=float)
    add_result(results, predictions, "Adapted CNN-XGBoost", "frontier", "cnn_embedding_plus_raw_deltaq", f"CNN val MAE={cnn_val_mae:.2f}; XGB development CV selected {xgb_id}", df, dev_mask, external_mask, pred_all)

    # Frontier baseline 2: adapted TCN-ECA direct regressor.
    tcn = TcnEcaRegressor(n_features=x_ldq.shape[1], channels=24)
    tcn, tcn_val_mae = train_torch_model(tcn, x_ldq[tr_idx], y[tr_idx], x_ldq[va_idx], y[va_idx], args.random_state + 17, args.epochs, lr=3e-3)
    pred_all = torch_predict(tcn, x_ldq)
    add_result(results, predictions, "Adapted TCN-ECA", "frontier", "local_deltaq_sequence", f"fixed architecture; holdout-development val MAE={tcn_val_mae:.2f}", df, dev_mask, external_mask, pred_all)

    metrics = pd.DataFrame(results)
    pred_df = pd.DataFrame(predictions)
    eval_df = metrics[metrics["split"] == "external_secondary"].copy()
    ranking = eval_df.sort_values(["mae", "rmse"]).reset_index(drop=True)
    metrics.to_csv(output_dir / "horizontal_metrics.csv", index=False)
    ranking.to_csv(output_dir / "horizontal_external_ranking.csv", index=False)
    pred_df.to_csv(output_dir / "horizontal_predictions.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output_dir / "traditional_cv_scores.csv", index=False)
    pd.DataFrame({"feature": raw_cols}).to_csv(output_dir / "features_raw_deltaq.csv", index=False)
    pd.DataFrame({"feature": local_cols}).to_csv(output_dir / "features_local_deltaq.csv", index=False)
    pd.DataFrame({"feature": ldq_cols}).to_csv(output_dir / "features_frontier_sequence.csv", index=False)
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "feature_path": args.feature_path,
                "development_split": "official train + primary_test",
                "external_split": "official secondary_test",
                "n_splits": args.n_splits,
                "n_repeats": args.n_repeats,
                "max_xgb_candidates": args.max_xgb_candidates,
                "frontier_note": "CNN-XGBoost and TCN-ECA are adapted reproducible baselines, not exact reimplementations of the original papers.",
            },
            fp,
            indent=2,
        )
    print("Saved outputs to", output_dir)
    print("\nExternal ranking:")
    print(ranking[["method", "family", "input_type", "mae", "rmse", "mape_percent", "r2", "selection_note"]].round(4).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-path", default="data/early_features_N150_local_deltaq_top40_20170630.csv")
    parser.add_argument("--xgb-params-path", default="results/hybrid_hpo_local_deltaq_top40_N150/xgb_params.csv")
    parser.add_argument("--output-dir", default="results/horizontal_model_comparison_N150")
    parser.add_argument("--random-state", type=int, default=123)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--max-xgb-candidates", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=250)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
