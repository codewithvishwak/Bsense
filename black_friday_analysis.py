
"""Black Friday purchase amount analysis and prediction.

Expected input: the standard Black Friday dataset with a ``Purchase`` column.
The script writes model metrics, predictions, a data summary, and charts to
``outputs/``. Use ``--synthetic`` to verify the pipeline without a dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


RANDOM_STATE = 42
TARGET = "Purchase"


def make_synthetic_data(rows: int = 2_500) -> pd.DataFrame:
    """Create a small dataset with the same schema as the public benchmark."""
    rng = np.random.default_rng(RANDOM_STATE)
    age = rng.choice(["0-17", "18-25", "26-35", "36-45", "46-50", "51-55"], rows)
    occupation = rng.integers(0, 21, rows).astype(str)
    city = rng.choice(["A", "B", "C"], rows, p=[0.45, 0.30, 0.25])
    stay = rng.choice(["0", "1", "2", "3", "4+", "5+"], rows)
    product = rng.choice([f"P{n}" for n in range(1, 31)], rows)
    age_effect = pd.Series(age).map(
        {"0-17": -500, "18-25": 0, "26-35": 350, "36-45": 500, "46-50": 450, "51-55": 250}
    ).to_numpy()
    city_effect = pd.Series(city).map({"A": 0, "B": 250, "C": 450}).to_numpy()
    purchase = (
        8_000
        + age_effect
        + city_effect
        + rng.normal(0, 1_000, rows)
        + rng.integers(0, 15, rows) * 120
    )
    return pd.DataFrame(
        {
            "User_ID": rng.integers(100_000, 110_000, rows),
            "Product_ID": product,
            "Gender": rng.choice(["F", "M"], rows),
            "Age": age,
            "Occupation": occupation,
            "City_Category": city,
            "Stay_In_Current_City_Years": stay,
            "Marital_Status": rng.integers(0, 2, rows),
            "Purchase": np.maximum(purchase, 0).round(0),
        }
    )


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET not in data.columns:
        raise ValueError(f"Input must contain a '{TARGET}' column.")
    data = data.copy()
    data = data.dropna(subset=[TARGET])
    y = pd.to_numeric(data.pop(TARGET), errors="coerce")
    valid = y.notna()
    data, y = data.loc[valid], y.loc[valid]
    # IDs identify records or products; treating them as numeric values adds noise.
    data = data.drop(columns=["User_ID", "Product_ID"], errors="ignore")
    for column in [column for column in data.columns if data[column].dtype == object]:
        data[column] = data[column].astype("string")
    return data, y


def build_model(features: pd.DataFrame) -> Pipeline:
    categorical = features.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    numeric = [column for column in features.columns if column not in categorical]
    preprocess = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=180,
                    min_samples_leaf=2,
                    max_features=0.8,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def save_charts(data: pd.DataFrame, predictions: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if "Age" in data.columns:
        age_summary = data.groupby("Age", sort=False)[TARGET].mean().sort_values(ascending=False)
        age_summary.plot(kind="bar", title="Average Purchase by Age Group", ylabel="Purchase")
        plt.tight_layout()
        plt.savefig(output_dir / "average_purchase_by_age.png", dpi=150)
        plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(predictions["actual"], predictions["predicted"], alpha=0.25, s=8)
    bounds = [predictions[["actual", "predicted"]].min().min(), predictions[["actual", "predicted"]].max().max()]
    plt.plot(bounds, bounds, "r--", label="Perfect prediction")
    plt.xlabel("Actual purchase")
    plt.ylabel("Predicted purchase")
    plt.title("Actual vs. Predicted Purchase")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "actual_vs_predicted.png", dpi=150)
    plt.close()


def run_analysis(input_path: Path | None, output_dir: Path, synthetic: bool, max_rows: int) -> dict[str, float]:
    if synthetic:
        raw = make_synthetic_data()
    elif input_path is None:
        raise ValueError("Provide --input path/to/train.csv or use --synthetic.")
    else:
        raw = pd.read_csv(input_path, nrows=max_rows if max_rows > 0 else None)

    features, target = prepare_data(raw)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=RANDOM_STATE
    )
    model = build_model(features)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)
    metrics = {
        "rows": int(len(raw)),
        "features": int(features.shape[1]),
        "mae": float(mean_absolute_error(y_test, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, predicted))),
        "r2": float(r2_score(y_test, predicted)),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({"actual": y_test.to_numpy(), "predicted": predicted})
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    raw.describe(include="all").transpose().to_csv(output_dir / "data_summary.csv")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_charts(raw, predictions, output_dir)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Path to the Black Friday train.csv file.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-rows", type=int, default=100_000, help="Rows to read; 0 means all rows.")
    parser.add_argument("--synthetic", action="store_true", help="Run with generated data.")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    results = run_analysis(arguments.input, arguments.output_dir, arguments.synthetic, arguments.max_rows)
    print(json.dumps(results, indent=2))
