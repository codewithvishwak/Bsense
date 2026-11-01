"""Serve the HTML frontend and a small JSON prediction API."""

from __future__ import annotations

import json
import threading
from io import BytesIO
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from black_friday_analysis import build_model, make_synthetic_data, prepare_data


ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"


def train_demo_model():
    features, target = prepare_data(make_synthetic_data())
    model = build_model(features)
    model.fit(features, target)
    return model, features.columns.tolist()


MODEL, FEATURE_COLUMNS = train_demo_model()
ANALYSIS_DATA = make_synthetic_data()
CHART_LOCK = threading.Lock()


def build_forecasting_report() -> dict[str, object]:
    features, target = prepare_data(ANALYSIS_DATA)
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=3, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_features=0.8, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100, max_features=0.8, min_samples_leaf=2, n_jobs=-1, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }
    results = []
    fitted = {}
    for name, estimator in models.items():
        pipeline = build_model(features).set_params(model=estimator)
        pipeline.fit(x_train, y_train)
        prediction = pipeline.predict(x_test)
        results.append(
            {
                "model": name,
                "rmse": round(float(np.sqrt(mean_squared_error(y_test, prediction))), 2),
                "r2": round(float(r2_score(y_test, prediction)), 4),
            }
        )
        fitted[name] = pipeline
    best = min(results, key=lambda item: item["rmse"])
    importance_pipeline = fitted["Random Forest"]
    names = importance_pipeline.named_steps["preprocess"].get_feature_names_out()
    importances = importance_pipeline.named_steps["model"].feature_importances_
    feature_importance = [
        {"feature": name.split("__", 1)[-1], "importance": round(float(value), 4)}
        for name, value in sorted(zip(names, importances), key=lambda item: item[1], reverse=True)[:10]
    ]
    return {
        "split": {"train": len(x_train), "test": len(x_test), "ratio": "80/20"},
        "models": sorted(results, key=lambda item: item["rmse"]),
        "best_model": best,
        "feature_importance": feature_importance,
        "hypotheses": {
            "product_level": ["Product category and product identity influence purchase value.", "Popular products create concentrated demand."],
            "store_level": ["Age, gender, occupation, and city category influence demand.", "Residence duration and marital status indicate customer context."],
        },
        "workflow": [
            "Hypothesis generation",
            "Data exploration",
            "Cleaning and encoding",
            "80/20 train-test split",
            "Multi-model regression",
            "RMSE and R² comparison",
        ],
    }


def build_analysis() -> dict[str, object]:
    age = (
        ANALYSIS_DATA.groupby("Age", sort=False)["Purchase"]
        .mean()
        .round(0)
        .astype(int)
        .to_dict()
    )
    city = ANALYSIS_DATA.groupby("City_Category")["Purchase"].agg(["mean", "count"]).round(0)
    return {
        "age_average": age,
        "city_average": {key: int(value) for key, value in city["mean"].to_dict().items()},
        "city_share": {key: int(value) for key, value in city["count"].to_dict().items()},
        "total_rows": len(ANALYSIS_DATA),
        "average_purchase": int(ANALYSIS_DATA["Purchase"].mean().round(0)),
    }


def build_analysis_chart(age: str, city: str, prediction: float | None) -> bytes:
    analysis = build_analysis()
    age_values = analysis["age_average"]
    city_values = analysis["city_average"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor="#121415")
    figure.subplots_adjust(wspace=0.32, bottom=0.2)
    for axis in axes:
        axis.set_facecolor("#121415")
        axis.tick_params(colors="#9a9b97", labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#333637")
        axis.grid(axis="y", color="#333637", linewidth=0.7)
        axis.set_axisbelow(True)

    age_labels = list(age_values)
    age_colors = ["#71d28a" if label == age else "#ffb84d" for label in age_labels]
    axes[0].bar(age_labels, list(age_values.values()), color=age_colors)
    axes[0].set_title("Average purchase by age", color="#f8f4ed", fontsize=11, loc="left")
    axes[0].set_ylabel("Purchase amount", color="#9a9b97", fontsize=8)
    axes[0].tick_params(axis="x", rotation=35)
    if prediction is not None:
        axes[0].axhline(prediction, color="#e87932", linestyle="--", linewidth=1.5, label="Your prediction")
        axes[0].legend(facecolor="#1c1e1e", edgecolor="#333637", labelcolor="#f8f4ed", fontsize=7)

    city_labels = list(city_values)
    city_colors = ["#71d28a" if label == city else "#e87932" for label in city_labels]
    axes[1].bar(city_labels, list(city_values.values()), color=city_colors, width=0.55)
    axes[1].set_title("Average purchase by city", color="#f8f4ed", fontsize=11, loc="left")
    axes[1].set_ylabel("Purchase amount", color="#9a9b97", fontsize=8)
    for label, value in zip(city_labels, city_values.values()):
        axes[1].text(label, value + 35, f"₹{value:,}", ha="center", color="#f8f4ed", fontsize=7)

    output = BytesIO()
    figure.savefig(output, format="png", dpi=150, facecolor=figure.get_facecolor())
    plt.close(figure)
    return output.getvalue()


class RequestHandler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in {"/", "/index.html"}:
            body = (WEB_DIR / "index.html").read_bytes()
            content_type = "text/html; charset=utf-8"
        elif self.path == "/styles.css":
            body = (WEB_DIR / "styles.css").read_bytes()
            content_type = "text/css; charset=utf-8"
        elif self.path == "/script.js":
            body = (WEB_DIR / "script.js").read_bytes()
            content_type = "application/javascript; charset=utf-8"
        elif self.path == "/black-friday-hero.png":
            body = (WEB_DIR / "black-friday-hero.png").read_bytes()
            content_type = "image/png"
        elif self.path == "/api/health":
            self.send_json(200, {"status": "ok", "model": "synthetic demo model"})
            return
        elif self.path == "/api/analysis":
            self.send_json(200, build_analysis())
            return
        elif self.path == "/api/forecasting-report":
            self.send_json(200, build_forecasting_report())
            return
        elif urlparse(self.path).path == "/api/analysis-chart":
            query = parse_qs(urlparse(self.path).query)
            age = query.get("age", ["26-35"])[0]
            city = query.get("city", ["B"])[0]
            prediction_value = query.get("prediction", [None])[0]
            prediction = float(prediction_value) if prediction_value else None
            with CHART_LOCK:
                body = build_analysis_chart(age, city, prediction)
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        else:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            required = {
                "Gender",
                "Age",
                "Occupation",
                "City_Category",
                "Stay_In_Current_City_Years",
                "Marital_Status",
            }
            missing = sorted(required - payload.keys())
            if missing:
                raise ValueError(f"Missing fields: {', '.join(missing)}")
            input_row = pd.DataFrame([{key: payload[key] for key in required}])
            input_row["Occupation"] = input_row["Occupation"].astype(str)
            input_row = input_row.reindex(columns=FEATURE_COLUMNS)
            prediction = float(MODEL.predict(input_row)[0])
            self.send_json(200, {"prediction": round(max(prediction, 0), 2)})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), RequestHandler)
    print("Black Friday frontend: http://127.0.0.1:8000")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
