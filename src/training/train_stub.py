from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
from pyspark.sql import functions as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.versioning import hash_contract_json
from src.common.spark import get_spark
from src.common.pipeline_logging import get_pipeline_logger, log_pipeline_event



FEATURE_COLUMNS = [
    "recency_days",
    "orders_30d",
    "orders_90d",
    "lifetime_orders",
    "customer_tenure_days",
    "avg_days_between_orders",
]


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash_obj(obj: Any) -> str:
    return hashlib.sha256(_canonical_json(obj)).hexdigest()[:16]


def _single_distinct_value(df, column_name: str) -> str:
    values = [row[0] for row in df.select(column_name).distinct().collect()]
    if len(values) != 1:
        raise RuntimeError(
            f"Expected exactly one distinct {column_name}, found {values}"
        )
    return str(values[0])


def _write_json(path: str, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: str, content: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")


def _write_pickle(path: str, obj: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(obj, f)


def _git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        value = proc.stdout.strip()
        return value or None
    except Exception:
        return None


def _schema_hash_from_columns(df) -> str:
    schema_obj = json.loads(df.schema.json())
    return _hash_obj(schema_obj)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_snapshot_path", required=True)
    parser.add_argument("--feature_contract", required=True)

    parser.add_argument("--out_model_path", required=True)
    parser.add_argument("--out_model_meta", required=True)
    parser.add_argument("--evaluation_summary_path", required=True)
    parser.add_argument("--approved_model_version_path", required=True)
    parser.add_argument("--feature_schema_info_path", required=True)

    parser.add_argument("--model_name", default="ecomm-churn")
    parser.add_argument("--validation_fraction", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--run_id", required=True)

    parser.add_argument("--mlflow_tracking_uri", default=None)
    parser.add_argument("--mlflow_experiment", default="ecomm-churn-slice")

    args = parser.parse_args()

    logger = get_pipeline_logger("pipeline.train_baseline_logreg")
    log_pipeline_event(
        logger,
        "started",
        run_id=args.run_id,
        training_snapshot_path=args.training_snapshot_path,
        model_name=args.model_name,
        validation_fraction=args.validation_fraction,
        random_state=args.random_state,
        mlflow_experiment=args.mlflow_experiment,
    )

    try:
        if not (0.0 < args.validation_fraction < 1.0):
            raise ValueError("--validation_fraction must be between 0 and 1.")

        if args.mlflow_tracking_uri:
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)

        mlflow.set_experiment(args.mlflow_experiment)

        spark = get_spark("train_baseline_logreg")

        snapshot = (
            spark.read.format("delta").load(args.training_snapshot_path)
            .select(
                "customer_id",
                "as_of_date",
                *FEATURE_COLUMNS,
                "churn_label",
                "_data_snapshot_id",
                "_feature_version",
                "_label_version",
            )
            .orderBy("as_of_date", "customer_id")
        )

        row_count = snapshot.count()
        if row_count == 0:
            raise RuntimeError("Training snapshot is empty; cannot train model.")

        contract_feature_version = hash_contract_json(args.feature_contract)
        snapshot_feature_version = _single_distinct_value(snapshot, "_feature_version")
        data_snapshot_id = _single_distinct_value(snapshot, "_data_snapshot_id")
        label_version = _single_distinct_value(snapshot, "_label_version")
        payload_schema_hash = _schema_hash_from_columns(snapshot)
        git_sha = _git_sha()

        if contract_feature_version != snapshot_feature_version:
            raise RuntimeError(
                "Feature contract hash does not match training snapshot feature version. "
                f"contract={contract_feature_version}, snapshot={snapshot_feature_version}"
            )

        feature_schema_info = {
            "feature_columns": FEATURE_COLUMNS,
            "feature_version": snapshot_feature_version,
            "feature_contract_hash": contract_feature_version,
            "training_snapshot_schema_hash": payload_schema_hash,
            "label_version": label_version,
            "data_snapshot_id": data_snapshot_id,
        }
        _write_json(args.feature_schema_info_path, feature_schema_info)

        pdf = snapshot.toPandas()
        if pdf.empty:
            raise RuntimeError("Training snapshot converted to pandas is empty.")

        unique_dates = sorted(pdf["as_of_date"].drop_duplicates().tolist())
        if len(unique_dates) < 2:
            raise RuntimeError(
                "Need at least 2 distinct as_of_date values for a time-based split."
            )

        valid_n_dates = max(1, math.ceil(len(unique_dates) * args.validation_fraction))
        valid_n_dates = min(valid_n_dates, len(unique_dates) - 1)

        train_dates = unique_dates[:-valid_n_dates]
        valid_dates = unique_dates[-valid_n_dates:]

        train_df = pdf[pdf["as_of_date"].isin(train_dates)].copy()
        valid_df = pdf[pdf["as_of_date"].isin(valid_dates)].copy()

        if train_df.empty or valid_df.empty:
            raise RuntimeError(
                "Time-based split produced an empty train or validation set."
            )

        if train_df["churn_label"].nunique() < 2:
            raise RuntimeError(
                "Training split contains only one class. Widen the as_of_date range."
            )

        if valid_df["churn_label"].nunique() < 2:
            raise RuntimeError(
                "Validation split contains only one class, so ROC-AUC cannot be computed. "
                "Widen the as_of_date range."
            )

        X_train = train_df[FEATURE_COLUMNS]
        y_train = train_df["churn_label"].astype(int)

        X_valid = valid_df[FEATURE_COLUMNS]
        y_valid = valid_df["churn_label"].astype(int)

        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=args.random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        )

        with mlflow.start_run(run_name=f"{args.model_name}-{data_snapshot_id}") as run:
            model.fit(X_train, y_train)

            valid_proba = model.predict_proba(X_valid)[:, 1]
            pr_auc = float(average_precision_score(y_valid, valid_proba))
            roc_auc = float(roc_auc_score(y_valid, valid_proba))
            brier = float(brier_score_loss(y_valid, valid_proba))

            model_version = _hash_obj(
                {
                    "model_name": args.model_name,
                    "algorithm": "logistic_regression",
                    "data_snapshot_id": data_snapshot_id,
                    "feature_version": snapshot_feature_version,
                    "label_version": label_version,
                    "validation_fraction": args.validation_fraction,
                    "random_state": args.random_state,
                    "version": 2,
                }
            )

            approved_payload = {
                "approved_model_version": model_version,
                "model_name": args.model_name,
                "data_snapshot_id": data_snapshot_id,
                "feature_version": snapshot_feature_version,
                "label_version": label_version,
                "mlflow_run_id": run.info.run_id,
            }
            _write_json(args.approved_model_version_path, approved_payload)

            model_bundle = {
                "model_name": args.model_name,
                "model_version": model_version,
                "algorithm": "logistic_regression",
                "feature_columns": FEATURE_COLUMNS,
                "feature_version": snapshot_feature_version,
                "label_version": label_version,
                "data_snapshot_id": data_snapshot_id,
                "training_snapshot_schema_hash": payload_schema_hash,
                "mlflow_run_id": run.info.run_id,
                "pipeline": model,
            }
            _write_pickle(args.out_model_path, model_bundle)

            meta = {
                "model_name": args.model_name,
                "model_version": model_version,
                "approved_model_version": model_version,
                "algorithm": "logistic_regression",
                "feature_columns": FEATURE_COLUMNS,
                "feature_version": snapshot_feature_version,
                "feature_contract_hash": contract_feature_version,
                "label_version": label_version,
                "data_snapshot_id": data_snapshot_id,
                "training_snapshot_path": args.training_snapshot_path,
                "training_snapshot_schema_hash": payload_schema_hash,
                "validation_fraction": args.validation_fraction,
                "random_state": args.random_state,
                "train_row_count": int(len(train_df)),
                "validation_row_count": int(len(valid_df)),
                "train_as_of_date_min": str(min(train_dates)),
                "train_as_of_date_max": str(max(train_dates)),
                "validation_as_of_date_min": str(min(valid_dates)),
                "validation_as_of_date_max": str(max(valid_dates)),
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_valid.mean()),
                "code_version": git_sha,
                "mlflow_tracking_uri": mlflow.get_tracking_uri(),
                "mlflow_experiment": args.mlflow_experiment,
                "mlflow_run_id": run.info.run_id,
                "metrics": {
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    "brier_score": brier,
                },
            }
            _write_json(args.out_model_meta, meta)

            summary = f"""# Baseline Logistic Regression Evaluation

Model name: {args.model_name}
Model version: {model_version}
Approved model version: {model_version}
Algorithm: logistic_regression
Data snapshot id: {data_snapshot_id}
Feature version: {snapshot_feature_version}
Label version: {label_version}
Training snapshot schema hash: {payload_schema_hash}
Code version (git SHA): {git_sha}
MLflow experiment: {args.mlflow_experiment}
MLflow run id: {run.info.run_id}

## Split
- Train as_of_date range: {min(train_dates)} to {max(train_dates)}
- Validation as_of_date range: {min(valid_dates)} to {max(valid_dates)}
- Train rows: {len(train_df)}
- Validation rows: {len(valid_df)}
- Train positive rate: {y_train.mean():.4f}
- Validation positive rate: {y_valid.mean():.4f}

## Metrics
- PR-AUC: {pr_auc:.6f}
- ROC-AUC: {roc_auc:.6f}
- Brier score: {brier:.6f}

## Features
- recency_days
- orders_30d
- orders_90d
- lifetime_orders
- customer_tenure_days
- avg_days_between_orders
"""
            _write_text(args.evaluation_summary_path, summary)

            mlflow.set_tags(
                {
                    "model_name": args.model_name,
                    "algorithm": "logistic_regression",
                    "model_version": model_version,
                    "approved_model_version": model_version,
                    "data_snapshot_id": data_snapshot_id,
                    "feature_version": snapshot_feature_version,
                    "label_version": label_version,
                    "training_snapshot_schema_hash": payload_schema_hash,
                    "code_version": git_sha or "unknown",
                }
            )

            mlflow.log_params(
                {
                    "model_name": args.model_name,
                    "algorithm": "logistic_regression",
                    "validation_fraction": args.validation_fraction,
                    "random_state": args.random_state,
                    "train_row_count": int(len(train_df)),
                    "validation_row_count": int(len(valid_df)),
                    "feature_version": snapshot_feature_version,
                    "feature_contract_hash": contract_feature_version,
                    "label_version": label_version,
                    "data_snapshot_id": data_snapshot_id,
                    "training_snapshot_schema_hash": payload_schema_hash,
                    "code_version": git_sha or "unknown",
                }
            )

            mlflow.log_metrics(
                {
                    "pr_auc": pr_auc,
                    "roc_auc": roc_auc,
                    "brier_score": brier,
                    "train_positive_rate": float(y_train.mean()),
                    "validation_positive_rate": float(y_valid.mean()),
                }
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )

            mlflow.log_artifact(args.out_model_path, artifact_path="artifacts")
            mlflow.log_artifact(args.out_model_meta, artifact_path="artifacts")
            mlflow.log_artifact(args.evaluation_summary_path, artifact_path="artifacts")
            mlflow.log_artifact(args.approved_model_version_path, artifact_path="artifacts")
            mlflow.log_artifact(args.feature_schema_info_path, artifact_path="artifacts")

            log_pipeline_event(
                logger,
                "completed",
                run_id=args.run_id,
                model_name=args.model_name,
                model_version=model_version,
                mlflow_run_id=run.info.run_id,
                data_snapshot_id=data_snapshot_id,
                feature_version=snapshot_feature_version,
                label_version=label_version,
                training_snapshot_path=args.training_snapshot_path,
                train_row_count=int(len(train_df)),
                validation_row_count=int(len(valid_df)),
                pr_auc=pr_auc,
                roc_auc=roc_auc,
                brier_score=brier,
                out_model_path=args.out_model_path,
                out_model_meta=args.out_model_meta,
            )

    except Exception as exc:
        log_pipeline_event(
            logger,
            "failed",
            run_id=args.run_id,
            training_snapshot_path=args.training_snapshot_path,
            model_name=args.model_name,
            error=str(exc),
        )
        raise
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_snapshot_path", required=True)
    parser.add_argument("--feature_contract", required=True)

    parser.add_argument("--out_model_path", required=True)
    parser.add_argument("--out_model_meta", required=True)
    parser.add_argument("--evaluation_summary_path", required=True)
    parser.add_argument("--approved_model_version_path", required=True)
    parser.add_argument("--feature_schema_info_path", required=True)

    parser.add_argument("--model_name", default="ecomm-churn")
    parser.add_argument("--validation_fraction", type=float, default=0.20)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--mlflow_tracking_uri", default=None)
    parser.add_argument("--mlflow_experiment", default="ecomm-churn-slice")

    args = parser.parse_args()

    if not (0.0 < args.validation_fraction < 1.0):
        raise ValueError("--validation_fraction must be between 0 and 1.")

    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    mlflow.set_experiment(args.mlflow_experiment)

    spark = get_spark("train_baseline_logreg")

    snapshot = (
        spark.read.format("delta").load(args.training_snapshot_path)
        .select(
            "customer_id",
            "as_of_date",
            *FEATURE_COLUMNS,
            "churn_label",
            "_data_snapshot_id",
            "_feature_version",
            "_label_version",
        )
        .orderBy("as_of_date", "customer_id")
    )

    row_count = snapshot.count()
    if row_count == 0:
        raise RuntimeError("Training snapshot is empty; cannot train model.")

    contract_feature_version = hash_contract_json(args.feature_contract)
    snapshot_feature_version = _single_distinct_value(snapshot, "_feature_version")
    data_snapshot_id = _single_distinct_value(snapshot, "_data_snapshot_id")
    label_version = _single_distinct_value(snapshot, "_label_version")
    payload_schema_hash = _schema_hash_from_columns(snapshot)
    git_sha = _git_sha()

    if contract_feature_version != snapshot_feature_version:
        raise RuntimeError(
            "Feature contract hash does not match training snapshot feature version. "
            f"contract={contract_feature_version}, snapshot={snapshot_feature_version}"
        )

    feature_schema_info = {
        "feature_columns": FEATURE_COLUMNS,
        "feature_version": snapshot_feature_version,
        "feature_contract_hash": contract_feature_version,
        "training_snapshot_schema_hash": payload_schema_hash,
        "label_version": label_version,
        "data_snapshot_id": data_snapshot_id,
    }
    _write_json(args.feature_schema_info_path, feature_schema_info)

    pdf = snapshot.toPandas()
    if pdf.empty:
        raise RuntimeError("Training snapshot converted to pandas is empty.")

    unique_dates = sorted(pdf["as_of_date"].drop_duplicates().tolist())
    if len(unique_dates) < 2:
        raise RuntimeError(
            "Need at least 2 distinct as_of_date values for a time-based split."
        )

    valid_n_dates = max(1, math.ceil(len(unique_dates) * args.validation_fraction))
    valid_n_dates = min(valid_n_dates, len(unique_dates) - 1)

    train_dates = unique_dates[:-valid_n_dates]
    valid_dates = unique_dates[-valid_n_dates:]

    train_df = pdf[pdf["as_of_date"].isin(train_dates)].copy()
    valid_df = pdf[pdf["as_of_date"].isin(valid_dates)].copy()

    if train_df.empty or valid_df.empty:
        raise RuntimeError(
            "Time-based split produced an empty train or validation set."
        )

    if train_df["churn_label"].nunique() < 2:
        raise RuntimeError(
            "Training split contains only one class. Widen the as_of_date range."
        )

    if valid_df["churn_label"].nunique() < 2:
        raise RuntimeError(
            "Validation split contains only one class, so ROC-AUC cannot be computed. "
            "Widen the as_of_date range."
        )

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["churn_label"].astype(int)

    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df["churn_label"].astype(int)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=args.random_state,
                    solver="liblinear",
                ),
            ),
        ]
    )

    with mlflow.start_run(run_name=f"{args.model_name}-{data_snapshot_id}") as run:
        model.fit(X_train, y_train)

        valid_proba = model.predict_proba(X_valid)[:, 1]
        pr_auc = float(average_precision_score(y_valid, valid_proba))
        roc_auc = float(roc_auc_score(y_valid, valid_proba))
        brier = float(brier_score_loss(y_valid, valid_proba))

        model_version = _hash_obj(
            {
                "model_name": args.model_name,
                "algorithm": "logistic_regression",
                "data_snapshot_id": data_snapshot_id,
                "feature_version": snapshot_feature_version,
                "label_version": label_version,
                "validation_fraction": args.validation_fraction,
                "random_state": args.random_state,
                "version": 2,
            }
        )

        approved_payload = {
            "approved_model_version": model_version,
            "model_name": args.model_name,
            "data_snapshot_id": data_snapshot_id,
            "feature_version": snapshot_feature_version,
            "label_version": label_version,
            "mlflow_run_id": run.info.run_id,
        }
        _write_json(args.approved_model_version_path, approved_payload)

        model_bundle = {
            "model_name": args.model_name,
            "model_version": model_version,
            "algorithm": "logistic_regression",
            "feature_columns": FEATURE_COLUMNS,
            "feature_version": snapshot_feature_version,
            "label_version": label_version,
            "data_snapshot_id": data_snapshot_id,
            "training_snapshot_schema_hash": payload_schema_hash,
            "mlflow_run_id": run.info.run_id,
            "pipeline": model,
        }
        _write_pickle(args.out_model_path, model_bundle)

        meta = {
            "model_name": args.model_name,
            "model_version": model_version,
            "approved_model_version": model_version,
            "algorithm": "logistic_regression",
            "feature_columns": FEATURE_COLUMNS,
            "feature_version": snapshot_feature_version,
            "feature_contract_hash": contract_feature_version,
            "label_version": label_version,
            "data_snapshot_id": data_snapshot_id,
            "training_snapshot_path": args.training_snapshot_path,
            "training_snapshot_schema_hash": payload_schema_hash,
            "validation_fraction": args.validation_fraction,
            "random_state": args.random_state,
            "train_row_count": int(len(train_df)),
            "validation_row_count": int(len(valid_df)),
            "train_as_of_date_min": str(min(train_dates)),
            "train_as_of_date_max": str(max(train_dates)),
            "validation_as_of_date_min": str(min(valid_dates)),
            "validation_as_of_date_max": str(max(valid_dates)),
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_valid.mean()),
            "code_version": git_sha,
            "mlflow_tracking_uri": mlflow.get_tracking_uri(),
            "mlflow_experiment": args.mlflow_experiment,
            "mlflow_run_id": run.info.run_id,
            "metrics": {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier_score": brier,
            },
        }
        _write_json(args.out_model_meta, meta)

        summary = f"""# Baseline Logistic Regression Evaluation

Model name: {args.model_name}
Model version: {model_version}
Approved model version: {model_version}
Algorithm: logistic_regression
Data snapshot id: {data_snapshot_id}
Feature version: {snapshot_feature_version}
Label version: {label_version}
Training snapshot schema hash: {payload_schema_hash}
Code version (git SHA): {git_sha}
MLflow experiment: {args.mlflow_experiment}
MLflow run id: {run.info.run_id}

## Split
- Train as_of_date range: {min(train_dates)} to {max(train_dates)}
- Validation as_of_date range: {min(valid_dates)} to {max(valid_dates)}
- Train rows: {len(train_df)}
- Validation rows: {len(valid_df)}
- Train positive rate: {y_train.mean():.4f}
- Validation positive rate: {y_valid.mean():.4f}

## Metrics
- PR-AUC: {pr_auc:.6f}
- ROC-AUC: {roc_auc:.6f}
- Brier score: {brier:.6f}

## Features
- recency_days
- orders_30d
- orders_90d
- lifetime_orders
- customer_tenure_days
- avg_days_between_orders
"""
        _write_text(args.evaluation_summary_path, summary)

        mlflow.set_tags(
            {
                "model_name": args.model_name,
                "algorithm": "logistic_regression",
                "model_version": model_version,
                "approved_model_version": model_version,
                "data_snapshot_id": data_snapshot_id,
                "feature_version": snapshot_feature_version,
                "label_version": label_version,
                "training_snapshot_schema_hash": payload_schema_hash,
                "code_version": git_sha or "unknown",
            }
        )

        mlflow.log_params(
            {
                "model_name": args.model_name,
                "algorithm": "logistic_regression",
                "validation_fraction": args.validation_fraction,
                "random_state": args.random_state,
                "train_row_count": int(len(train_df)),
                "validation_row_count": int(len(valid_df)),
                "feature_version": snapshot_feature_version,
                "feature_contract_hash": contract_feature_version,
                "label_version": label_version,
                "data_snapshot_id": data_snapshot_id,
                "training_snapshot_schema_hash": payload_schema_hash,
                "code_version": git_sha or "unknown",
            }
        )

        mlflow.log_metrics(
            {
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "brier_score": brier,
                "train_positive_rate": float(y_train.mean()),
                "validation_positive_rate": float(y_valid.mean()),
            }
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        mlflow.log_artifact(args.out_model_path, artifact_path="artifacts")
        mlflow.log_artifact(args.out_model_meta, artifact_path="artifacts")
        mlflow.log_artifact(args.evaluation_summary_path, artifact_path="artifacts")
        mlflow.log_artifact(args.approved_model_version_path, artifact_path="artifacts")
        mlflow.log_artifact(args.feature_schema_info_path, artifact_path="artifacts")


if __name__ == "__main__":
    main()