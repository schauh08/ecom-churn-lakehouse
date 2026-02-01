from __future__ import annotations
import argparse, json
import mlflow
from src.common.versioning import hash_contract_json

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="ecomm-churn")
    p.add_argument("--feature_contract", required=True)
    p.add_argument("--out_model_meta", required=True)
    args = p.parse_args()

    feature_version = hash_contract_json(args.feature_contract)

    mlflow.set_experiment("ecomm-churn-slice")
    with mlflow.start_run() as run:
        mlflow.log_param("feature_version", feature_version)
        mlflow.log_metric("auc", 0.50)

        model_version = run.info.run_id[:12]
        meta = {
            "model_name": args.model_name,
            "model_version": model_version,
            "feature_version": feature_version
        }
        with open(args.out_model_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f)

if __name__ == "__main__":
    main()
