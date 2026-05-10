
## RUNBOOK.md

```md
# Runbook

## 1. Purpose

This runbook is the first place to look when a pipeline stage or API call fails.

The project is intentionally thin, so recovery is also intentionally simple: inspect the explicit artifact, identify the failing boundary, and rerun the smallest necessary stage with a fresh `run_id`.

## 2. First checks on failure

When anything fails, check these first:

    1. pipeline `run_id` or API `request_id`
    2. the most recent stage-specific artifact
    3. whether the failure is:
    - data quality
    - feature/label mismatch
    - model artifact issue
    - serving artifact issue
    - API validation/auth issue

Fast checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/version
curl http://127.0.0.1:8000/metrics
```

## 3. What happens when Silver DQ fails

Silver is the trusted publish boundary. If a critical DQ rule fails, Silver must not publish.

    Blocking checks
    order_id not null
    customer_id not null
    unique order_id
    allowed order_status
    Expected behavior

When DQ fails:

    the Silver job exits with an error
    the DQ report is written
    failed-row samples or quarantine output is written when available
    downstream Gold, labels, training snapshot, and training should not be treated as valid until Silver is fixed
    What to inspect

Check:

    DQ JSON report path
    quarantine directory
    rejected duplicate rows
    malformed or null rows from precleaning

Typical paths:

    .../silver/orders_dq_reports/<run_id>.json
    .../silver/orders_quarantine/<run_id>/
    Operator action
    inspect the DQ report and quarantine output
    determine whether the issue is source-data quality or transformation logic
    fix source data or transform code
    rerun the Silver publish with a new run_id
    only continue downstream after Silver succeeds

## 4. Retraining

Retraining should always start from a versioned training snapshot, not from ad hoc queries.

    Inputs required
    trusted Silver table
    Gold feature snapshots
    label snapshots
    assembled training snapshot
    Gold feature contract
    Retraining sequence
    build or refresh Gold features
    build or refresh labels
    build the training snapshot
    train the model with the baseline training job
    review:
    PR-AUC
    ROC-AUC
    Brier score
    MLflow run metadata
    approved model version artifact
    if acceptable, promote or update the approved model artifact used by the API
    Command template
    python -m src.training.train_stub \
    --training_snapshot_path "$LAKEHOUSE_ROOT/training/customer_training_snapshot" \
    --feature_contract data/contracts/gold/customer_features_daily.v1.json \
    --out_model_path "$ARTIFACTS_ROOT/models/ecomm_churn_baseline.pkl" \
    --out_model_meta "$ARTIFACTS_ROOT/models/model_meta.json" \
    --evaluation_summary_path "$ARTIFACTS_ROOT/models/evaluation_summary.md" \
    --approved_model_version_path "$ARTIFACTS_ROOT/models/approved_model_version.json" \
    --feature_schema_info_path "$ARTIFACTS_ROOT/models/feature_schema_info.json" \
    --model_name ecomm-churn \
    --validation_fraction 0.20 \
    --run_id "$RUN_ID-train" \
    --mlflow_tracking_uri "file:$ARTIFACTS_ROOT/mlruns" \
    --mlflow_experiment ecomm-churn-local

## 5. Rollback

The API is intentionally simple: it reads local model metadata and the approved model pointer.

Rollback goal

Restore the last known good model without rebuilding the entire stack.

    Minimum rollback steps
    identify the previous good model artifact and metadata
    confirm the feature version expected by that model
    restore or repoint:
    MODEL_PATH
    MODEL_META_PATH
    APPROVED_MODEL_VERSION_PATH
    confirm the latest-features export is compatible with the restored model’s feature_version
    call GET /ready
    run one known-good prediction request
    Rollback checks
    model_version in /version is the expected old version
    feature_version matches what the model expects
    /ready returns ready
    prediction requests succeed again

## 6. Backfill

    Backfills are allowed, but must preserve point-in-time correctness.

    Backfill rules
    do not compute Gold with future information
    preserve the fixed 60-day label horizon
    exclude training snapshots that do not have a full future observation window
    keep a new run_id for each rerun or backfill execution
    Typical backfill sequence
    ingest missing raw parquet into Bronze
    rerun Silver for the affected source range
    rerun Gold for the required as_of_date values
    rerun labels for the same dates
    rebuild training snapshot
    retrain only if the backfill materially changes the training data or deployed feature state
    rebuild latest-features export if serving data changed

## 7. Serving and API issues
/ready is not ready

Check:

    does the approved model artifact exist?
    does the model metadata file exist?
    does the latest-features parquet path exist?
    does the model’s feature_version match the feature export?
    Prediction returns 404

Cause:

    no serving features found for the requested customer id

Action:

    verify the customer exists in the latest-features export
    verify the latest-features build ran after Gold
    Prediction returns 401

Cause:

    missing or invalid X-API-Key

Action:

    confirm API_KEY in the environment
    resend request with correct header
    Prediction returns 422

Cause:

    invalid request payload

Action:

    check request schema and customer id formatting

## 8. Observability checklist

When troubleshooting, capture:

    pipeline run_id
    API request_id
    model_version
    feature_version
    data_snapshot_id if the issue relates to training
    MLflow run id if the issue relates to a trained model


## 9. Handy health checks

curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/version
curl http://127.0.0.1:8000/metrics

## 10. Recovery philosophy

This thin slice is intentionally designed so that recovery is mostly file- and metadata-driven:

    rerun a stage with a fresh run_id
    inspect explicit artifacts
    promote or restore a known-good model artifact
    avoid debugging by intuition alone
