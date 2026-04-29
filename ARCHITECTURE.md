# E-Commerce Churn Lakehouse V2 Architecture

## 1. V2 scope

This V2 build intentionally covers only the minimum thin slice required for a reliable,
traceable churn prediction system:

- one raw source domain: orders
- one Bronze ingest path
- one Silver trusted publish path with a blocking DQ gate
- one Gold feature table
- one real training pipeline
- one real FastAPI prediction endpoint
- CI, Docker, docs, runbook, and version traceability

Out of scope for V2:

- online feature store
- canary or shadow deployments
- rich drift tooling
- multi-environment cloud deployment
- full orchestrator rollout

---

## 2. Layer responsibilities

### Bronze
Bronze preserves raw evidence from the source dataset with ingest metadata and replayability.

### Silver
Silver is the trusted boundary. It contains normalized identifiers and timestamps, standardized
status values, deterministic deduplication, cleaned malformed rows, and blocking data-quality checks.

### Gold
Gold contains point-in-time-correct customer feature snapshots used for training and serving.
Gold must never include data that would not have been available at the snapshot cutoff.

---

## 3. Dataset and domain assumptions

V2 uses only the orders domain.

Primary business entities:

- `customer_id`
- `order_id`

Primary event timestamp for feature timing:

- `order_purchase_ts`

Trusted upstream source for Gold:

- Silver orders table only

V2 does **not** assume access to historical status-change events. Because of that, Gold features
must avoid any logic that depends on knowing when a status changed over time.

---

## 4. Gold grain and snapshot cadence

### Gold grain
The Gold feature table has exactly one row per:

- `customer_id`
- `as_of_date`

### Snapshot cadence
Snapshots are generated daily.

This means the model is trained and served against daily customer snapshots rather than event-level rows.

---

## 5. Exact point-in-time policy

## 5.1 Definition of `as_of_date`

`as_of_date` is the daily snapshot cutoff date for a Gold customer feature row.

For a Gold row keyed by `(customer_id, as_of_date)`:

- the snapshot represents everything known **through the end of `as_of_date`**
- the snapshot must exclude all information that occurs **after `as_of_date`**

Implementation rule for comparisons:

- derive `event_date = to_date(order_purchase_ts)`
- a record is historical for the snapshot only if `event_date <= as_of_date`

This keeps the policy deterministic and avoids mixed date/timestamp boundary ambiguity.

---

## 5.2 Data allowed before cutoff

Gold features may only use data that satisfies **all** of the following:

1. the row comes from the trusted Silver orders table
2. the row belongs to the same `customer_id`
3. `to_date(order_purchase_ts) <= as_of_date`
4. the field is safe from future leakage under the V2 data model

For V2, Gold feature logic may use:

- `customer_id`
- `order_id`
- `order_purchase_ts`
- date-derived trailing-window calculations built from `order_purchase_ts`

Examples of allowed Gold features:

- `recency_days`
- `orders_30d`
- `orders_90d`
- `lifetime_orders`
- `customer_tenure_days`
- `avg_days_between_orders`

For V2, Gold feature logic must **not** use:

- future orders
- future statuses
- delivery timestamps
- approval timestamps
- carrier timestamps
- customer-delivery timestamps
- payment outcomes
- any field whose final value may only be known after the snapshot cutoff

Rationale:
the V2 plan requires Gold to be point-in-time correct and leakage-safe, and V2 does not include
status-history reconstruction. So Gold should be built only from purchase-time history, not from
eventual downstream outcomes. :contentReference[oaicite:1]{index=1}

---

## 5.3 Feature window rule

A feature window is any historical time range used to compute Gold features.

All feature windows must end at `as_of_date`, inclusive.

Examples:

- `orders_30d` uses rows where `event_date` is in `[as_of_date - 29 days, as_of_date]`
- `orders_90d` uses rows where `event_date` is in `[as_of_date - 89 days, as_of_date]`
- `lifetime_orders` uses rows where `event_date <= as_of_date`
- `customer_tenure_days` uses the first observed historical order date through `as_of_date`
- `avg_days_between_orders` uses only order dates on or before `as_of_date`

No feature window may include rows with `event_date > as_of_date`.

---

## 6. Label-window policy

## 6.1 Label definition

The churn horizon for V2 is fixed at **60 days**.

For each `(customer_id, as_of_date)` snapshot:

- `churn_label = 1` if the customer places **no valid future order** in the next 60 days
- `churn_label = 0` if the customer places **at least one valid future order** in the next 60 days

A valid future order is an order that:

- belongs to the same `customer_id`
- has `to_date(order_purchase_ts)` in the future label window
- has final status **not** in the invalid bucket

For V2, the invalid bucket is:

- `canceled`
- `unavailable`

---

## 6.2 Exact label window

The future label window is:

- `(as_of_date, as_of_date + 60 days]`

Implementation rule:

- a row is eligible for label generation only if
  `to_date(order_purchase_ts) > as_of_date`
  and
  `to_date(order_purchase_ts) <= date_add(as_of_date, 60)`

This rule must be used only in label generation, never in Gold feature generation.

---

## 6.3 Training eligibility rule

A snapshot is eligible for training only if the full future label window is observable.

Implementation rule:

- exclude any snapshot where `date_add(as_of_date, 60) > dataset_end_date`

This prevents mislabeled training examples near the end of the dataset.

---

## 7. Separation of feature window and label window

Gold feature generation and label generation are separate steps.

### Feature step
Uses only historical data:

- `event_date <= as_of_date`

### Label step
Uses only future data:

- `event_date > as_of_date`
- `event_date <= date_add(as_of_date, 60)`

### Hard separation rule
There must be no overlap between the feature window and the label window.

That means:

- data used to compute Gold features cannot be reused from the future label window
- data used to assign `churn_label` cannot appear in the Gold feature calculation for the same snapshot
- the training dataset must be formed by joining an already-built Gold snapshot to an already-built label snapshot

This separation is mandatory for point-in-time correctness.

---

## 8. Leakage-prevention rules

The following rules are non-negotiable for V2:

### Rule 1: no future timestamps in Gold
If a row has `event_date > as_of_date`, it cannot be used in Gold features for that snapshot.

### Rule 2: no future-derived outcomes in Gold
Gold features cannot use fields that encode eventual post-purchase outcomes, including:

- final delivery outcomes
- final payment outcomes
- future status results
- any downstream timestamp that happens after purchase and may not have been known by `as_of_date`

### Rule 3: no label fields inside Gold
`churn_label`, future-order counts, and any future-retention indicators belong to label generation only.
They must never be materialized as Gold features.

### Rule 4: no overlap between feature windows and label windows
Historical feature windows end at `as_of_date`.
Future label windows begin strictly after `as_of_date`.

### Rule 5: no ad hoc training queries
Training must consume a versioned feature snapshot and a versioned label snapshot.
Do not build training data from one-off queries that mix historical and future logic in the same step.

### Rule 6: serving must mirror training semantics
Online or batch scoring for a given `as_of_date` must use the same Gold feature logic used during training.
Serving must not compute features using data beyond the intended cutoff.

---

## 9. Operational definition of Gold correctness

A Gold snapshot is considered correct only if all of the following are true:

1. one row exists per `customer_id, as_of_date`
2. every feature is computed only from data with `event_date <= as_of_date`
3. no future-derived fields are used in feature computation
4. the label window is computed separately using only future data
5. snapshots without a fully observable 60-day future window are excluded from training
6. the resulting snapshot can be reproduced for the same `as_of_date`

---

## 10. Consequences for implementation

The implementation in `src/features/customer_features_daily.py` must follow this contract exactly:

- input: trusted Silver orders
- grain: one row per `customer_id, as_of_date`
- timing rule: historical rows only, through `as_of_date`
- no use of future-derived order outcome fields as predictors
- no label logic inside feature generation

The implementation in label generation must follow this contract exactly:

- input: trusted Silver orders
- label horizon: 60 days
- future window: `(as_of_date, as_of_date + 60 days]`
- exclude snapshots beyond dataset coverage

---

## 11. Summary

V2 Gold is a daily, point-in-time-correct customer snapshot layer.

The exact timing contract is:

- features use only `event_date <= as_of_date`
- labels use only `event_date > as_of_date` and `event_date <= as_of_date + 60 days`
- feature logic and label logic must remain separate
- no future data or future-derived outcomes may leak into Gold features

This is the core policy that makes the Gold layer trustworthy for both training and serving.
