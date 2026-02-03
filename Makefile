RUN_ID := $(shell python -c "import uuid; print(uuid.uuid4())")

LAKEHOUSE_ROOT ?= lakehouse
ARTIFACTS_ROOT ?= artifacts
INPUT_ORDERS ?= data/sample/orders.parquet
AS_OF_DATE ?= 2026-01-31

slice:
	mkdir -p lakehouse/bronze/orders lakehouse/silver/orders lakehouse/gold/customer_features_daily artifacts

	python -m src.ingestion.orders_to_bronze \
	  --input data/sample/orders.parquet \
	  --bronze_path lakehouse/bronze/orders \
	  --run_id $(RUN_ID)

	python -m src.transformations.orders_bronze_to_silver \
	  --bronze_path lakehouse/bronze/orders \
	  --silver_path lakehouse/silver/orders \
	  --contract data/contracts/silver/orders.v1.json \
	  --expectations data/expectations/silver/orders.yml \
	  --run_id $(RUN_ID)

	python -m src.features.customer_features_daily \
	  --silver_path lakehouse/silver/orders \
	  --gold_path lakehouse/gold/customer_features_daily \
	  --contract data/contracts/gold/customer_features_daily.v1.json \
	  --as_of_date 2026-01-31 \
	  --run_id $(RUN_ID)

	python -m src.training.train_stub \
	  --feature_contract data/contracts/gold/customer_features_daily.v1.json \
	  --out_model_meta artifacts/model_meta.json

