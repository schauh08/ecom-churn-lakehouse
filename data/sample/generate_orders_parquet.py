from __future__ import annotations
from datetime import datetime, timedelta
import pandas as pd

def main():
    now = datetime(2026, 1, 31, 12, 0, 0)
    rows = []
    for i in range(1, 51):
        rows.append({
            "order_id": f"o{i:04d}",
            "customer_id": f"c{(i % 10) + 1:03d}",
            "order_purchase_ts": now - timedelta(days=(i % 40)),
            "order_status": ["delivered", "shipped", "processing"][i % 3],
        })
    df = pd.DataFrame(rows)
    df.to_parquet("data/sample/orders.parquet", index=False)

if __name__ == "__main__":
    main()
