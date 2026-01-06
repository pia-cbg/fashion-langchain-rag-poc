import csv
from collections import Counter

csv_path = "data/ab_metrics/ab_benchmark_result.csv"
results = []
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        results.append({
            **row,
            "found": row["found"].strip().lower() in ("true","1"),
            "rank": int(row["rank"])
        })

# Recall@K, Top1@K 계산
models = sorted(set(r["model"] for r in results if r["model"]))
for m in models:
    rows = [r for r in results if r["model"]==m]
    total = len(rows)
    recall_k = sum(r["found"] for r in rows)/total
    top1 = sum(r["found"] and r["rank"]==1 for r in rows)/total
    print(f"{m} - Recall@3: {recall_k:.2%}, Top-1: {top1:.2%}")

# 잘 틀린쿼리, 모델별 found/titles/분포 등 추가보고서도 쉽게 가능