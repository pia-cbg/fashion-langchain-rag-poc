import json
import csv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

INDEXES = {
    "minilm":  ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "data/processed/faiss_index_minilm"),
    "e5large": ("intfloat/multilingual-e5-large", "data/processed/faiss_index_e5large"),
}
K = 3

# 골든셋 로드
with open("data/ab_metrics/golden_queries.csv") as f:
    reader = csv.DictReader(f)
    golden = [row for row in reader]

# 평가 결과 저장
results = []

for label, (model_name, faiss_path) in INDEXES.items():
    print(f"\n[모델: {label} 평가 시작]")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    for row in golden:
        q, ans = row['query'].strip(), row['answer_title'].strip()
        found = False
        docs = db.similarity_search(q, k=K)
        titles = [d.metadata["title"].strip() for d in docs]
        for rank, t in enumerate(titles, 1):
            if ans in t or t in ans:
                found = True
                found_rank = rank
                break
        results.append({
            "query": q,
            "answer_title": ans,
            "model": label,
            "found": found,
            "rank": found_rank if found else 0,
            "titles": "; ".join(titles)
        })

# csv 출력
with open("data/ab_metrics/ab_benchmark_result.csv", "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# 간단 통계
tot = len(golden)
for label in INDEXES:
    ok = sum(r["found"] for r in results if r["model"] == label)
    print(f"{label}: Recall@{K} = {ok}/{tot} ({ok/tot:.2%})")