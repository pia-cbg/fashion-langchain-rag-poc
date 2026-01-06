import json
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

DATA_PATH = "data/processed/articles_preprocessed.json"
with open(DATA_PATH, encoding='utf-8') as f:
    articles = json.load(f)

MODELS = {
    "minilm":  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e5large": "intfloat/multilingual-e5-large"
}
INDEX_BASE = "data/processed/faiss_index_"

for key, model_name in MODELS.items():
    print(f"\n==> {key} ({model_name}) 임베딩 및 인덱스 생성 시작")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    docs = [
        Document(
            page_content=a['content_for_embedding'],
            metadata={
                "article_id": a['id'],
                "title": a['title'],
                "meta": a['meta'],
                "image": a['image'],
                "snippet": a['snippet']
            }
        )
        for a in tqdm(articles)
    ]
    index_dir = INDEX_BASE + key
    os.makedirs(index_dir, exist_ok=True)
    db = FAISS.from_documents(docs, embedding=embeddings)
    db.save_local(index_dir)
    print(f"{key} 인덱스 저장 완료 : {index_dir}")

print("\n★★ 모든 모델 인덱싱 완료! ★★")