from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

INDEX_PATH = 'data/processed/faiss_index'

query = input('검색어를 입력하세요: ')
k = 5  # top k

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

# 검색
docs_and_scores = db.similarity_search_with_score(query, k=k)
for i, (doc, score) in enumerate(docs_and_scores, 1):
    print(f'\n==={i}위 (score: {score:.4f})===')
    md = doc.metadata
    print(f"[{md['title']}]")
    print(md.get('meta', ''))
    print('-' * 40)
    print(doc.page_content[:500], "...")
    print(f"기사링크: {md.get('article_id')}")