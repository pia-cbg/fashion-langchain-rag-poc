import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI  # pip install langchain-openai


# 1. 환경변수 자동 로드(.env → os.environ)
load_dotenv()  # pip install python-dotenv
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

INDEX_PATH = 'data/processed/faiss_index'

embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
db = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

query = input("질문을 입력하세요: ")
k = 3
docs = db.similarity_search(query, k=k)
context = "\n\n".join([f"제목: {d.metadata.get('title')}\n{d.page_content}" for d in docs])

# 프롬프트 예시(질문, 검색문서 컨텍스트 포함)
prompt = f"""
아래 기사는 패션 트렌드 관련 뉴스기사다.
주어진 정보 내에서만 사용하여, 사용자의 질의에 대해 객관적이고 명확하며 핵심을 잘 요약해 답변하라.

질문: {query}
---
검색된 기사들:
{context}
"""

# OpenAI LLM 예시 (openai api-key 필요)
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model="gpt-3.5-turbo",            # 필요시 gpt-4 등으로 변경
    temperature=0.2
)

response = llm.invoke(prompt)
print("\n[답변]:\n" + response.content)