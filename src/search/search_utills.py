from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

def search_faiss_with_score(model_name, faiss_path, query, k=3):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return db.similarity_search_with_score(query, k=k)

def search_faiss(model_name, faiss_path, query, k=3):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
    return db.similarity_search(query, k=k)

def get_llm_answer(prompt, openai_api_key, temperature=0.2):
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=temperature)
    return llm.invoke(prompt).content