import streamlit as st
from dotenv import load_dotenv
import os, platform
import pandas as pd
import matplotlib.pyplot as plt

# 한글 폰트 설정 (mac/win/linux 대응)
if platform.system() == "Darwin":
    plt.rcParams['font.family'] = 'AppleGothic'
elif platform.system() == "Windows":
    plt.rcParams['font.family'] = 'Malgun Gothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

from src.embedding.embedding_configs import INDEXES
from src.prompts.prompt_modes import PROMPT_MODES
from src.search.search_utills import search_faiss_with_score, search_faiss, get_llm_answer
from src.eval.benchmark_plot import load_benchmark_csv, get_recall_top1_stats, plot_benchmark_stats

st.set_page_config(layout="wide")
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
st.title("패션 뉴스 RAG 실험 대시보드")

query = st.text_input("질문을 입력하세요:")
k = 3

tab1, tab2, tab3 = st.tabs([
    "임베딩 모델 A/B 비교", 
    "프롬프트 엔지니어링(답변 형태) 비교",
    "검색 품질 벤치마크(그래프)"
])

# ────────── 1. 임베딩 모델 A/B 비교 탭 ──────────
with tab1:
    st.markdown("##### [임베딩 모델별 검색+LLM 답변 결과 비교]")
    if query and query.strip():
        col1, spacer, col2 = st.columns([1, 0.04, 1], gap="large")
        for idx, (label, (model_name, faiss_path)) in enumerate(INDEXES.items()):
            with [col1, col2][idx]:
                st.markdown(
                    f"<div style='text-align:center;background:#333;color:#fff;padding:6px 0;margin-bottom:8px;border-radius:8px;font-weight:bold;font-size:1.13em'>{label} <span style='font-size:0.8em;'>({model_name})</span></div>",
                    unsafe_allow_html=True
                )
                results = search_faiss_with_score(model_name, faiss_path, query, k)
                ctx = ""
                for i, (doc, score) in enumerate(results, 1):
                    ctx += f"\n\n제목: {doc.metadata.get('title')}\n{doc.page_content[:300]}"
                if OPENAI_API_KEY:
                    prompt = f"""아래는 패션 뉴스 검색 결과입니다.
기사 내용 내에서만 근거를 들어 간략명확하게 답변하세요.
[질문]: {query}\n---\n{ctx}
"""
                    answer = get_llm_answer(prompt, OPENAI_API_KEY, temperature=0.2)
                    st.markdown("**LLM 답변:**", unsafe_allow_html=True)
                    st.markdown(
                        f"<div style='background:#f6f6ee;border-radius:9px;border:1.5px solid #cfcfaf;margin:6px 0 14px 0;padding:12px 24px 12px 24px;color:#222;font-size:1.11em'>{answer}</div>",
                        unsafe_allow_html=True
                    )
                else:
                    st.warning("OPENAI_API_KEY가 .env에 없으면 LLM 답변 비교가 불가능합니다.")
                st.markdown("<b>Top-K 검색 기사:</b>", unsafe_allow_html=True)
                for i, (doc, score) in enumerate(results, 1):
                    with st.container():
                        image_url = doc.metadata.get("image")
                        title = doc.metadata.get("title")
                        url = doc.metadata.get("article_id")
                        st.markdown(f"<b>{i}위 (score: {score:.4f})</b>", unsafe_allow_html=True)
                        if image_url:
                            st.image(image_url, width=180, caption=title)
                        else:
                            st.caption("이미지 없음")
                        st.markdown(
                            f"<a href='{url}' style='font-size:1.05em;color:#004FA3;font-weight:bold' target='_blank'>{title}</a>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"<span style='font-size:1.02em'>{doc.page_content[:220].strip()} ...</span>", unsafe_allow_html=True)
                        st.markdown("---")
    else:
        st.info("질문을 입력하면 임베딩 모델별 검색 결과와 LLM 답변이 나옵니다.")

# ────────── 2. 프롬프트 최적화/엔지니어링 탭 ──────────
with tab2:
    st.markdown("##### [프롬프트(답변 형태)·LLM 컨텍스트 최적화 실험]")
    pmode = st.selectbox("프롬프트/답변 스타일", list(PROMPT_MODES.keys()))
    mlabel = st.selectbox("사용 임베딩 모델", list(INDEXES.keys()))
    if query and query.strip():
        model_name, faiss_path = INDEXES[mlabel]
        docs = search_faiss(model_name, faiss_path, query, k)
        context = "\n\n".join(
            [f"[{i+1}] {d.metadata.get('title')}:\n{d.page_content[:280]}" for i, d in enumerate(docs)]
        )
        prompt = PROMPT_MODES[pmode](context, query)
        if OPENAI_API_KEY:
            answer = get_llm_answer(prompt, OPENAI_API_KEY, temperature=0.2)
            st.markdown(f"**LLM 답변 ({pmode}):**", unsafe_allow_html=True)
            st.markdown(
                f"<div style='background:#f9f6f0;border-radius:9px;border:1.5px solid #e5dfcf;margin:10px 0 20px 0;padding:15px 22px;color:#222;font-size:1.13em'>{answer}</div>",
                unsafe_allow_html=True)
        else:
            st.warning("OPENAI_API_KEY가 .env에 없으면 LLM 답변 실험이 불가능합니다.")
        st.markdown("**Top-K 뉴스 기사:**", unsafe_allow_html=True)
        for i, d in enumerate(docs, 1):
            with st.container():
                image_url = d.metadata.get("image")
                title = d.metadata.get("title")
                url = d.metadata.get("article_id")
                st.markdown(f"**{i}위**: [{title}]({url})", unsafe_allow_html=True)
                if image_url:
                    st.image(image_url, width=180, caption=title)
                else:
                    st.caption("이미지 없음")
                st.caption(d.page_content[:200].strip() + "...")
                st.markdown("---")
    else:
        st.info("탭/프롬프트/임베딩 모델을 선택하고 질문을 입력하세요.")

# ────────── 3. 검색 품질 벤치마크/그래프 탭 ──────────
with tab3:
    st.markdown("##### [임베딩별 검색 품질 평가(Benchmark)]")
    BENCHMARK_CSV = "data/ab_metrics/ab_benchmark_result.csv"
    if not os.path.exists(BENCHMARK_CSV):
        st.warning("벤치마크 결과 파일이 없습니다. 먼저 ab_benchmarker.py로 생성해 주세요.")
    else:
        df = load_benchmark_csv(BENCHMARK_CSV)
        grouped_pct = get_recall_top1_stats(df)
        st.markdown("**Recall@3 (Top-3 내 정답 포함률) / Top-1(1위 정답률) 그래프**")
        st.dataframe(grouped_pct, height=180, width=500)

        fig = plot_benchmark_stats(grouped_pct)
        st.pyplot(fig, use_container_width=False)

        st.markdown("""
        - Recall@3: Top-3 내 정답 기사 포함률(%) - IR/RAG 검색 모델의 실효 성능
        - Top-1: 1등 랭킹에 정답 포함 비율(%) - 추천결과 신뢰도
        """)