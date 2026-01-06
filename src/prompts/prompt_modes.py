PROMPT_MODES = {
    "기본요약": lambda ctx, q: f"질문: {q}\n기사: {ctx}\n답만 깔끔하게 요약해줘.",
    "출처 포함(번호)": lambda ctx, q: f"질문: {q}\n기사: {ctx}\n답변 마지막에 기사 제목/번호를 넣어줘.",
    "핵심포인트 나열": lambda ctx, q: f"질문: {q}\n기사에서 답을 ● 리스트로 정리해줘.\n{ctx}",
    "장문 상세답변": lambda ctx, q: f"질문: {q}\n기사만 참고해서 길고 자세한 답변 작성.\n{ctx}"
}