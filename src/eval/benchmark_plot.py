import pandas as pd
import matplotlib.pyplot as plt

def load_benchmark_csv(path):
    return pd.read_csv(path)

def get_recall_top1_stats(df):
    grouped = df.groupby('model').agg(
        recall3=('found', lambda g: sum(str(x).lower() == "true" for x in g) / len(g) if len(g) > 0 else 0),
        top1=('rank', lambda g: sum(int(r) == 1 for r in g if str(r).strip() not in ("", "0")) / len(g) if len(g) > 0 else 0)
    )
    return (grouped * 100).round(1).reset_index()

def plot_benchmark_stats(grouped_pct):
    fig, ax = plt.subplots(figsize=(4, 2.2))
    width = 0.35
    x = range(len(grouped_pct))
    ax.bar(x, grouped_pct["recall3"], width, label='Recall@3')
    ax.bar([i + width for i in x], grouped_pct["top1"], width, label='Top-1')
    ax.set_xticks([i + width / 2 for i in x])
    ax.set_xticklabels(grouped_pct["model"])
    ax.set_ylabel("정확도(%)")
    ax.set_title("임베딩별 검색 품질 벤치마크")
    ax.legend()
    return fig