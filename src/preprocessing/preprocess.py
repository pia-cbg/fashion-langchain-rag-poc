import json, os, re

IN_PATH = 'data/raw/articles_full.json'
OUT_PATH = 'data/processed/articles_preprocessed.json'
os.makedirs('data/processed', exist_ok=True)

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text or "")
    text = re.sub(r'\s+', ' ', text or "")
    return text.strip()

with open(IN_PATH, encoding='utf-8') as f:
    data = json.load(f)

processed = []
for d in data:
    body = clean_text(d.get("body", ""))
    title = clean_text(d.get("title", ""))
    if not body or len(body) < 30:
        continue
    content = f"{title}\n{body}"
    out = {
        "id": d.get("url", ""),
        "title": title,
        "body": body,
        "meta": d.get("meta", ""),
        "snippet": d.get("snippet", ""),
        "image": d.get("image", ""),
        "content_for_embedding": content
    }
    processed.append(out)

with open(OUT_PATH, 'w', encoding='utf-8') as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)

print(f"전처리 완료 → {OUT_PATH} ({len(processed)}개 기사)")