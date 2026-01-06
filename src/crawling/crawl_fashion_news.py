import os
import json
import time
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.kfashionnews.com"
LIST_BASE_URL = BASE_URL + "/news/bbs/board.php?bo_table=knews&sca=%ED%8C%A8%EC%85%98&page={page}"

def extract_article_content(url):
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        s = BeautifulSoup(resp.text, "html.parser")
        tags = s.select('#view-content.view-content')
        if tags:
            return tags[0].get_text(separator='\n', strip=True)
    except Exception as e:
        print(f"본문 크롤 실패: {url} ({e})")
    return ''

def crawl_all_pages(start=1, end=10):
    results = []
    for page in range(start, end + 1):
        print(f"\n[{page} 페이지 크롤링 중...]")
        list_url = LIST_BASE_URL.format(page=page)
        resp = requests.get(list_url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(resp.text, "html.parser")
        rows = soup.select("table.list-pc > tbody > tr")
        for row in rows:
            img_tag = row.select_one("td:nth-of-type(2) img")
            image_url = img_tag['src'] if img_tag else None
            subj_td = row.select_one("td.list-subject")
            if not subj_td: continue
            a_tag = subj_td.select_one("span.zTitle > a")
            if not a_tag: continue
            title = a_tag.get_text(strip=True)
            article_url = a_tag['href']
            if article_url.startswith("/"):
                article_url = BASE_URL + article_url
            # snippet
            snippet = ""
            for sibling in a_tag.parent.next_siblings:
                if getattr(sibling, 'name', None) == "span" and "zCat" in sibling.get("class", []):
                    break
                if isinstance(sibling, str):
                    txt = sibling.strip()
                    if txt:
                        snippet += txt + " "
                elif getattr(sibling, 'name', None) == 'br':
                    continue
                elif sibling.get_text(strip=True):
                    snippet += sibling.get_text(strip=True) + " "
            snippet = snippet.strip()
            # 카테고리/날짜/기자
            cat_tag = subj_td.select_one("span.zCat span._text-muted")
            category = cat_tag.get_text(" ", strip=True) if cat_tag else ""

            print(f"크롤링: {title}")
            article_body = extract_article_content(article_url)
            time.sleep(0.5)  # 사이트 보호
            results.append({
                "title": title,
                "url": article_url,
                "image": image_url,
                "snippet": snippet,
                "meta": category,
                "body": article_body
            })
    return results

def main():
    results = crawl_all_pages(1, 10)
    os.makedirs("data/raw", exist_ok=True)
    out_path = "data/raw/articles_full.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n총 {len(results)}개 기사 크롤 & 저장 완료 → {out_path}")

if __name__ == "__main__":
    main()