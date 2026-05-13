from flask import Flask, request, jsonify
import requests
import json
import os
from openai import OpenAI
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re

app = Flask(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

COMMUNITY_THRESHOLD = 10  # 이 수 미만이면 site: 방식으로 전환

# --- 번역 함수 ---
def translate_to_jp(query):
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "사용자의 한국어 검색어를 일본어로 번역하세요. 설명이나 따옴표 없이 번역된 결과만 출력하세요."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query

def translate_to_en(query):
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "사용자의 한국어 검색어를 영어로 번역하세요. 설명이나 따옴표 없이 번역된 결과만 출력하세요."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return query

# --- 이미지 검색 ---
def fetch_top_images(query):
    url = "https://google.serper.dev/images"
    payload = json.dumps({"q": query, "gl": "kr", "hl": "ko", "num": 3})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    image_urls = []
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        items = response.json().get('images', [])
        for item in items[:3]:
            image_urls.append(item.get('imageUrl'))
    except Exception:
        pass
    return image_urls

# --- LLM 요약 ---
def generate_core_summary(context_text):
    if not context_text:
        return "분석할 데이터가 없습니다."
    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = """
    당신은 수많은 다국어 커뮤니티 반응을 하나로 꿰뚫어 보는 전문 분석가입니다.
    제공된 검색 결과들을 종합하여 핵심 여론을 분석하세요.
    1. 원본 데이터가 외국어(영어, 일본어 등)라도 반드시 한국어로 작성할 것
    2. 사이트별 구분 없이 통합 분석
    3. 번호 매기지 않음
    4. 딱 3줄 정도로 핵심만 명확하게 작성
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"데이터:\n{context_text}"}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM 분석 에러: {e}"

# --- 날짜 파싱 ---
def parse_date(date_str):
    if not date_str:
        return datetime.min
    now = datetime.now()
    try:
        if 'ago' in date_str:
            num = int(re.search(r'\d+', date_str).group())
            if 'hour' in date_str: return now - timedelta(hours=num)
            elif 'day' in date_str: return now - timedelta(days=num)
            elif 'week' in date_str: return now - timedelta(weeks=num)
            elif 'month' in date_str: return now - timedelta(days=num * 30)
        for fmt in ("%b %d, %Y", "%Y. %m. %d.", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str.strip('. '), fmt)
            except Exception:
                continue
    except Exception:
        pass
    return datetime.min

# --- URL에서 루트 도메인 추출 ---
def extract_root_domain(url):
    try:
        netloc = urlparse(url).netloc
        parts = netloc.split('.')
        return '.'.join(parts[-2:])
    except Exception:
        return ''

# -------------------------------------------------------------------
# 방식 A: site 없이 40개 검색 후 도메인 분류 (관련도 기반 자연 가중치)
# -------------------------------------------------------------------
def fetch_by_broad_search(query, target_sites):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"

    payload = json.dumps({"q": query, "gl": "kr", "hl": "ko", "num": 40})
    try:
        res = requests.post(url, headers=headers, data=payload).json()
        all_items = res.get('organic', [])
        print(f"[방식A] 총 {len(all_items)}개 수신")
    except Exception as e:
        print(f"[방식A 에러] {e}")
        return [], {site: 0 for site in target_sites}, 0

    target_root_domains = {site: extract_root_domain(site) for site in target_sites}
    site_counts = {site: 0 for site in target_sites}
    other_count = 0
    raw_list = []

    for item in all_items:
        link = item.get('link', '')
        root_domain = extract_root_domain(link)
        matched_site = None

        for site, target_root in target_root_domains.items():
            if target_root and (target_root in root_domain or root_domain in target_root):
                matched_site = site
                break

        if matched_site:
            site_counts[matched_site] += 1
            raw_list.append({
                "site": matched_site,
                "title": item.get('title', '제목 없음'),
                "snippet": item.get('snippet', '내용 없음'),
                "link": link,
                "date": item.get('date', ''),
            })
        else:
            other_count += 1

    for site in target_sites:
        print(f"[방식A 분류] {site}: {site_counts[site]}개")
    print(f"[방식A 분류] 기타: {other_count}개")

    return raw_list, site_counts, other_count


# -------------------------------------------------------------------
# 방식 B: site: 개별 호출 (각 사이트 3개씩 균등, 최신순)
# -------------------------------------------------------------------
def fetch_by_site_search(query, target_sites):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"
    site_counts = {}
    raw_list = []

    for site in target_sites:
        payload = json.dumps({"q": f"site:{site} {query}", "gl": "kr", "hl": "ko", "num": 3})
        try:
            res = requests.post(url, headers=headers, data=payload).json()
            items = res.get('organic', [])
            site_counts[site] = len(items)
            print(f"[방식B] {site}: {len(items)}개")

            for entry in items:
                raw_list.append({
                    "site": site,
                    "title": entry.get('title', '제목 없음'),
                    "snippet": entry.get('snippet', '내용 없음'),
                    "link": entry.get('link', '#'),
                    "date": entry.get('date', ''),
                    "dt_object": parse_date(entry.get('date', ''))
                })
        except Exception as e:
            print(f"[방식B 에러] {site}: {e}")
            site_counts[site] = 0

    # 최신순 정렬
    raw_list.sort(key=lambda x: x.pop('dt_object'), reverse=True)

    return raw_list, site_counts


# -------------------------------------------------------------------
# 통합 진입점: A 시도 → 결과 부족 시 B로 전환
# -------------------------------------------------------------------
def fetch_community_data(query, target_sites):
    print(f"\n========== 검색 시작: '{query}' ==========")

    # 1단계: 방식 A 시도
    raw_list_a, site_counts_a, other_count = fetch_by_broad_search(query, target_sites)
    total_community = sum(site_counts_a.values())

    if total_community >= COMMUNITY_THRESHOLD:
        # 방식 A 결과로 충분 → 그대로 사용
        print(f"[전략] 방식A 채택 (커뮤니티 글 {total_community}개 >= 기준 {COMMUNITY_THRESHOLD}개)")
        raw_list = raw_list_a
        site_counts = site_counts_a
        site_stats = [{"site": site, "count": site_counts[site]} for site in target_sites]
        site_stats.append({"site": "기타", "count": other_count})
    else:
        # 방식 A 결과 부족 → 방식 B로 전환
        print(f"[전략] 방식B로 전환 (커뮤니티 글 {total_community}개 < 기준 {COMMUNITY_THRESHOLD}개)")
        raw_list, site_counts = fetch_by_site_search(query, target_sites)
        # 방식 B는 기타 개념 없음 (site: 로 직접 수집)
        site_stats = [{"site": site, "count": site_counts.get(site, 0)} for site in target_sites]

    # 컨텍스트 생성
    all_context = ""
    for entry in raw_list:
        all_context += f"제목: {entry['title']}\n내용: {entry['snippet']}\n\n"

    print(f"[최종] raw_list {len(raw_list)}개")
    print("========== 검색 완료 ==========\n")

    return all_context, raw_list, site_stats


@app.route('/api/search', methods=['POST'])
def search_handler():
    data = request.json
    query = data.get("query", "")
    region = data.get("region", "KR")

    if not query:
        return jsonify({"error": "검색어를 입력해주세요."}), 400

    if region == "JP":
        search_query = translate_to_jp(query)
        target_sites = ["5ch.net", "x.com", "youtube.com"]
    elif region == "US":
        search_query = translate_to_en(query)
        target_sites = ["reddit.com", "x.com", "youtube.com", "4chan.org", "quora.com"]
    else:
        search_query = query
        target_sites = ["dcinside.com", "fmkorea.com", "ruliweb.com", "theqoo.net", "arca.live"]

    images = fetch_top_images(search_query)
    collected_context, raw_list, site_stats = fetch_community_data(search_query, target_sites)
    final_report = generate_core_summary(collected_context)

    return jsonify({
        "images": images,
        "report": final_report,
        "raw_data_list": raw_list,
        "site_stats": site_stats,
        "translated_query": search_query
    })


@app.route('/api/translate', methods=['POST'])
def translate_snippet():
    data = request.json
    text_to_translate = data.get("text", "")

    if not text_to_translate:
        return jsonify({"translated_text": "번역할 텍스트가 없습니다."})

    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "다음 텍스트를 자연스러운 한국어로 번역하세요. 다른 부가 설명 없이 번역된 텍스트만 출력하세요."},
                {"role": "user", "content": text_to_translate}
            ],
            temperature=0.3
        )
        translated_text = response.choices[0].message.content.strip()
        return jsonify({"translated_text": translated_text})
    except Exception as e:
        return jsonify({"translated_text": f"번역 실패: {e}"}), 500