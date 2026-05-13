from flask import Flask, request, jsonify
import requests
import json
import os
from openai import OpenAI
from datetime import datetime, timedelta
import re

app = Flask(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
    1. 원본 데이터가 외국어라도 반드시 한국어로 작성할 것. 2. 통합 분석. 3. 번호 생략. 4. 딱 3줄 요약.
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
    if not date_str: return datetime.min
    now = datetime.now()
    try:
        if 'ago' in date_str:
            num = int(re.search(r'\d+', date_str).group())
            if 'hour' in date_str: return now - timedelta(hours=num)
            elif 'day' in date_str: return now - timedelta(days=num)
            elif 'week' in date_str: return now - timedelta(weeks=num)
            elif 'month' in date_str: return now - timedelta(days=num * 30)
        for fmt in ("%b %d, %Y", "%Y. %m. %d.", "%Y-%m-%d"):
            try: return datetime.strptime(date_str.strip('. '), fmt)
            except: continue
    except: pass
    return datetime.min

# --- [수정 핵심] 언급량 비율에 따른 가중치 배분 로직 ---
def fetch_community_data_weighted(query, sites):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"
    TOTAL_TARGET = 15

    site_buckets = []
    total_weight = 0

    print(f"\n========== 검색 및 가중치 분석 시작: '{query}' ==========")

    for site in sites:
        payload = json.dumps({"q": f"site:{site} {query}", "gl": "kr", "hl": "ko", "num": 10})
        try:
            res = requests.post(url, headers=headers, data=payload).json()
            items = res.get('organic', [])
            
            # 1. 구글이 추정한 전체 결과 수 가져오기 (콤마 제거)
            raw_total = res.get('searchInformation', {}).get('totalResults', '0')
            google_estimate = int(str(raw_total).replace(',', ''))
            
            # 2. [가중치 결정 로직]
            # 구글이 숫자를 줬다면 그 숫자를 쓰고, 0이라고 줬는데 글이 검색됐다면 최소 가중치 100 부여
            weight = google_estimate
            if weight == 0 and len(items) > 0:
                weight = 100 

            site_buckets.append({"site": site, "weight": weight, "items": items})
            total_weight += weight
            print(f"[데이터 수집] {site} -> 추정 언급량: {weight}, 가져온 샘플: {len(items)}개")
        except Exception as e:
            print(f"[에러] {site}: {e}")
            site_buckets.append({"site": site, "weight": 0, "items": []})

    raw_list = []
    for bucket in site_buckets:
        site = bucket["site"]
        weight = bucket["weight"]
        items = bucket["items"]

        # 3. 전체 가중치 대비 비율로 '최종 노출 개수' 결정
        if total_weight > 0 and weight > 0:
            # 최소 1개, 최대 8개로 제한하여 다양성과 비중을 모두 잡음
            keep = max(1, min(8, round((weight / total_weight) * TOTAL_TARGET)))
        else:
            keep = max(1, TOTAL_TARGET // len(sites))

        print(f"[최종 배분] {site}: 가중치 {weight}에 따라 {keep}개 채택")

        for entry in items[:keep]:
            raw_list.append({
                "site": site,
                "title": entry.get('title', '제목 없음'),
                "snippet": entry.get('snippet', '내용 없음'),
                "link": entry.get('link', '#'),
                "date": entry.get('date', ''),
                "dt_object": parse_date(entry.get('date', ''))
            })

    # 최신순 정렬
    raw_list.sort(key=lambda x: x['dt_object'], reverse=True)

    all_context = ""
    for entry in raw_list:
        all_context += f"제목: {entry['title']}\n내용: {entry['snippet']}\n\n"
        del entry['dt_object']

    site_stats = [{"site": b["site"], "count": b["weight"]} for b in site_buckets]
    return all_context, raw_list, site_stats

@app.route('/api/search', methods=['POST'])
def search_handler():
    data = request.json
    query, region = data.get("query", ""), data.get("region", "KR")
    if not query: return jsonify({"error": "검색어 필수"}), 400
        
    if region == "JP":
        search_query = translate_to_jp(query)
        target_sites = ["5ch.net", "x.com", "youtube.com"]
    elif region == "US":
        search_query = translate_to_en(query)
        target_sites = ["reddit.com", "x.com", "youtube.com", "4chan.org", "quora.com"]
    else:
        search_query, target_sites = query, ["dcinside.com", "fmkorea.com", "ruliweb.com", "theqoo.net", "arca.live"]
    
    images = fetch_top_images(search_query)
    collected_context, raw_list, site_stats = fetch_community_data_weighted(search_query, target_sites)
    final_report = generate_core_summary(collected_context)
    
    return jsonify({
        "images": images, "report": final_report, 
        "raw_data_list": raw_list, "site_stats": site_stats, 
        "translated_query": search_query
    })

@app.route('/api/translate', methods=['POST'])
def translate_snippet():
    data = request.json
    text = data.get("text", "")
    if not text: return jsonify({"translated_text": ""})
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "한국어 번역"}, {"role": "user", "content": text}],
            temperature=0.3
        )
        return jsonify({"translated_text": response.choices[0].message.content.strip()})
    except: return jsonify({"translated_text": "번역 실패"}), 500