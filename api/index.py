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
            if 'hour' in date_str:
                return now - timedelta(hours=num)
            elif 'day' in date_str:
                return now - timedelta(days=num)
            elif 'week' in date_str:
                return now - timedelta(weeks=num)
            elif 'month' in date_str:
                return now - timedelta(days=num * 30)
        
        for fmt in ("%b %d, %Y", "%Y. %m. %d.", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str.strip('. '), fmt)
            except Exception:
                continue
    except Exception:
        pass
    return datetime.min

# --- 핵심 수정: 전체 긁어온 뒤 가중치로 필터링 ---
def fetch_community_data_weighted(query, sites):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"
    TOTAL_TARGET = 15

    # 1단계: 모든 사이트에서 num=10으로 최대한 수집 + totalResults 파악
    site_buckets = []
    total_mentions = 0

    for site in sites:
        payload = json.dumps({
            "q": f"site:{site} {query}",
            "gl": "kr", "hl": "ko",
            "num": 10  # 항상 최대로 요청
        })
        try:
            res = requests.post(url, headers=headers, data=payload).json()
            count = int(res.get('searchInformation', {}).get('totalResults', 0) or 0)
            items = res.get('organic', [])

            site_buckets.append({
                "site": site,
                "count": count,
                "items": items
            })
            total_mentions += count
        except Exception:
            site_buckets.append({"site": site, "count": 0, "items": []})

    # 2단계: totalResults 비율로 각 사이트에서 몇 개 포함할지 결정 후 슬라이싱
    raw_list = []

    for bucket in site_buckets:
        site = bucket["site"]
        count = bucket["count"]
        items = bucket["items"]

        if total_mentions > 0 and count > 0:
            keep = max(1, round((count / total_mentions) * TOTAL_TARGET))
        else:
            # 언급량 데이터가 없으면 균등 배분
            keep = max(1, TOTAL_TARGET // len(sites))

        # 실제 수집된 것 중에서 keep개만 선택
        for entry in items[:keep]:
            raw_list.append({
                "site": site,
                "title": entry.get('title', '제목 없음'),
                "snippet": entry.get('snippet', '내용 없음'),
                "link": entry.get('link', '#'),
                "date": entry.get('date', ''),
                "dt_object": parse_date(entry.get('date', ''))
            })

    # 3단계: 최신순 정렬
    raw_list.sort(key=lambda x: x['dt_object'], reverse=True)

    # 4단계: 요약용 컨텍스트 생성 + dt_object 제거
    all_context = ""
    for entry in raw_list:
        all_context += f"제목: {entry['title']}\n내용: {entry['snippet']}\n\n"
        del entry['dt_object']

    # 언급량 통계도 함께 반환 (차트용)
    site_stats = [{"site": b["site"], "count": b["count"]} for b in site_buckets]

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
    collected_context, raw_list, site_stats = fetch_community_data_weighted(search_query, target_sites)
    final_report = generate_core_summary(collected_context)
    
    return jsonify({
        "images": images,
        "report": final_report,
        "raw_data_list": raw_list,
        "site_stats": site_stats,       # 언급량 차트용 데이터 (프론트엔드에서 선택적으로 활용)
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