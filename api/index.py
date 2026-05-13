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

# --- 복구된 기존 함수들 ---
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

def fetch_top_images(query):
    url = "https://google.serper.dev/images"
    payload = json.dumps({"q": query, "gl": "kr", "hl": "ko", "num": 3})
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    
    image_urls = []
    try:
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()
        items = response.json().get('images', [])
        
        for item in items[:3]:
            image_urls.append(item.get('imageUrl'))
    except Exception:
        pass
        
    return image_urls

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
# --------------------------

# --- 새롭게 추가했던 가중치 및 날짜 정렬 함수 ---
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
            except:
                continue
    except:
        pass
    return datetime.min

def fetch_community_data_weighted(query, sites):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"
    
    site_stats = []
    total_mentions = 0
    
    # 1단계: 각 사이트별 언급량(totalResults) 파악
    for site in sites:
        payload = json.dumps({"q": f"site:{site} {query}", "gl": "kr", "hl": "ko", "num": 1})
        try:
            res = requests.post(url, headers=headers, data=payload).json()
            count = res.get('searchInformation', {}).get('totalResults', 0)
            count = int(count) if count else 0
            
            site_stats.append({"site": site, "count": count})
            total_mentions += count
        except:
            site_stats.append({"site": site, "count": 0})

    # 2단계: 가중치에 따른 수집 개수 할당 (총 15개 목표)
    TOTAL_TARGET = 15
    raw_list = []
    all_context = ""

    for item in site_stats:
        site = item['site']
        count = item['count']
        
        if total_mentions > 0:
            assigned_num = max(1, min(8, round((count / total_mentions) * TOTAL_TARGET)))
        else:
            assigned_num = 3 
            
        payload = json.dumps({"q": f"site:{site} {query}", "gl": "kr", "hl": "ko", "num": assigned_num})
        try:
            res = requests.post(url, headers=headers, data=payload).json()
            items = res.get('organic', [])
            for entry in items:
                raw_list.append({
                    "site": site,
                    "title": entry.get('title', '제목 없음'),
                    "snippet": entry.get('snippet', '내용 없음'),
                    "link": entry.get('link', '#'),
                    "date": entry.get('date', ''),
                    "dt_object": parse_date(entry.get('date', ''))
                })
        except:
            continue

    # 3단계: 날짜 기준 내림차순 정렬 (최신순)
    raw_list.sort(key=lambda x: x['dt_object'], reverse=True)

    # 4단계: 요약용 텍스트 컨텍스트 생성
    for entry in raw_list:
        all_context += f"제목: {entry['title']}\n내용: {entry['snippet']}\n\n"
        del entry['dt_object']

    return all_context, raw_list

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
    collected_context, raw_list = fetch_community_data_weighted(search_query, target_sites)
    final_report = generate_core_summary(collected_context)
    
    return jsonify({
        "images": images,
        "report": final_report,
        "raw_data_list": raw_list,
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