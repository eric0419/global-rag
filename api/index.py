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

def parse_date(date_str):
    if not date_str:
        return datetime.min
    
    now = datetime.now()
    try:
        # "3 hours ago", "5 days ago" 형태 처리
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
        
        # "Mar 25, 2026" 또는 "2026. 03. 25." 형태 처리
        # 구글의 다양한 날짜 형식을 시도
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
            # totalResults가 없을 경우 0으로 처리
            count = res.get('searchInformation', {}).get('totalResults', 0)
            # 문자열 형태일 수 있으므로 숫자로 변환
            count = int(count) if count else 0
            
            site_stats.append({"site": site, "count": count})
            total_mentions += count
        except:
            site_stats.append({"site": site, "count": 0})

    # 2단계: 가중치에 따른 수집 개수 할당 (총 15~20개 목표)
    TOTAL_TARGET = 15
    raw_list = []
    all_context = ""

    for item in site_stats:
        site = item['site']
        count = item['count']
        
        # 점유율 계산 (최소 1개, 최대 8개 제한으로 다양성 확보)
        if total_mentions > 0:
            assigned_num = max(1, min(8, round((count / total_mentions) * TOTAL_TARGET)))
        else:
            assigned_num = 3 # 정보가 없으면 기본 3개
            
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
                    "dt_object": parse_date(entry.get('date', '')) # 정렬용 임시 객체
                })
        except:
            continue

    # 3단계: 날짜 기준 내림차순 정렬 (최신순)
    raw_list.sort(key=lambda x: x['dt_object'], reverse=True)

    # 4단계: 요약용 텍스트 컨텍스트 생성 (정렬된 순서대로)
    for entry in raw_list:
        all_context += f"제목: {entry['title']}\n내용: {entry['snippet']}\n\n"
        # 정렬용 임시 객체는 JSON 응답에 포함되지 않도록 삭제
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
    
    # 가중치 및 정렬 로직이 적용된 함수 호출
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