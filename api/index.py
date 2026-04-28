from flask import Flask, request, jsonify
import requests
import json
import os
from openai import OpenAI

app = Flask(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def translate_to_jp(query):
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "사용자의 한국어 검색어를 일본어로 번역하세요. 설명이나 따옴표 없이 번역된 결과(일본어)만 출력하세요."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"번역 에러: {e}")
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

def fetch_community_data(query, sites):
    all_context = ""
    raw_list = []
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"

    for site in sites:
        payload = json.dumps({"q": f"site:{site} {query}", "gl": "kr", "hl": "ko", "num": 3})
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            items = response.json().get('organic', [])
            if items:
                for item in items:
                    title = item.get('title', '제목 없음')
                    snippet = item.get('snippet', '내용 없음')
                    link = item.get('link', '#')
                    date = item.get('date', '') # 날짜 데이터 추출 시도
                    
                    all_context += f"제목: {title}\n내용: {snippet}\n\n"
                    
                    raw_list.append({
                        "site": site,
                        "title": title,
                        "snippet": snippet,
                        "link": link,
                        "date": date
                    })
        except Exception:
            continue
            
    return all_context, raw_list

def generate_core_summary(context_text):
    if not context_text:
        return "분석할 데이터가 없습니다."
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = """
    당신은 수많은 다국어 커뮤니티 반응을 하나로 꿰뚫어 보는 전문 분석가입니다.
    제공된 검색 결과들을 종합하여 핵심 여론을 분석하세요.
    1. 원본 데이터가 외국어라도 반드시 '한국어'로 작성할 것
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
    else:
        search_query = query
        target_sites = ["dcinside.com", "fmkorea.com", "ruliweb.com", "theqoo.net", "arca.live"]
    
    images = fetch_top_images(search_query)
    collected_context, raw_list = fetch_community_data(search_query, target_sites)
    final_report = generate_core_summary(collected_context)
    
    return jsonify({
        "images": images,
        "report": final_report,
        "raw_data_list": raw_list,
        "translated_query": search_query
    })

# 원본 데이터 번역을 위한 새로운 API 엔드포인트
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