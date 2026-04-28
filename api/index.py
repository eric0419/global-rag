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

# 검색어 기반 상위 3개 이미지 가져오기
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
    except Exception as e:
        print(f"이미지 검색 에러: {e}")
        
    return image_urls

def fetch_community_data(query, sites):
    all_context = ""
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"

    for site in sites:
        payload = json.dumps({"q": f"site:{site} {query}", "gl": "kr", "hl": "ko", "num": 3})
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            items = response.json().get('organic', [])
            if items:
                for item in items:
                    all_context += f"제목: {item.get('title')}\n내용: {item.get('snippet')}\n\n"
        except Exception:
            continue
    return all_context

def generate_core_summary(context_text):
    if not context_text:
        return "분석할 데이터가 없습니다."
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    system_prompt = """
    당신은 수많은 커뮤니티 반응을 하나로 꿰뚫어 보는 전문 분석가입니다.
    제공된 검색 결과들을 종합하여 핵심 여론을 분석하세요.
    1. 사이트별 구분 없이 통합 분석
    2. 번호 매기지 않음
    3. 딱 3줄 정도로 핵심만 명확하게 작성
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
    
    if not query:
        return jsonify({"error": "검색어를 입력해주세요."}), 400
        
    target_sites = ["dcinside.com", "fmkorea.com", "ruliweb.com", "theqoo.net", "arca.live"]
    
    images = fetch_top_images(query)
    collected_context = fetch_community_data(query, target_sites)
    final_report = generate_core_summary(collected_context)
    
    return jsonify({
        "images": images,
        "report": final_report,
        "raw_data": collected_context
    })