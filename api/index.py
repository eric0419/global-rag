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

COMMUNITY_THRESHOLD = 10

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

def fetch_top_images(query, tbs=""):
    url = "https://google.serper.dev/images"
    payload_dict = {"q": query, "gl": "kr", "hl": "ko", "num": 3}
    if tbs:
        payload_dict["tbs"] = tbs
    
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    image_urls = []
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload_dict))
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

def extract_root_domain(url):
    try:
        netloc = urlparse(url).netloc
        parts = netloc.split('.')
        return '.'.join(parts[-2:])
    except Exception:
        return ''

def fetch_paginated(query, gl="kr", hl="ko", tbs="", target=40):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"
    all_items = []
    page = 1

    while len(all_items) < target:
        payload_dict = {
            "q": query,
            "gl": gl, "hl": hl,
            "num": 10,
            "page": page
        }
        if tbs:
            payload_dict["tbs"] = tbs
            
        try:
            res = requests.post(url, headers=headers, data=json.dumps(payload_dict)).json()
            items = res.get('organic', [])
            if not items:
                break
            all_items.extend(items)
            page += 1
        except Exception as e:
            break

    return all_items[:target]

def fetch_by_broad_search(query, target_sites, gl="kr", hl="ko", tbs=""):
    all_items = fetch_paginated(query, gl=gl, hl=hl, tbs=tbs, target=40)
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

    return raw_list, site_counts, other_count

def fetch_by_site_search(query, target_sites, gl="kr", hl="ko", tbs=""):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"
    site_counts = {}
    raw_list = []

    for site in target_sites:
        payload_dict = {"q": f"site:{site} {query}", "gl": gl, "hl": hl, "num": 3}
        if tbs:
            payload_dict["tbs"] = tbs
            
        try:
            res = requests.post(url, headers=headers, data=json.dumps(payload_dict)).json()
            items = res.get('organic', [])
            site_counts[site] = len(items)

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
            site_counts[site] = 0

    raw_list.sort(key=lambda x: x.pop('dt_object'), reverse=True)
    return raw_list, site_counts

def fetch_community_data(query, target_sites, gl="kr", hl="ko", tbs=""):
    raw_list, site_counts, other_count = fetch_by_broad_search(query, target_sites, gl=gl, hl=hl, tbs=tbs)
    total_community = sum(site_counts.values())

    if total_community >= COMMUNITY_THRESHOLD:
        site_stats = [{"site": site, "count": site_counts[site]} for site in target_sites]
        site_stats.append({"site": "기타", "count": other_count})
    else:
        raw_list, site_counts = fetch_by_site_search(query, target_sites, gl=gl, hl=hl, tbs=tbs)
        site_stats = [{"site": site, "count": site_counts.get(site, 0)} for site in target_sites]

    all_context = ""
    for entry in raw_list:
        all_context += f"제목: {entry['title']}\n내용: {entry['snippet']}\n\n"

    return all_context, raw_list, site_stats

def cosine_similarity(vec1, vec2):
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def attach_similarity_scores(summary, raw_list):
    if not raw_list or not summary:
        return raw_list

    client = OpenAI(api_key=OPENAI_API_KEY)
    texts = [summary] + [f"{item.get('title', '')} {item.get('snippet', '')}" for item in raw_list]
    
    try:
        response = client.embeddings.create(input=texts, model="text-embedding-3-small")
        embeddings = [item.embedding for item in response.data]
        
        summary_emb = embeddings[0]
        snippet_embs = embeddings[1:]
        
        best_idx = -1
        best_score = -1.0
        
        for i, emb in enumerate(snippet_embs):
            score = cosine_similarity(summary_emb, emb)
            raw_list[i]['similarity_score'] = score
            raw_list[i]['is_top_reference'] = False
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_idx != -1:
            raw_list[best_idx]['is_top_reference'] = True
            
    except Exception:
        pass
        
    return raw_list

@app.route('/api/parse_intent', methods=['POST'])
def parse_intent_handler():
    data = request.json
    user_input = data.get("query", "")
    current_region = data.get("current_region", "KR")
    
    if not user_input:
        return jsonify({"region": current_region, "optimized_query": ""})

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    system_prompt = f"""
    당신은 글로벌 여론 검색 엔진의 '쿼리 라우터(Query Router)'입니다.
    사용자의 자연어 질문을 분석하여 타겟 국가코드(KR, JP, US)와 검색 엔진에 입력할 핵심 키워드를 추출하세요.
    
    [규칙]
    1. 사용자의 현재 기본 국가 설정은 "{current_region}"입니다.
    2. '일본', '미국', '해외' 등 질문 내에 명백하게 특정 국가를 지칭하는 단어가 있다면 그에 맞춰 region을 "JP" 또는 "US" 등으로 변경하세요.
    3. 특정 국가를 지칭하는 단어가 없다면, 반드시 사용자의 현재 기본 국가 설정인 "{current_region}"을 그대로 유지하세요.
    4. [중요] optimized_query를 만들 때, '일본', '미국', '한국', '해외' 같은 국가/지역 지칭 단어와 '~알려줘', '~어때', '~찾아줘' 같은 대화형 서술어만 제거하세요.
    5. '후기', '반응', '논란', '평가' 등 검색 목적을 나타내는 명사 키워드는 절대 지우지 말고 그대로 포함하세요.
    
    [변환 예시]
    - "프로젝트 헤일메리 일본 반응 알려줘" -> region: "JP", optimized_query: "프로젝트 헤일메리 반응"
    - "체인소맨 결말 미국 후기 어때" -> region: "US", optimized_query: "체인소맨 결말 후기"
    - "왕이 사는 남자 후기" -> region: "{current_region}", optimized_query: "왕이 사는 남자 후기"
    
    6. 반드시 JSON 형식으로만 출력할 것.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return jsonify({
            "region": result.get("region", current_region),
            "optimized_query": result.get("optimized_query", user_input)
        })
        
    except Exception:
        return jsonify({
            "region": current_region,
            "optimized_query": user_input
        })

@app.route('/api/search', methods=['POST'])
def search_handler():
    data = request.json
    query = data.get("query", "")
    region = data.get("region", "KR")
    tbs = data.get("tbs", "qdr:w")

    if not query:
        return jsonify({"error": "검색어를 입력해주세요."}), 400

    if region == "JP":
        search_query = translate_to_jp(query)
        target_sites = ["5ch.net", "x.com", "youtube.com"]
        gl, hl = "jp", "ja"
    elif region == "US":
        search_query = translate_to_en(query)
        target_sites = ["reddit.com", "x.com", "youtube.com", "4chan.org", "quora.com"]
        gl, hl = "us", "en"
    else:
        search_query = query
        target_sites = ["dcinside.com", "fmkorea.com", "ruliweb.com", "theqoo.net", "arca.live"]
        gl, hl = "kr", "ko"

    images = fetch_top_images(search_query, tbs=tbs)
    collected_context, raw_list, site_stats = fetch_community_data(search_query, target_sites, gl=gl, hl=hl, tbs=tbs)
    final_report = generate_core_summary(collected_context)
    
    raw_list = attach_similarity_scores(final_report, raw_list)

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
    except Exception:
        return jsonify({"translated_text": "번역 실패"}), 500