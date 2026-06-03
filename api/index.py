from flask import Flask, request, jsonify
import requests
import json
import os
from openai import OpenAI
from datetime import datetime, timedelta
from urllib.parse import urlparse, parse_qs
import re

app = Flask(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

COMMUNITY_THRESHOLD = 10

def translate_query(query, target_language):
    client = OpenAI(api_key=OPENAI_API_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": f"사용자의 한국어 검색어를 {target_language}로 번역하세요. 설명이나 따옴표 없이 번역된 결과만 출력하세요."},
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
        except Exception:
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
        except Exception:
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

def fetch_fixed_jp_data(query, gl="jp", hl="ja", tbs=""):
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"
    target_sites = ["youtube.com", "x.com"]
    site_counts = {}
    raw_list = []

    for site in target_sites:
        payload_dict = {"q": f"site:{site} {query}", "gl": gl, "hl": hl, "num": 6}
        if tbs:
            payload_dict["tbs"] = tbs
            
        try:
            res = requests.post(url, headers=headers, data=json.dumps(payload_dict)).json()
            items = res.get('organic', [])[:6]
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
        except Exception:
            site_counts[site] = 0

    raw_list.sort(key=lambda x: x.pop('dt_object'), reverse=True)
    
    all_context = ""
    for entry in raw_list:
        all_context += f"제목: {entry['title']}\n내용: {entry['snippet']}\n\n"
        
    site_stats = [{"site": site, "count": site_counts.get(site, 0)} for site in target_sites]
    
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

def extract_youtube_video_id(url):
    try:
        parsed = urlparse(url)
        if parsed.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.hostname == "youtu.be":
            return parsed.path[1:]
    except Exception:
        pass
    return None

def is_low_quality_comment(text):
    text = text.strip().lower()
    if len(text) < 12:
        return True
    return False

def classify_comments_batch(comment_texts):
    if not comment_texts:
        return []

    client = OpenAI(api_key=OPENAI_API_KEY)
    formatted = "\n".join([f"{idx+1}. {text}" for idx, text in enumerate(comment_texts)])

    prompt = f"""
다음 유튜브 댓글들을 각각 분류하세요.
가능한 분류:
- positive
- negative
- other
반드시 'results'라는 키에 분류 결과를 배열로 담은 JSON 객체 형식으로 반환하세요.
예시:
{{
  "results": ["positive", "other"]
}}

댓글:
{formatted}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={ "type": "json_object" },
            messages=[
                {"role": "system", "content": "당신은 유튜브 댓글 감정 분류기입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)

        if isinstance(parsed, dict):
            parsed = parsed.get("results", [])
        if not isinstance(parsed, list):
            return ["other"] * len(comment_texts)
        return parsed
    except Exception:
        return ["other"] * len(comment_texts)

def fetch_youtube_sidebar_data(raw_list):
    youtube_entries = []
    seen_ids = set()

    for item in raw_list:
        link = item.get("link", "")
        if "youtube.com" not in link and "youtu.be" not in link:
            continue
        video_id = extract_youtube_video_id(link)
        if not video_id or video_id in seen_ids:
            continue
        seen_ids.add(video_id)
        youtube_entries.append({
            "video_id": video_id,
            "title": item.get("title", "제목 없음"),
            "thumbnail": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
            "video_url": f"https://www.youtube.com/watch?v={video_id}"
        })

    youtube_entries = youtube_entries[:6]
    comments_to_classify = []

    for entry in youtube_entries:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": entry["video_id"],
            "maxResults": 10,
            "order": "relevance",
            "textFormat": "plainText",
            "key": YOUTUBE_API_KEY
        }
        try:
            res = requests.get(url, params=params).json()
            usable_comments = []
            for item in res.get("items", []):
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                text = snippet.get("textDisplay", "")
                likes = snippet.get("likeCount", 0)
                if is_low_quality_comment(text):
                    continue
                usable_comments.append({"text": text, "likes": likes})
            usable_comments.sort(key=lambda x: x["likes"], reverse=True)
            top_comment = usable_comments[0] if usable_comments else {"text": "데이터가 부족합니다.", "likes": 0}
        except Exception:
            top_comment = {"text": "댓글 데이터를 불러올 수 없습니다.", "likes": 0}
        
        entry["top_comment"] = top_comment
        comments_to_classify.append(top_comment["text"])

    sentiments = classify_comments_batch(comments_to_classify)

    for idx, entry in enumerate(youtube_entries):
        sentiment = "other"
        if idx < len(sentiments) and sentiments[idx] in ["positive", "negative", "other"]:
            sentiment = sentiments[idx]
        entry["top_comment"]["sentiment"] = sentiment

    return youtube_entries

@app.route('/api/parse_intent', methods=['POST'])
def parse_intent_handler():
    data = request.json
    user_input = data.get("query", "")
    current_region = data.get("current_region", "KR")
    
    if not user_input:
        return jsonify({"region": current_region, "optimized_query": ""})

    client = OpenAI(api_key=OPENAI_API_KEY)
    
    system_prompt = f"""
    당신은 글로벌 여론 분석 대시보드의 '검색어 라우팅 에이전트'입니다.
    사용자가 입력한 자연어(문장형) 질문의 핵심 의도를 파악하여, 타겟 국가(region)와 검색 엔진에 입력할 최적의 명사형 키워드(optimized_query)를 추출하세요.

    [핵심 규칙]
    1. 불용어 제거: '~알려줘', '~어때', '~찾아봐', '요즘', '애들은', '진짜', '좀' 등 대화형 서술어와 수식어를 완벽하게 제거하세요.
    2. 명사 압축: 검색 엔진(Google)이 가장 좋아할 만한 2~3개의 핵심 고유명사와 목적어(예: 후기, 반응, 리뷰)의 조합으로만 쿼리를 재구성하세요.
    3. 지역 라우팅: 문장 내에 국가를 지칭하는 단어가 있다면 아래의 region 코드로 변경하고, 해당 국가 키워드 자체는 optimized_query에서 지우세요.
       - 지원 국가 및 코드: KR(한국), JP(일본), US(미국), DE(독일), FR(프랑스), GB(영국), AU(호주), TR(터키)
    4. 질문에 특정 국가 지칭 단어가 없다면 반드시 사용자의 현재 기본 국가 설정인 "{current_region}"을 유지하세요.

    [변환 예시 (Few-Shot)]
    - Input: "요즘 미국 애들은 마블 영화 개봉하면 반응이 어때?"
      Output: {{"region": "US", "optimized_query": "마블 영화 반응"}}
    - Input: "독일 현지에서 폭스바겐 논란 여론 찾아줘"
      Output: {{"region": "DE", "optimized_query": "폭스바겐 논란 여론"}}
    - Input: "흑백요리사 안성재 셰프 논란 요약해줘"
      Output: {{"region": "{current_region}", "optimized_query": "흑백요리사 안성재 논란"}}

    반드시 위 예시와 같은 JSON 형식으로만 응답하세요.
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
        search_query = translate_query(query, "일본어")
        gl, hl = "jp", "ja"
        images = fetch_top_images(search_query, tbs=tbs)
        collected_context, raw_list, site_stats = fetch_fixed_jp_data(search_query, gl=gl, hl=hl, tbs=tbs)
    elif region == "US":
        search_query = translate_query(query, "영어")
        target_sites = ["reddit.com", "x.com", "youtube.com", "4chan.org", "quora.com"]
        gl, hl = "us", "en"
        images = fetch_top_images(search_query, tbs=tbs)
        collected_context, raw_list, site_stats = fetch_community_data(search_query, target_sites, gl=gl, hl=hl, tbs=tbs)
    elif region == "DE":
        search_query = translate_query(query, "독일어")
        target_sites = ["reddit.com", "x.com", "youtube.com", "facebook.com"]
        gl, hl = "de", "de"
        images = fetch_top_images(search_query, tbs=tbs)
        collected_context, raw_list, site_stats = fetch_community_data(search_query, target_sites, gl=gl, hl=hl, tbs=tbs)
    elif region == "FR":
        search_query = translate_query(query, "프랑스어")
        target_sites = ["reddit.com", "x.com", "youtube.com", "facebook.com"]
        gl, hl = "fr", "fr"
        images = fetch_top_images(search_query, tbs=tbs)
        collected_context, raw_list, site_stats = fetch_community_data(search_query, target_sites, gl=gl, hl=hl, tbs=tbs)
    elif region == "GB":
        search_query = translate_query(query, "영어")
        target_sites = ["reddit.com", "x.com", "youtube.com", "facebook.com"]
        gl, hl = "gb", "en"
        images = fetch_top_images(search_query, tbs=tbs)
        collected_context, raw_list, site_stats = fetch_community_data(search_query, target_sites, gl=gl, hl=hl, tbs=tbs)
    elif region == "AU":
        search_query = translate_query(query, "영어")
        target_sites = ["reddit.com", "x.com", "youtube.com", "facebook.com"]
        gl, hl = "au", "en"
        images = fetch_top_images(search_query, tbs=tbs)
        collected_context, raw_list, site_stats = fetch_community_data(search_query, target_sites, gl=gl, hl=hl, tbs=tbs)
    elif region == "TR":
        search_query = translate_query(query, "튀르키예어")
        target_sites = ["x.com", "youtube.com", "facebook.com"]
        gl, hl = "tr", "tr"
        images = fetch_top_images(search_query, tbs=tbs)
        collected_context, raw_list, site_stats = fetch_community_data(search_query, target_sites, gl=gl, hl=hl, tbs=tbs)
    else:
        search_query = query
        target_sites = ["dcinside.com", "fmkorea.com", "ruliweb.com", "theqoo.net", "arca.live", "youtube.com"]
        gl, hl = "kr", "ko"
        images = fetch_top_images(search_query, tbs=tbs)
        collected_context, raw_list, site_stats = fetch_community_data(search_query, target_sites, gl=gl, hl=hl, tbs=tbs)

    youtube_sidebar = fetch_youtube_sidebar_data(raw_list)
    final_report = generate_core_summary(collected_context)
    raw_list = attach_similarity_scores(final_report, raw_list)

    return jsonify({
        "images": images,
        "report": final_report,
        "raw_data_list": raw_list,
        "site_stats": site_stats,
        "translated_query": search_query,
        "youtube_sidebar": youtube_sidebar
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