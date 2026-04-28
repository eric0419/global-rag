import requests
import json
import os 
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

SERPER_API_KEY = os.getenv("SERPER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def fetch_community_data(query, sites):
    all_context = ""
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}
    url = "https://google.serper.dev/search"

    print(f"🔍 [{query}] 5대 커뮤니티 정밀 탐색을 시작합니다...\n")

    for site in sites:
        # 각 사이트별로 site: 연산자를 붙여 3개씩 가져옴
        payload = json.dumps({
            "q": f"site:{site} {query}",
            "gl": "kr",
            "hl": "ko",
            "num": 3
        })
        
        try:
            response = requests.request("POST", url, headers=headers, data=payload)
            response.raise_for_status()
            items = response.json().get('organic', [])

            if items:
                print(f"🌐 [출처: {site}]")
                for idx, item in enumerate(items, 1):
                    title = item.get('title')
                    link = item.get('link')
                    snippet = item.get('snippet')
                    
                    # 터미널 출력 (초기 코드 형식)
                    print(f"--- 결과 {idx} ---")
                    print(f"제목: {title}")
                    print(f"링크: {link}")
                    print(f"요약(Snippet): {snippet}\n")
                    
                    # LLM 전송용 데이터 축적 (사이트 구분 없이 합침)
                    all_context += f"제목: {title}\n내용: {snippet}\n\n"
            else:
                print(f"🌐 [출처: {site}] 검색 결과가 없습니다.\n")

        except Exception as e:
            print(f"❌ {site} 검색 중 에러 발생: {e}")

    return all_context

def generate_core_summary(context_text):
    if not context_text:
        return "분석할 데이터가 없습니다."

    print("🤖 수집된 데이터를 바탕으로 핵심 여론을 분석 중입니다...\n")
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = """
    당신은 수많은 커뮤니티 반응을 하나로 꿰뚫어 보는 전문 분석가입니다.
    제공된 모든 검색 결과들을 종합하여, 현재 대중이 가장 중요하게 생각하는 핵심 여론을 분석하세요.

    [작성 규칙]
    1. 출처나 사이트별로 구분하지 말고 전체를 하나로 통합해서 분석하세요.
    2. 번호를 매기지 마세요.
    3. 딱 3줄 정도로 핵심적인 부분만 명확하게 작성하세요.
    4. 분석적이고 객관적인 어조를 유지하세요.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"다음은 수집된 여론 데이터입니다. 핵심만 3줄 요약해주세요:\n\n{context_text}"}
            ],
            temperature=0.5
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM 에러: {e}"

# --- 메인 실행 ---
if __name__ == "__main__":
    # 타겟 사이트 목록
    target_sites = [
        "dcinside.com", 
        "fmkorea.com", 
        "ruliweb.com", 
        "theqoo.net", 
        "arca.live"
    ]
    
    user_query = "체인소맨 결말 반응"
    
    # 1. 사이트별 데이터 수집 및 개별 출력
    collected_context = fetch_community_data(user_query, target_sites)
    
    # 2. 통합 3줄 요약 리포트 생성
    final_report = generate_core_summary(collected_context)
    
    print("=======================================================")
    print("📊 [Global Echo RAG 통합 여론 분석 리포트]")
    print("=======================================================")
    print(final_report)
    print("=======================================================")