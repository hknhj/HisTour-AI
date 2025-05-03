import json
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# 🔐 OpenAI API 키
openai.api_key = os.getenv("OPENAI_API_KEY")

# 📄 데이터 로딩
df = pd.read_csv("https://raw.githubusercontent.com/HisTour-capstone04/HisTour-AI/main/HisTour/heritage/korea_heritage.csv")
df.columns = df.columns.str.strip().str.replace('\ufeff', '')

with open("heritage_docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

index = faiss.read_index("heritage_index.index")
model = SentenceTransformer("BAAI/bge-m3")
# 대화 히스토리 초기화
messages = [{"role": "system", "content": "너는 한국 문화유산을 친절하게 설명해주는 챗봇이야."}]

last_mentioned_title = None

def is_location_only_question(question: str):
    keywords = ["위치만", "소재지만", "장소만", "위치 알려줘", "소재지 알려줘", "어디야", "어디에 있어", "어디", "위치", "위치를 알려줘", "장소를 알려줘", "장소", "주소", "주소를 알려줘"]
    return any(k in question for k in keywords)


def get_location_only(df, question):
    # 1) 후보 찾기
    candidates = []
    for idx, row in df.iterrows():
        title = row["문화재명"].strip()
        if title and title in question:
            candidates.append((len(title), idx))
    if not candidates:
        return None

    # 2) 가장 긴 제목 우선
    best_idx = sorted(candidates, reverse=True)[0][1]
    row = df.iloc[best_idx]
    title = row["문화재명"].strip()

    # 3) 상세주소 혹은 소재지 우선 사용
    addr = row.get("상세주소", "")
    if not addr:
        addr = row.get("소재지", "")
    addr = addr.strip()

    # 4) 통일된 포맷으로 응답
    if addr:
        return f"'{title}'의 위치는 {addr}입니다."
    else:
        return f"'{title}'의 위치 정보는 등록되어 있지 않아요."



def find_longest_matching_title(df, question):
    question = question.replace(" ", "").replace("\n", "")  # 사용자 질문도 정리

    candidates = []
    for idx, row in df.iterrows():
        title = str(row["문화재명"]).strip().replace(" ", "").replace("\n", "")
        if title in question:
            candidates.append(title)

    if not candidates:
        return None

    best_match = max(candidates, key=len)

    # 모든 공백 문자 제거 후 비교
    filtered = df[df["문화재명"].str.replace(r"\s+", "", regex=True) == best_match]
    if filtered.empty:
        return None

    match_row = filtered.iloc[0]
    return {
        "title": match_row["문화재명"],
        "location": str(match_row.get("상세주소", "")).strip(),
        "description": str(match_row.get("상세설명", "")).strip(),
    }

def ask_with_rag(question, top_k=3):
    q_vec = model.encode([question])
    D, I = index.search(q_vec, k=top_k)
    retrieved_docs = [documents[i] for i in I[0]]
    context = "\n---\n".join(retrieved_docs)

    context_message = {
        "role": "system",
        "content": f"""참고 문서를 기반으로 답변해줘. 모르는 내용은 '자료가 없습니다.'라고 해줘.\n\n[참고 문서]\n{context}"""
    }

    local_messages = messages + [context_message, {"role": "user", "content": question}]

    response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=local_messages,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def ask_heritage_chatbot(question):
    global last_mentioned_title

    # 지시어 목록 (필요시 더 추가 가능)
    pronouns = ["것", "곳", "그 곳", "이 곳","여기는", "그건", "그것은", "여긴", "이건", "그 곳은", "이 곳은", "이곳은", "그곳은", "그것의", "이것의", "그 것의", "이 것의", "걔", "얘", "방금 거", "방금", "아까"]
    # 3. 제목 매칭 시도
    # 1. 지시어를 포함한 질문 → 마지막 언급 문화재명으로 대체
    if any(p in question for p in pronouns) and last_mentioned_title:
        for p in pronouns:
            if p in question:
                question = question.replace(p, last_mentioned_title)
        print(f"지시어 대체: {question}")  # 디버깅용

    match = find_longest_matching_title(df, question)
    print(f"문서에서 찾은 문화재: {match['title'] if match else '없음'}")  # 디버깅용

    # 2. 위치만 묻는 질문이면 바로 응답
    if is_location_only_question(question):
        location_answer = get_location_only(df, question)
        if location_answer:
            print("문서에서 장소 정보만 제공")
            return location_answer

    if match is not None:
        last_mentioned_title = match["title"]
        print("문서 내에서 일치함")

        prompt = f"""사용자가 "{match['title']}"에 대해 질문했어.
        아래는 공식 설명이야. 이를 참고해서 친절하고 자연스럽게 설명해줘.

        [설명]
        {match['description']}

        [소재지]
        {match['location']}

        [답변]
        """
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0.7
        )
        answer = response.choices[0].message.content.strip()
        messages.append({"role": "assistant", "content": answer})
        return answer

    # 4. fallback: 검색
    print("GPT 통해서 프롬프트 검색")
    answer = ask_with_rag(question)
    messages.append({"role": "user", "content": question})
    messages.append({"role": "assistant", "content": answer})
    return answer