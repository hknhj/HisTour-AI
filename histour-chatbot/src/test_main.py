from __future__ import annotations

import os
import json
from datetime import datetime

import faiss
import pandas as pd
import numpy as np
import psycopg2
from math import radians, sin, cos, sqrt, atan2
from sentence_transformers import SentenceTransformer
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from db import save_chat_to_db
from typing import Optional
import re
import time
from difflib import SequenceMatcher

# 환경변수 로드
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 앱 초기화
app = FastAPI()

# 📄 CSV 데이터 로딩 (위치·설명 등) citeturn6file0
df = pd.read_csv(
    "https://raw.githubusercontent.com/HisTour-capstone04/HisTour-AI/main/HisTour/heritage/korea_heritage.csv"
)
df.columns = df.columns.str.strip().str.replace('\ufeff', '')

# 🔄 전역 RAG 리소스 로딩 (한 번만) citeturn6file0
model = SentenceTransformer("BAAI/bge-m3")
index = faiss.read_index("heritage_index.index")
with open("heritage_docs.json", "r", encoding="utf-8") as f:
    documents = json.load(f)

# 대화 메모리
messages = [{"role": "system", "content": "너는 한국 문화유산 전문 챗봇이야."}]
MAX_TURNS = 6
last_mentioned_title = None
pending_choices = None

# 🚩 Pydantic 요청 모델
tmp = (0.0, 0.0)
class ChatRequest(BaseModel):
    user_id: str
    question: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None

# 🛰️ GPS 거리 필터링 함수
# 2) GPS 거리 필터링 함수
def filter_by_distance(df: pd.DataFrame, user_loc: tuple | None, radius_km: float = 10.0) -> pd.DataFrame:
    # 컬럼 숫자형 확인 및 변환
    # user_loc 없으면 원본 반환
    if not user_loc:
        return df.reset_index(drop=True)
    lat1, lon1 = map(float, user_loc)
    def haversine(row):
        if row["위도"]=="위도":
            return float('inf')
        lon2 = float(row["위도"])
        lat2 = float(row["경도"])
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return 6371.0 * 2 * atan2(sqrt(a), sqrt(1 - a))
    df_copy = df.copy()
    df_copy["__dist__"] = df_copy.apply(haversine, axis=1)
    nearby = df_copy[df_copy["__dist__"] <= radius_km].drop(columns="__dist__")
    print("근처 유적지", len(nearby))
    if len(nearby)>0:
        return nearby
    else:
        return df

# ❓ 위치 전용 질문 감지
def is_location_only_question(q: str) -> bool:
    kws = ["위치만", "소재지만", "장소만", "위치 알려줘", "소재지 알려줘", "어디야", "어디에 있어"
        , "어디", "위치", "위치를 알려줘", "장소를 알려줘", "장소", "주소", "주소를 알려줘"]
    return any(k in q for k in kws)

# 📍 위치만 응답
def get_location_only(df: pd.DataFrame, question: str):
    cands = []
    for idx, row in df.iterrows():
        title = row['문화재명'].strip()
        if title and title in question:
            cands.append((len(title), idx))
    if not cands:
        return None
    _, best = max(cands)
    r = df.loc[best]
    loc = (r.get('상세주소') or r.get('소재지') or '').strip()
    return f"📍 '{r['문화재명']}'의 위치는 {loc} 입니다." if loc else None

# 🔍 부분 토큰 매칭 + 짧은 제목 우선
# def find_shortest_matching_title(df, question: str):
#     clean_q = question.replace(" ", "").replace("\n", "")
#     candidates = []
#     # ➊ 양방향 substring 매칭
#     for idx, row in df.iterrows():
#         title = str(row["문화재명"]).strip()
#         t_clean = title.replace(" ", "").replace("\n", "")
#         if t_clean and (t_clean in clean_q or clean_q in t_clean):
#             candidates.append((len(t_clean), idx))
#
#     # ➋ substring 매칭이 하나도 없으면, 부분 토큰 매칭
#     if not candidates:
#         for idx, row in df.iterrows():
#             title = str(row["문화재명"]).strip()
#             for part in title.split():
#                 if len(part) >= 2 and part in question:
#                     candidates.append((len(title), idx))
#                     break
#
#     if not candidates:
#         return None
#     print("매칭 문화재", len(candidates))
#     for i in candidates:
#         print(i, df.loc[i[1]]["문화재명"])
#     # ▶︎ 여기만 max → min
#     # _, best_idx = min(candidates, key=lambda x: x[0])
#     best_idx=candidates[0][1]
#     if best_idx in df.index:
#         r = df.loc[best_idx]
#     # r = df.iloc[best_idx]
#     return {
#         "title": r["문화재명"],
#         "description": r.get("상세설명", "").strip(),
#         "location": (r.get("상세주소","") or r.get("소재지","")).strip(),
#     }

def find_best_matching_title(df, question: str):
    clean_q = question.replace(" ", "").replace("\n", "")
    # ➊ 후보 수집은 기존과 동일
    candidates = []
    for idx, row in df.iterrows():
        title = str(row["문화재명"]).strip()
        t_clean = title.replace(" ", "").replace("\n", "")
        if t_clean and (t_clean in clean_q or clean_q in t_clean):
            candidates.append(idx)
    if not candidates:
        for idx, row in df.iterrows():
            title = str(row["문화재명"]).strip()
            for part in title.split():
                if len(part) >= 2 and part in question:
                    candidates.append(idx)
                    break
    if not candidates:
        return None

    # ➋ 점수 계산
    scored = []
    for idx in candidates:
        title   = str(df.loc[idx, "문화재명"]).strip()
        t_clean = title.replace(" ", "").replace("\n", "")
        tokens      = title.split()
        # 질문에 나온 토큰이 제목에 몇 개나 들어있는지
        match_count = sum(1 for tok in tokens if tok in question)
        # 남은(매칭되지 않은) 토큰 개수
        unmatched   = len(tokens) - match_count
        # 순수 문자열 유사도
        sim         = SequenceMatcher(None, t_clean, clean_q).ratio()
        # ( match_count 내림, unmatched 오름, sim 내림, 제목 길이 짧은 순 )
        scored.append(( match_count, -unmatched, sim, -len(tokens), idx ))

    # ➌ 정렬: 튜플 순서대로 내림차순
    scored.sort(reverse=True)
    print("매칭 문화재", len(scored))
    for i in scored:
        print(i, df.loc[i[4]]["문화재명"])
    # ➍ 최상위 후보 선택
    _, _, _, _, best_idx = scored[0]
    r = df.loc[best_idx]
    return {
        "title":       r["문화재명"],
        "description": r.get("상세설명", "").strip(),
        "location":    (r.get("상세주소","") or r.get("소재지","")).strip(),
    }

# 💬 GPT-direct (프롬프트 엔지니어링)
def ask_gpt_direct(question: str):
    resp = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"system","content":"너는 한국 문화유산 전문 챗봇이야."},
                  {"role":"user","content":question}],
        temperature=0.7
    )
    return resp.choices[0].message.content.strip()

# 🤖 RAG → GPT-direct
def ask_with_rag(question: str, top_k: int = 3):
    qv = model.encode([question])
    D,I = index.search(qv, k=top_k)
    docs = [documents[i] for i in I[0] if documents[i].strip()]
    if not docs:
        return ask_gpt_direct(question)
    ctx = "\n---\n".join(docs)
    prompt = f"""
아래 문서를 참고해 정확하고 친절하게 답변해줘. 문서에 없으면 '자료가 없습니다.'라고 응답해줘.

[문서]
{ctx}

[질문]
{question}
[답변]"""
    try:
        r = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            temperature=0.7
        )
        return r.choices[0].message.content.strip()
    except:
        return ask_gpt_direct(question)

# 🔑 챗봇 핵심 로직
def ask_heritage_chatbot(question: str, user_id: str, user_gps: tuple) -> str:
    global messages, last_mentioned_title, pending_choices
    # 1) 지시어 처리
    pronouns = ["것", "곳", "그 곳", "이 곳", "여기는", "그건", "그것은", "여긴", "이건", "그 곳은", "거기", "여기"
        , "이 곳은", "이곳은", "그곳은", "그것의", "이것의", "그 것의", "이 것의", "걔", "얘", "방금 거", "방금", "아까"]
    if any(p in question for p in pronouns) and last_mentioned_title:
        for p in pronouns:
            if p in question:
                question = question.replace(p, last_mentioned_title)
    # 2) GPS 기반 필터링
    # 사용자 위치 기반 필터링
    print("근처 유적지 찾기 시작 : ", datetime.now().second, datetime.now().microsecond)
    if user_gps:
        sub_df = filter_by_distance(df, user_gps)
    else:
        sub_df = df

    # sub_df가 완전히 비어 있으면 원본 df 사용
    if sub_df.empty:
        sub_df = df
    print("그 중에서 매칭 시작 : ", datetime.now().second, datetime.now().microsecond)

    # 3) 위치 전용
    if is_location_only_question(question):
        loc = get_location_only(sub_df, question)
        if loc:
            return loc

    # 4) 정확 매칭
    match = find_best_matching_title(sub_df, question)
    print(match)
    if match:
        print("match")
        last_mentioned_title = match['title']
        # RAG로도 문서+DB 통합 검색 실행 (토큰 절감 위해 DB 검색 코드 생략)
        print("매칭 후 답변 생성 시작 : ", datetime.now().second, datetime.now().microsecond)
        return ask_with_rag(question)
    # 5) 매칭 없음 → GPT-direct + 저장
    print("non match")
    ans = ask_gpt_direct(question)
    print("매칭 못하고 답변 생성 시작 : ", datetime.now().second, datetime.now().microsecond)
    return ans

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat")
async def chat(req: ChatRequest):
    # GPS 정보가 둘 다 있을 때만 tuple, 아니면 None
    if req.latitude is not None and req.longitude is not None:
        user_gps = (float(req.latitude), float(req.longitude))
    else:
        user_gps = None

    answer = ask_heritage_chatbot(
        question=req.question,
        user_id=req.user_id,
        user_gps=user_gps   # 이제 기본값(None) 허용되도록 수정해 두셨죠?
    )
    save_chat_to_db(req.user_id, req.question, answer)
    print("답변 생성 : ", datetime.now().second, datetime.now().microsecond)
    return {"answer": answer}


# latitude가 위도, longitude가 경도 -> 근데 데이터는 반대로 되어 있음....
# uvicorn test_main:app --reload