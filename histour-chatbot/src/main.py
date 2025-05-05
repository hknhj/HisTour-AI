from fastapi import FastAPI, Request
from pydantic import BaseModel
from chatbot_core import ask_heritage_chatbot
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS 설정 (프론트엔드와 연동할 경우 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 서비스에서는 도메인을 지정하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    
@app.post("/chat")
async def chat(request: ChatRequest):
    response = ask_heritage_chatbot(request.question)
    return {"answer": response}