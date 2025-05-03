import psycopg2
import os

def save_chat_to_db(user_id, question, answer):
    print(f"[저장 생략] {user_id=} {question=} {answer[:30]}...")
    return
