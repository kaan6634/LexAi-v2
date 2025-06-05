from fastapi import FastAPI
from pydantic import BaseModel
from post_processing_model import ask_pipeline

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    answer_tr = ask_pipeline(request.question)
    return {"answer": answer_tr}
