from fastapi import FastAPI
from pydantic import BaseModel
from llm_model.inference import predict_next_words_llm

app = FastAPI(
    title="Next Word Prediction API",
    description="LLM-based Next Word Prediction using fine-tuned GPT-2",
    version="1.0"
)

class PredictionRequest(BaseModel):
    text: str
    max_new_tokens: int = 30

class PredictionResponse(BaseModel):
    input_text: str
    generated_text: str

@app.post("/predict", response_model=PredictionResponse)
def predict_next_word(req: PredictionRequest):
    output = predict_next_words_llm(
        prompt=req.text,
        max_new_tokens=req.max_new_tokens
    )
    return {
        "input_text": req.text,
        "generated_text": output
    }
