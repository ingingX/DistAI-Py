import random
import asyncio
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from transformers import CLIPTokenizer, CLIPModel


MODEL_NAME = os.getenv("MODEL", "openai/clip-vit-base-patch32")
WORKER_ID = os.getenv("WORKER_ID", "w3-clip")
FAIL_RATE = float(os.getenv("FAIL_RATE", "0.1"))
DELAY_RANGE = (0.1, 1.0)

print(f"[{WORKER_ID}] Loading model {MODEL_NAME}...")
tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
model = CLIPModel.from_pretrained(MODEL_NAME)
model.eval()
print(f"[{WORKER_ID}] Model loaded.")

app = FastAPI()

class InferenceRequest(BaseModel):
    input: str  # 输入文本

@app.post("/infer")
async def infer(request: InferenceRequest):

    await asyncio.sleep(random.uniform(*DELAY_RANGE))


    if random.random() < FAIL_RATE:
        raise HTTPException(status_code=500, detail=f"{WORKER_ID} simulated failure")


    inputs = tokenizer(request.input, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)  
        # 提取文本特征
        vector = outputs.squeeze().tolist()

    return {
        "worker_id": WORKER_ID,
        "model": MODEL_NAME,
        "input": request.input,
        "text_vector": vector[:5]  # 返回前5维向量（为简洁）
    }

@app.get("/health")
async def health():
    return {"status": "ok", "worker_id": WORKER_ID}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9003))
    uvicorn.run("w3-CLIP:app", host="0.0.0.0", port=port)
