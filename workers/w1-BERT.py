import random
import asyncio
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import torch
from transformers import BertTokenizer, BertModel

# 初始化模型
MODEL_NAME = os.getenv("MODEL", "prajjwal1/bert-tiny")
WORKER_ID = os.getenv("WORKER_ID", "w1-bert")
FAIL_RATE = float(os.getenv("FAIL_RATE", "0.2"))
DELAY_RANGE = (0.1, 1.0)  # 模拟延迟范围（秒）

print(f"[{WORKER_ID}] Loading model {MODEL_NAME}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
model.eval()
print(f"[{WORKER_ID}] Model loaded.")

app = FastAPI()

class InferenceRequest(BaseModel):
    input: str

@app.post("/infer")
async def infer(request: InferenceRequest):
    # 模拟网络延迟
    await asyncio.sleep(random.uniform(*DELAY_RANGE))

    # 模拟失败
    if random.random() < FAIL_RATE:
        raise HTTPException(status_code=500, detail=f"{WORKER_ID} simulated failure")

    # 文本编码
    inputs = tokenizer(request.input, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # 提取 CLS token 的向量作为输出
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().tolist()

    return {
        "worker_id": WORKER_ID,
        "model": MODEL_NAME,
        "input": request.input,
        "output_cls_vector": cls_embedding[:5]  # 只返回前5维以简化结果
    }

@app.get("/health")
async def health():
    return {"status": "ok", "worker_id": WORKER_ID}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 9001))
    uvicorn.run("w1-BERT:app", host="0.0.0.0", port=port)
