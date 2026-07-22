from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import asyncio
import json
import logging
import random
import time
from typing import AsyncGenerator

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mock-vllm-server")

# Sample Urdu responses
URDU_RESPONSES = [
    "ہم کسٹم سافٹ ویئر ڈویلپمنٹ، ویب سائٹ ڈیزائن اور ڈویلپمنٹ، موبائل ایپلیکیشن ڈویلپمنٹ، سی آر ایم سسٹمز، بزنس اور مینجمنٹ سسٹمز، بزنس آٹومیشن، ایس ای او اور ڈیجیٹل مارکیٹنگ کی خدمات فراہم کرتے ہیں۔",
    "کوٹیشن تیار کرنے کے لیے ہمیں آپ کے کاروبار کی نوعیت، مطلوبہ فیچرز، صارفین کی تعداد، اور دیگر ضروریات کے بارے میں جاننا ہوگا۔",
    "ہماری ٹیم آپ کے ساتھ رابطہ کر کے آپ کی ضروریات کا جائزہ لے گی اور پھر ایک باضابطہ کوٹیشن تیار کرے گی۔",
    "ہماری خدمات کی قیمت آپ کی ضروریات کے مطابق مختلف ہوتی ہے۔ ہم آپ کے ساتھ تفصیلی بات چیت کے بعد ہی کوئی قیمت بتا سکتے ہیں۔",
    "ہم آپ کے موجودہ سسٹم کا جائزہ لے سکتے ہیں۔ اس کے لیے ہمیں آپ کے سسٹم تک رسائی اور مطلوبہ تبدیلیوں کے بارے میں جاننا ہوگا۔"
]

async def generate_streaming_response(prompt: str) -> AsyncGenerator[str, None]:
    """Generate a streaming response for the given prompt."""
    # Extract RAG content if present
    rag_content = ""
    if "Relevant Knowledge Base Information:" in prompt:
        rag_start = prompt.find("Relevant Knowledge Base Information:")
        rag_end = prompt.find("USER:", rag_start)
        rag_content = prompt[rag_start:rag_end].strip()
        logger.info(f"Extracted RAG content: {rag_content[:100]}...")

    # Select a response based on the prompt
    response = random.choice(URDU_RESPONSES)

    # Simulate streaming by sending the response in chunks
    words = response.split()
    for i in range(0, len(words), 3):
        chunk = " ".join(words[i:i+3])
        if chunk:
            yield json.dumps({"text": chunk}) + "\n"
            await asyncio.sleep(0.05)  # Simulate processing time

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests with streaming."""
    data = await request.json()
    prompt = data.get("prompt", "")
    logger.info(f"Received prompt: {prompt[:100]}...")

    async def generate():
        async for chunk in generate_streaming_response(prompt):
            yield chunk

    return StreamingResponse(generate(), media_type="application/x-ndjson")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting mock vLLM server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8001)