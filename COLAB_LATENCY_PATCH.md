# Colab latency patch

The deployed LLM is inside the notebook. Measure it with CUDA synchronization;
otherwise asynchronous GPU timings are wrong.

```python
import time
def generate(messages, deterministic=True, max_new_tokens=120):
    t0 = time.perf_counter()
    prompt = text_tokenizer.apply_chat_template(messages, tokenize=True,
        add_generation_prompt=True, return_tensors="pt",
        enable_thinking=False).to(model.device)
    torch.cuda.synchronize(); prompt_ready = time.perf_counter()
    kwargs = dict(max_new_tokens=max_new_tokens, repetition_penalty=1.08,
        logits_processor=[URDU_ONLY_PROCESSOR], eos_token_id=IM_END_ID,
        pad_token_id=text_tokenizer.pad_token_id, use_cache=True,
        do_sample=not deterministic)
    if not deterministic: kwargs.update(temperature=0.7, top_p=0.9)
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    with torch.inference_mode(): out = model.generate(prompt, **kwargs)
    end.record(); torch.cuda.synchronize()
    ids = out[0, prompt.shape[1]:]; generation_ms = start.elapsed_time(end)
    metrics = {"prompt_tokens": int(prompt.shape[1]),
      "output_tokens": int(ids.numel()), "generation_ms": round(generation_ms,2),
      "tokens_per_second": round(ids.numel()/(generation_ms/1000),2),
      "prompt_construction_ms": round((prompt_ready-t0)*1000,2),
      "cuda_active": torch.cuda.is_available() and next(model.parameters()).is_cuda,
      "gpu_name": torch.cuda.get_device_name(0),
      "vram_allocated_mb": round(torch.cuda.memory_allocated()/1048576,1),
      "vram_reserved_mb": round(torch.cuda.memory_reserved()/1048576,1)}
    text = text_tokenizer.decode(ids, skip_special_tokens=True).strip()
    return text, IM_END_ID in ids.tolist(), metrics
```

Propagate metrics through `generate_agent_reply`, including
`generation_attempts`, and return them from Flask:

```python
@app.route('/chat', methods=['POST'])
def chat_endpoint():
    started=time.perf_counter(); data=request.get_json(force=True)
    reply, metrics=generate_agent_reply(data.get('history', []))
    metrics['server_total_ms']=round((time.perf_counter()-started)*1000,2)
    metrics['turn_id']=data.get('turn_id')
    return jsonify(reply=reply, metrics=metrics)
```

Also time `FastLanguageModel.from_pretrained` once and log
`next(model.parameters()).is_cuda`, GPU name, dtype, quantization and
`MAX_SEQ_LENGTH`. Remove `demo.launch(share=True, debug=True)` in production.
The current endpoint cannot expose real TTFT because `model.generate` and JSON
both buffer the answer; use `TextIteratorStreamer` or vLLM/SGLang for streaming.
