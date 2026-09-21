# Architecture and measurement boundary

```text
Client requests (Python, sequential)
              |
              v
Shared system/context prefix + distinct question
              |
              v
Ollama tokenizer / prompt renderer (raw prompt)
              |
              v
Prefill: evaluate new prompt tokens
              |
              v
KV cache / reusable prefix state (if retained by runner)
              |
              v
Decode: generate one token at a time
              |
              v
Streamed response + final Ollama timing record
```

`benchmark.py` uses `/api/generate` with `raw=true` and streaming. The client records time to the first nonempty streamed response chunk and total wall time. The final Ollama record reports model time, load time, prompt and generated token counts, prompt evaluation duration, generation duration, and (on this local version) cached prompt tokens. Values documented by Ollama are in nanoseconds and are converted to seconds. The client samples whole-device GPU memory after each request through `nvidia-smi`; this is an approximate snapshot, not peak or model-only memory.

The experiment sends requests one at a time. Before measurement, two short requests warm the model and one long distinct prompt warms the long-context path. Before each shared-prefix block, an unmeasured request primes the shared prefix. Distinct-prefix requests have a different deterministic reference set per request; all sets have equal line counts and the same structure. Block order alternates each trial to reduce simple order effects. Data from all measured requests is saved immediately to CSV, and summary statistics are written to JSON at completion.

## What the caches mean

During prefill, the model processes the input prompt and stores each attention layer's key and value tensors for prior tokens. During decode, each new query can attend to these stored keys and values instead of recomputing them within that request. If the engine retains and matches those tensors across independent requests, an identical token prefix can also bypass most of its prefill. That cross-request reuse depends on exact token identity, model and context compatibility, cache retention and capacity, and the runner's matching behavior. A warm model in VRAM is a separate effect; it avoids load time without proving reuse of prompt computation.

The `prompt_eval_cached_count` field is especially useful evidence when present, but the public API does not reveal every internal cache decision or hit source. The benchmark therefore calls observed speedup "consistent with cache reuse" when cached tokens are reported, while avoiding claims about a specific internal algorithm. Without that field, latency alone is insufficient to establish cross-request KV reuse.

## Source documentation

- [Ollama generate API](https://docs.ollama.com/api/generate) describes streaming and final timing fields.
- [Ollama API reference source](https://github.com/ollama/ollama/blob/main/docs/api.md) includes `prompt_eval_cached_count` and identifies prompt evaluation duration as work on uncached tokens. Check the installed Ollama version: older versions may omit the cached count.
