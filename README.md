<img width="1911" height="1044" alt="Screenshot 2026-09-21 090014" src="https://github.com/user-attachments/assets/f9d78c6b-c83e-4019-8c63-9969efd63585" />



# Shared-Prefix LLM Inference Benchmark

A small local ML systems experiment on whether repeated long prompt prefixes change inference performance on Ollama. It runs on Windows with the installed `qwen2.5:7b`, records every request, and treats cache attribution separately from model warm-up.

## Problem and hypothesis

Long contexts can make **prefill** expensive before the first answer token appears. If a serving engine retains attention state for a token-identical prefix between independent requests, a shared-prefix request should evaluate fewer new prompt tokens and reduce prompt evaluation time and often time to first token (TTFT). A faster second request alone does not prove prefix caching; model loading, kernel warm-up, scheduling, GPU clocks, and OS caching can also change timing.

## Run on Windows

Open PowerShell in this folder. Start the Ollama app (or run `ollama serve` in another terminal), confirm `ollama list` includes `qwen2.5:7b`, and run:

```powershell
python --version
ollama --version
ollama ps
python benchmark.py
```

No `pip install` is required; Python's standard library is sufficient. The default run uses three trials with five requests per scenario in each trial (30 measured requests), plus model warm-up and shared-prefix priming. It may take several minutes on a 6 GB laptop GPU because the 7B model may split across CPU and GPU. A shorter smoke run is `python benchmark.py --trials 1 --requests 2`; write its output to other paths if you want to keep the full run's results. CLI options include `--model`, `--url`, `--prefix-lines`, `--num-ctx`, `--max-output-tokens`, `--csv`, and `--summary`.

## Experiment

| Control | Value |
| --- | --- |
| Model / server | Local `qwen2.5:7b` / Ollama |
| Generation | Temperature 0, at most 80 output tokens |
| Context | 8,192 tokens; deterministic synthetic reference sets |
| Workload | 5 requests per scenario per trial, 3 trials |
| Concurrency | 1, sequential |
| Warm-up | 2 unmeasured short requests and 1 long request; shared prefix primed before each shared block |
| Order | Different/shared in odd trials; shared/different in even trials |

**Different prefixes:** Every request receives a different long reference set and a short question. **Shared prefix:** Every request receives the same long reference set and a distinct short question. Each run has a random identifier embedded in its reference sets so an earlier run cannot prime this run's baseline. All reference sets use the same template and line count, so actual prompt token counts can be checked in the CSV rather than assumed equal. `raw=true` avoids model chat template changes between requests. The common prefix is primed outside the measured shared requests; this experiment measures a warm-prefix workload, including the cost of priming only in the unmeasured setup.

## Architecture

```text
Client Requests
      |
      v
Shared System / Context Prefix + Question
      |
      v
Tokenizer
      |
      v
Prefill
      |
      v
KV Cache / Prefix Cache
      |
      v
Decode
      |
      v
Response
```

See [architecture.md](docs/architecture.md) for measurement details.

## Metrics

- **TTFT:** Client wall time from request start to the first nonempty streamed response chunk. Includes network and scheduling overhead.
- **Prompt evaluation:** Ollama `prompt_eval_duration`, the time spent evaluating uncached prompt tokens on current API versions. `prompt_eval_count` is full prompt length; `prompt_eval_cached_count` reports tokens read from cache when exposed.
- **Generation:** Ollama `eval_duration` and `eval_count`; decode tokens per second is their ratio.
- **Total:** Client wall latency and Ollama `total_duration` are both recorded. The two need not match because their timing boundaries differ.
- **Throughput:** Requests per second over the sum of measured request wall times, for each scenario. Prompt tokens per second divides uncached prompt tokens by prompt evaluation time; it is unavailable when the API omits the cached-token count.
- **GPU memory:** Whole-device `nvidia-smi` used-memory snapshot after each request, if available. It is neither model-only usage nor peak allocation.

`results/raw_results.csv` holds request-level data. `results/summary.json` holds mean and median for each metric. P95 is emitted only with at least 20 observations in a scenario; five or fifteen points cannot support a useful tail estimate. Positive improvement percentages favor shared prefix.

## Results

Measured locally on 21 September 2026 with Ollama 0.34.2, `qwen2.5:7b`, Windows, an RTX 4050 Laptop GPU (6 GB VRAM), and 16 GB system RAM. Ollama reported the model split as 24% CPU / 76% GPU at an 8,192-token context. Each scenario has 15 measured requests across three trials; input prompts ranged from 2,818 to 2,849 tokens. The shared block was primed before measurement, so these results describe repeated use of an already warm prefix.

| Metric | Different prefixes | Shared prefix | Change favoring shared |
| --- | ---: | ---: | ---: |
| Median prompt tokens | 2,834 | 2,818 | — |
| Median cached prompt tokens | 22 | 2,788 | — |
| Median uncached prompt tokens | 2,818 | 30 | — |
| Median prompt evaluation | 2.534 s | 0.559 s | 77.9% less |
| Median TTFT | 2.653 s | 0.585 s | 77.9% less |
| Median wall latency | 7.284 s | 5.529 s | 24.1% less |
| Mean wall latency | 7.357 s | 5.354 s | — |
| Mean generation duration | 4.731 s | 4.826 s | — |
| Request throughput | 0.136 req/s | 0.187 req/s | 37.4% more |
| Median whole-device GPU memory snapshot | 4,613 MiB | 4,612 MiB | — |

The measured cached-token gap is direct API evidence that this Ollama run reused most of the shared prefix across independent requests. The smaller change in total wall latency is expected because generation still took roughly 4.7–4.8 seconds on average. Uncached prompt tokens per second was lower in the shared case (median 54 versus 1,110) because the fixed overhead of evaluating about 30 remaining tokens dominates that small denominator; it is not a sign that shared prefill did more work. The run had 15 observations per scenario, so P95 is intentionally absent. See [raw measurements](results/raw_results.csv) and [summary statistics](results/summary.json) for request-level values, means, and medians.

## Interpretation and limitations

Input processing is called *prefill* because the model fills attention KV state for the prompt before autoregressive decode begins. A long prompt can delay TTFT even if the answer is short: the full input must be processed before the first output token. The KV cache stores per-layer key and value tensors for processed tokens. Identical prefixes can theoretically reuse that state, skipping repeated attention computation for those tokens. This tends to affect prefill and TTFT more directly than per-token decode speed, since every new output token still needs a forward step.

This matters when RAG requests repeat a large retrieved context, agents repeat tool instructions or history, long system prompts repeat policy text, few-shot prompts repeat examples, and multi-user production serving has a common system prompt. Reuse requires the same model and tokenized prefix, a compatible context and runner, enough cache capacity, and requests that reach the retained state before eviction. Prompt text that looks similar may tokenize differently; divergent system templates or metadata can break a match. On this machine, the 7B model partly ran on CPU, so GPU memory and latency reflect that hardware split. Small samples, thermal drift, background activity, and the excluded priming cost also limit generalization.

**Conclusion:** This Ollama 0.34.2 run showed a large cached-token count and shorter prompt evaluation for identical prefixes, consistent with cross-request KV/prefix reuse. The result establishes the behavior of this local model, runner, hardware, and warm-prefix workload; it does not establish universal performance across Ollama versions or a specific internal caching algorithm. A zero or missing cached-token count in another run should trigger an investigation of engine behavior, cache settings, prefix identity, retention, and measurement noise. Repeating the same workload under an engine with an explicit prefix-cache hit metric, such as vLLM, would provide a useful independent comparison.

## Interview explanation

“I built a controlled local inference benchmark that compares equal-sized long prompts with unique versus identical prefixes. It streams from Ollama so I can measure client TTFT, and it records server prompt/decode timings and cached-token counts. I warm the model, prime the shared context deliberately, alternate scenario order, and save per-request data so any claimed speedup is tied to evidence of prompt reuse rather than a faster warm model.”

## Five-sentence result explanation

I benchmarked 30 sequential requests to a local `qwen2.5:7b` model on Ollama 0.34.2, comparing distinct roughly 2,800-token prefixes with a repeated prefix of similar length. The shared case cut median prompt evaluation from 2.534 seconds to 0.559 seconds and median TTFT from 2.653 seconds to 0.585 seconds, while median wall latency fell from 7.284 seconds to 5.529 seconds. Ollama reported a median of 2,788 cached prompt tokens for shared requests versus 22 for distinct requests, so the prefill reduction is consistent with cross-request prefix reuse in this run. This demonstrates how retaining KV state for identical prompt tokens can remove repeated prefill work while generation time remains similar. In production RAG, agent, and multi-user systems, the same mechanism can improve response latency and throughput when requests reuse a stable context.
