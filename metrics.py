"""Statistics and honest handling of unavailable API fields."""

import math
import statistics


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def describe(rows: list[dict], field: str) -> dict:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": percentile(values, 0.95) if len(values) >= 20 else None,
    }


def improvement(baseline: float | None, shared: float | None, lower_is_better=True):
    if baseline is None or shared is None or baseline == 0:
        return None
    return 100 * ((baseline - shared) / baseline if lower_is_better else (shared - baseline) / baseline)


def summarize(rows: list[dict]) -> dict:
    fields = (
        "prompt_tokens", "cached_prompt_tokens", "uncached_prompt_tokens", "output_tokens", "ttft_s",
        "prefill_s", "generation_s", "api_total_s", "wall_latency_s",
        "prompt_tokens_per_s", "generation_tokens_per_s", "gpu_memory_mib",
    )
    groups = {scenario: [r for r in rows if r["scenario"] == scenario]
              for scenario in ("different_prefix", "shared_prefix")}
    stats = {scenario: {field: describe(group, field) for field in fields}
             for scenario, group in groups.items()}
    for scenario, group in groups.items():
        duration = sum(float(r["wall_latency_s"]) for r in group)
        stats[scenario]["request_throughput_per_s"] = len(group) / duration if duration else None
        stats[scenario]["end_to_end_output_tokens_per_s"] = (
            sum(int(r["output_tokens"] or 0) for r in group) / duration if duration else None
        )
    a, b = stats["different_prefix"], stats["shared_prefix"]
    stats["improvement_percent"] = {
        "prefill_median": improvement(a["prefill_s"]["median"], b["prefill_s"]["median"]),
        "latency_median": improvement(a["wall_latency_s"]["median"], b["wall_latency_s"]["median"]),
        "request_throughput": improvement(a["request_throughput_per_s"], b["request_throughput_per_s"], False),
    }
    return stats
