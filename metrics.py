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


def summarize_sweep(rows: list[dict], prefix_lengths: tuple[int, ...]) -> dict:
    """Keep observations separate by configured prefix-token target and scenario."""
    fields = (
        "prompt_tokens", "cached_prompt_tokens", "uncached_prompt_tokens", "output_tokens",
        "cached_fraction", "prefill_s", "ttft_s", "generation_s", "wall_latency_s",
        "request_throughput_per_s",
    )
    result = {}
    for length in prefix_lengths:
        groups = {
            scenario: [row for row in rows if row.get("prefix_target_tokens", row.get("prefix_lines")) == length and row["scenario"] == scenario]
            for scenario in ("different_prefix", "shared_prefix")
        }
        stats = {}
        for scenario, group in groups.items():
            enriched = [
                {**row, "request_throughput_per_s": 1 / row["wall_latency_s"]
                 if row.get("wall_latency_s") else None}
                for row in group
            ]
            stats[scenario] = {field: describe(enriched, field) for field in fields}
            wall_sum = sum(row["wall_latency_s"] for row in group if row.get("wall_latency_s") is not None)
            stats[scenario]["aggregate_request_throughput_per_s"] = (
                len(group) / wall_sum if wall_sum else None
            )
            stats[scenario]["short_output_count"] = sum(bool(row.get("short_output")) for row in group)
            stats[scenario]["cache_shortfall_count"] = sum(bool(row.get("cache_shortfall")) for row in group)
            stats[scenario]["cache_drop_count"] = sum(bool(row.get("cache_drop_within_block")) for row in group)
        baseline = stats["different_prefix"]
        shared = stats["shared_prefix"]
        gains = {}
        for field in ("uncached_prompt_tokens", "prefill_s", "ttft_s", "generation_s", "wall_latency_s", "request_throughput_per_s"):
            gains[field] = {
                stat: improvement(
                    baseline[field][stat], shared[field][stat],
                    lower_is_better=field != "request_throughput_per_s",
                )
                for stat in ("median", "mean")
            }
        gains["aggregate_request_throughput_per_s"] = improvement(
            baseline["aggregate_request_throughput_per_s"],
            shared["aggregate_request_throughput_per_s"], False,
        )
        shared_rows = groups["shared_prefix"]
        baseline_rows = groups["different_prefix"]
        counts_available = all(row.get("cached_prompt_tokens") is not None for row in shared_rows + baseline_rows)
        shared_fraction = shared["cached_fraction"]["median"]
        baseline_fraction = baseline["cached_fraction"]["median"]
        cache_supported = bool(
            counts_available and shared_rows and baseline_rows
            and shared_fraction is not None and baseline_fraction is not None
            and shared_fraction >= 0.5 and shared_fraction > baseline_fraction
        )
        result[str(length)] = {
            **stats, "improvement_percent": gains,
            "cache_evidence": {
                "counts_available": counts_available,
                "supports_prefix_reuse": cache_supported,
                "shared_shortfall_count": shared["cache_shortfall_count"],
                "shared_drop_count": shared["cache_drop_count"],
                "interpretation": (
                    "Cached-token counts support substantial shared-prefix reuse."
                    if cache_supported else
                    "Timing differences alone do not establish shared-prefix reuse at this length."
                ),
            },
        }
    return result
