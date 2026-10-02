"""Paired cold-prefix cumulative latency calculations."""

import statistics


METRICS = {"wall": "wall_latency_s", "prompt_eval": "prefill_s"}
SCENARIOS = ("different_prefix", "shared_prefix")


def curve_point(trial, request_count, totals):
    point = {"trial": trial, "request_count": request_count}
    for name in METRICS:
        distinct = totals["different_prefix"][name]
        shared = totals["shared_prefix"][name]
        savings = distinct - shared
        point.update({
            f"cumulative_{name}_distinct_s": distinct,
            f"cumulative_{name}_shared_s": shared,
            f"{name}_savings_s": savings,
            f"{name}_savings_percent": 100 * savings / distinct if distinct else None,
            f"amortized_{name}_distinct_s": distinct / request_count,
            f"amortized_{name}_shared_s": shared / request_count,
        })
    return point


def first_break_even(points, name):
    return next((
        point["request_count"] for point in points
        if point[f"cumulative_{name}_shared_s"] < point[f"cumulative_{name}_distinct_s"]
    ), None)


def summarize_break_even(rows: list[dict], trials: int, max_requests: int) -> dict:
    """Pair equal-index requests, then aggregate each trial's cumulative curve."""
    indexed = {}
    for row in rows:
        key = (row["trial"], row["scenario"], row["request_index"])
        if key in indexed:
            raise ValueError(f"duplicate request: {key}")
        indexed[key] = row

    per_trial = []
    trial_break_even = []
    for trial in range(1, trials + 1):
        totals = {scenario: {name: 0.0 for name in METRICS} for scenario in SCENARIOS}
        trial_points = []
        for request_count in range(1, max_requests + 1):
            for scenario in SCENARIOS:
                key = (trial, scenario, request_count)
                if key not in indexed:
                    raise ValueError(f"missing request: {key}")
                for name, field in METRICS.items():
                    value = indexed[key].get(field)
                    if value is None or value < 0:
                        raise ValueError(f"missing or invalid {field}: {key}")
                    totals[scenario][name] += value
            trial_points.append(curve_point(trial, request_count, totals))
        per_trial.extend(trial_points)
        trial_break_even.append({
            "trial": trial,
            "wall_first_n": first_break_even(trial_points, "wall"),
            "prompt_eval_first_n": first_break_even(trial_points, "prompt_eval"),
        })
    if len(indexed) != trials * max_requests * len(SCENARIOS):
        raise ValueError("request rows contain unexpected trial, scenario, or index values")

    aggregate = []
    for request_count in range(1, max_requests + 1):
        peers = [point for point in per_trial if point["request_count"] == request_count]
        totals = {
            scenario: {
                name: statistics.mean(point[f"cumulative_{name}_{'shared' if scenario == 'shared_prefix' else 'distinct'}_s"] for point in peers)
                for name in METRICS
            }
            for scenario in SCENARIOS
        }
        point = curve_point("mean", request_count, totals)
        for name in METRICS:
            for scenario in ("shared", "distinct"):
                point[f"median_cumulative_{name}_{scenario}_s"] = statistics.median(
                    peer[f"cumulative_{name}_{scenario}_s"] for peer in peers
                )
        aggregate.append(point)

    first = aggregate[0]
    return {
        "per_trial_curves": per_trial,
        "by_request_count": {str(point["request_count"]): point for point in aggregate},
        "break_even": {
            "wall_first_n": first_break_even(aggregate, "wall"),
            "prompt_eval_first_n": first_break_even(aggregate, "prompt_eval"),
            "basis": "strictly lower mean cumulative time across paired trials",
            "measured_through_n": max_requests,
        },
        "per_trial_break_even": trial_break_even,
        "first_request_extra_s": {
            "wall": -first["wall_savings_s"],
            "prompt_eval": -first["prompt_eval_savings_s"],
        },
    }
