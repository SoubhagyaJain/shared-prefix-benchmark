"""Small, explicit configuration for the local experiment."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5:7b"
    trials: int = 3
    requests_per_trial: int = 5
    prefix_lines: int = 105
    temperature: float = 0.0
    max_output_tokens: int = 80
    context_window: int = 8192
    timeout_s: int = 300
    keep_alive: str = "30m"
    raw_csv: str = "results/raw_results.csv"
    summary_json: str = "results/summary.json"
