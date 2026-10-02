"""Sweep behavior that can be checked without an Ollama server."""

import contextlib
import io
import unittest

from benchmark import parse_args, sweep_length_order, verify_context
from config import Config
from metrics import summarize_sweep
from prompts import make_token_sweep_prefix, token_sweep_prompt


class SweepArgumentsTest(unittest.TestCase):
    def test_single_length_command_is_preserved(self):
        config = parse_args(["--prefix-lines", "80"])
        self.assertFalse(config.sweep)
        self.assertEqual(config.prefix_lines, 80)
        self.assertEqual(config.raw_csv, "results/raw_results.csv")
        self.assertEqual(config.summary_json, "results/summary.json")

    def test_sweep_defaults_and_custom_lengths(self):
        config = parse_args(["--sweep"])
        self.assertEqual(config.prefix_lengths, (256, 512, 1024, 2048, 4096, 6144))
        self.assertEqual(config.max_output_tokens, 16)
        self.assertIn("token_sweep_", config.raw_csv)
        self.assertIn("token_sweep_", config.summary_json)
        custom = parse_args(["--prefix-lengths", "12, 24", "--csv", "custom.csv"])
        self.assertTrue(custom.sweep)
        self.assertEqual(custom.prefix_lengths, (12, 24))
        self.assertEqual(custom.raw_csv, "custom.csv")

    def test_invalid_lengths_are_rejected(self):
        for value in ("", "20,", "20,0", "20,20", "x,40"):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--prefix-lengths", value])

    def test_context_budget_includes_output(self):
        config = Config(context_window=100, max_output_tokens=20)
        verify_context(80, config, "test")
        with self.assertRaisesRegex(RuntimeError, "exceeds num_ctx"):
            verify_context(81, config, "test")
        with self.assertRaisesRegex(RuntimeError, "exceeds num_ctx"):
            verify_context(80, config, "test", reserve=1)

    def test_counterbalanced_order_and_distinct_prompts(self):
        lengths = (256, 512, 1024, 2048, 4096, 6144)
        self.assertEqual(sweep_length_order(lengths, 1), lengths)
        self.assertEqual(sweep_length_order(lengths, 2), (1024, 2048, 4096, 6144, 256, 512))
        self.assertEqual(sweep_length_order(lengths, 3), (4096, 6144, 256, 512, 1024, 2048))
        a = make_token_sweep_prefix("a", 8)
        self.assertEqual(a, make_token_sweep_prefix("a", 8))
        self.assertNotEqual(a, make_token_sweep_prefix("b", 8))
        self.assertEqual(len({token_sweep_prompt(a, i) for i in range(1, 6)}), 5)


class SweepAggregationTest(unittest.TestCase):
    def test_lengths_and_scenarios_remain_separate(self):
        rows = []
        for length, baseline, shared in ((20, (4, 6), (2, 2)), (40, (10, 10), (5, 5))):
            for scenario, latencies in (("different_prefix", baseline), ("shared_prefix", shared)):
                for latency in latencies:
                    rows.append({
                        "prefix_lines": length, "scenario": scenario,
                        "prompt_tokens": length * 10,
                        "output_tokens": 8,
                        "cached_prompt_tokens": length if scenario == "shared_prefix" else 0,
                        "uncached_prompt_tokens": length * 9 if scenario == "shared_prefix" else length * 10,
                        "prefill_s": latency / 2, "ttft_s": latency / 2,
                        "generation_s": latency / 2, "wall_latency_s": latency,
                    })
        result = summarize_sweep(rows, (20, 40))
        self.assertEqual(result["20"]["different_prefix"]["wall_latency_s"]["median"], 5)
        self.assertEqual(result["40"]["different_prefix"]["wall_latency_s"]["median"], 10)
        self.assertEqual(result["20"]["shared_prefix"]["cached_prompt_tokens"]["mean"], 20)
        self.assertEqual(result["20"]["different_prefix"]["request_throughput_per_s"]["median"], (1/4 + 1/6)/2)
        self.assertEqual(result["20"]["different_prefix"]["aggregate_request_throughput_per_s"], 2/10)
        self.assertEqual(result["20"]["improvement_percent"]["ttft_s"]["median"], 60)
        self.assertEqual(result["40"]["improvement_percent"]["ttft_s"]["median"], 50)

    def test_cache_evidence_and_flagged_rows_are_retained(self):
        rows = []
        for scenario, cached_values in (("different_prefix", (0, 0)),
                                        ("shared_prefix", (180, 0))):
            for cached in cached_values:
                rows.append({
                    "prefix_target_tokens": 256, "scenario": scenario,
                    "prompt_tokens": 280, "cached_prompt_tokens": cached,
                    "uncached_prompt_tokens": 280 - cached,
                    "cached_fraction": cached / 280,
                    "output_tokens": 16, "short_output": False,
                    "cache_shortfall": scenario == "shared_prefix" and cached == 0,
                    "cache_drop_within_block": scenario == "shared_prefix" and cached == 0,
                    "prefill_s": 1, "ttft_s": 1.1, "generation_s": .2,
                    "wall_latency_s": 1.3,
                })
        result = summarize_sweep(rows, (256,))["256"]
        self.assertEqual(result["shared_prefix"]["prompt_tokens"]["count"], 2)
        self.assertEqual(result["shared_prefix"]["cache_shortfall_count"], 1)
        self.assertEqual(result["shared_prefix"]["cache_drop_count"], 1)
        self.assertFalse(result["cache_evidence"]["supports_prefix_reuse"])


if __name__ == "__main__":
    unittest.main()
