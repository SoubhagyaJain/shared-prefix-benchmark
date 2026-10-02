"""Sweep behavior that can be checked without an Ollama server."""

import contextlib
import io
import unittest

from benchmark import parse_args, verify_context
from config import Config
from metrics import summarize_sweep


class SweepArgumentsTest(unittest.TestCase):
    def test_single_length_command_is_preserved(self):
        config = parse_args(["--prefix-lines", "80"])
        self.assertFalse(config.sweep)
        self.assertEqual(config.prefix_lines, 80)
        self.assertEqual(config.raw_csv, "results/raw_results.csv")
        self.assertEqual(config.summary_json, "results/summary.json")

    def test_sweep_defaults_and_custom_lengths(self):
        config = parse_args(["--sweep"])
        self.assertEqual(config.prefix_lengths, (20, 40, 80, 105, 160, 220))
        self.assertEqual(config.raw_csv, "results/sweep_raw_results.csv")
        self.assertEqual(config.summary_json, "results/sweep_summary.json")
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


if __name__ == "__main__":
    unittest.main()
