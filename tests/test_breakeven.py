"""Cold-prefix break-even calculations without Ollama."""

import contextlib
import io
import unittest

from benchmark import parse_args
from breakeven import summarize_break_even


def rows_for_trial(trial, distinct_wall, shared_wall, distinct_prefill, shared_prefill):
    rows = []
    for scenario, wall_values, prefill_values in (
        ("different_prefix", distinct_wall, distinct_prefill),
        ("shared_prefix", shared_wall, shared_prefill),
    ):
        for index, (wall, prefill) in enumerate(zip(wall_values, prefill_values), 1):
            rows.append({"trial": trial, "scenario": scenario, "request_index": index,
                         "wall_latency_s": wall, "prefill_s": prefill})
    return rows


class BreakEvenArgumentsTest(unittest.TestCase):
    def test_mode_has_separate_outputs(self):
        config = parse_args(["--break-even"])
        self.assertTrue(config.break_even)
        self.assertEqual(config.max_requests_per_trial, 8)
        self.assertEqual(config.raw_csv, "results/breakeven_raw_results.csv")
        self.assertEqual(config.summary_json, "results/breakeven_summary.json")
        self.assertEqual(config.cumulative_csv, "results/breakeven_cumulative.csv")
        self.assertEqual(config.chart_svg, "results/breakeven_chart.svg")
        custom = parse_args(["--max-requests", "3", "--trials", "2"])
        self.assertTrue(custom.break_even)
        self.assertEqual(custom.max_requests_per_trial, 3)
        self.assertEqual(custom.trials, 2)

    def test_invalid_mode_combinations(self):
        for argv in (["--max-requests", "0"], ["--sweep", "--break-even"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(argv)


class BreakEvenCalculationTest(unittest.TestCase):
    def test_paired_cumulative_savings_and_distinct_break_even_points(self):
        rows = [
            *rows_for_trial(1, [10, 10, 10], [15, 8, 5], [6, 6, 6], [9, 2, 2]),
            *rows_for_trial(2, [10, 10, 10], [13, 9, 5], [6, 6, 6], [8, 3, 2]),
        ]
        result = summarize_break_even(rows, trials=2, max_requests=3)
        self.assertEqual(result["break_even"]["wall_first_n"], 3)
        self.assertEqual(result["break_even"]["prompt_eval_first_n"], 2)
        self.assertEqual(result["first_request_extra_s"]["wall"], 4)
        third = result["by_request_count"]["3"]
        self.assertEqual(third["cumulative_wall_distinct_s"], 30)
        self.assertEqual(third["cumulative_wall_shared_s"], 27.5)
        self.assertEqual(third["wall_savings_s"], 2.5)
        self.assertAlmostEqual(third["wall_savings_percent"], 100 * 2.5 / 30)
        self.assertAlmostEqual(third["amortized_wall_shared_s"], 27.5 / 3)
        self.assertEqual(result["per_trial_break_even"][0]["wall_first_n"], 3)

    def test_tie_does_not_count_and_no_break_even_is_reported(self):
        rows = [
            *rows_for_trial(1, [10, 10, 10], [12, 8, 10], [5, 5, 5], [6, 4, 5]),
            *rows_for_trial(2, [10, 10, 10], [12, 8, 10], [5, 5, 5], [6, 4, 5]),
        ]
        result = summarize_break_even(rows, trials=2, max_requests=3)
        self.assertIsNone(result["break_even"]["wall_first_n"])
        self.assertIsNone(result["break_even"]["prompt_eval_first_n"])
        self.assertEqual(result["by_request_count"]["2"]["wall_savings_s"], 0)


if __name__ == "__main__":
    unittest.main()
