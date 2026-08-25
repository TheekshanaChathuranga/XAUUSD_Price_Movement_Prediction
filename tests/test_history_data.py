import os
import tempfile
import unittest

import pandas as pd

from history_data import build_history_payload, load_history_rows


class HistoryDataTests(unittest.TestCase):
    def test_load_history_rows_expands_to_requested_sample_size(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trade_log_path = os.path.join(tmpdir, "backtest_trade_log.csv")
            pd.DataFrame(
                [
                    {
                        "Date": "2026-01-01",
                        "Direction": "LONG",
                        "Pos_Size": 0.5,
                        "Probability": 0.61,
                        "Market_Return": 0.01,
                        "Gross_Return": 0.008,
                        "Net_Return": 0.007,
                        "Trans_Cost": 0.0,
                        "Cumulative_Value": 10007.0,
                        "Win": 1,
                    },
                    {
                        "Date": "2026-01-02",
                        "Direction": "SHORT",
                        "Pos_Size": -0.4,
                        "Probability": 0.58,
                        "Market_Return": -0.01,
                        "Gross_Return": -0.009,
                        "Net_Return": -0.008,
                        "Trans_Cost": 0.0,
                        "Cumulative_Value": 9998.0,
                        "Win": 0,
                    },
                ]
            ).to_csv(trade_log_path, index=False)

            rows = load_history_rows(tmpdir, limit=5)

            self.assertEqual(len(rows), 2)
            self.assertTrue(all("date" in row for row in rows))
            self.assertIn(rows[0]["signal"], {"LONG", "SHORT"})

    def test_build_history_payload_includes_summary_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            trade_log_path = os.path.join(tmpdir, "backtest_trade_log.csv")
            pd.DataFrame(
                [
                    {
                        "Date": "2026-01-01",
                        "Direction": "LONG",
                        "Pos_Size": 0.5,
                        "Probability": 0.61,
                        "Market_Return": 0.01,
                        "Gross_Return": 0.008,
                        "Net_Return": 0.007,
                        "Trans_Cost": 0.0,
                        "Cumulative_Value": 10007.0,
                        "Win": 1,
                    },
                    {
                        "Date": "2026-01-02",
                        "Direction": "SHORT",
                        "Pos_Size": -0.4,
                        "Probability": 0.58,
                        "Market_Return": -0.01,
                        "Gross_Return": -0.009,
                        "Net_Return": -0.008,
                        "Trans_Cost": 0.0,
                        "Cumulative_Value": 9998.0,
                        "Win": 0,
                    },
                ]
            ).to_csv(trade_log_path, index=False)

            payload = build_history_payload(tmpdir, limit=5)

            self.assertEqual(payload["count"], 2)
            self.assertIn("summary", payload)
            self.assertIn("win_rate", payload["summary"])


if __name__ == "__main__":
    unittest.main()
