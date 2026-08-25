import unittest
from datetime import datetime

from refresh_state import build_refresh_status


class RefreshStateTests(unittest.TestCase):
    def test_refresh_status_reports_failure_when_refresh_finishes_with_error(self):
        started = datetime(2026, 7, 14, 12, 0, 0)
        completed = datetime(2026, 7, 14, 12, 3, 0)

        status = build_refresh_status(
            is_refreshing=False,
            refresh_started_at=started,
            refresh_completed_at=completed,
            refresh_succeeded=False,
            refresh_error="daily refresh failed",
        )

        self.assertFalse(status["refreshing_daily"])
        self.assertTrue(status["refresh_complete"])
        self.assertFalse(status["refresh_succeeded"])
        self.assertEqual(status["refresh_error"], "daily refresh failed")


if __name__ == "__main__":
    unittest.main()
