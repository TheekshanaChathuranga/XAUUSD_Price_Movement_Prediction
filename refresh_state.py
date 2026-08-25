from datetime import datetime
from typing import Optional


def build_refresh_status(
    is_refreshing: bool,
    refresh_started_at: Optional[datetime] = None,
    refresh_completed_at: Optional[datetime] = None,
    refresh_succeeded: Optional[bool] = None,
    refresh_error: Optional[str] = None,
    default_complete: bool = True,
) -> dict:
    """Return a backend refresh state payload for the UI.

    The UI should not treat a refresh as done until the backend has
    successfully completed and updated the cache. This helper makes that
    contract explicit and testable.
    """
    if is_refreshing:
        return {
            "refreshing_daily": True,
            "refresh_complete": False,
            "refresh_succeeded": False,
            "refresh_error": refresh_error,
        }

    if refresh_started_at is None:
        return {
            "refreshing_daily": False,
            "refresh_complete": default_complete,
            "refresh_succeeded": default_complete,
            "refresh_error": refresh_error,
        }

    completed = (
        refresh_completed_at is not None
        and refresh_completed_at >= refresh_started_at
    )
    resolved_succeeded = refresh_succeeded
    if resolved_succeeded is None:
        resolved_succeeded = completed
    return {
        "refreshing_daily": False,
        "refresh_complete": completed,
        "refresh_succeeded": resolved_succeeded,
        "refresh_error": refresh_error,
    }
