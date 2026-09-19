import asyncio
import time

from starlette.concurrency import run_in_threadpool

from app.timing import timed, track_timings


def test_timed_is_a_noop_outside_track_timings() -> None:
    with timed("anything"):
        pass  # no active tracker: must neither raise nor record anywhere


def test_timed_accumulates_repeated_stages_in_milliseconds() -> None:
    with track_timings() as timings:
        with timed("rerank"):
            time.sleep(0.02)
        with timed("rerank"):
            time.sleep(0.02)
        with timed("weaviate"):
            pass

    assert set(timings) == {"rerank", "weaviate"}
    assert timings["rerank"] >= 40
    assert timings["weaviate"] < timings["rerank"]


def test_timed_records_a_stage_even_when_its_block_raises() -> None:
    with track_timings() as timings:
        try:
            with timed("graph"):
                raise ValueError("boom")
        except ValueError:
            pass

    assert "graph" in timings


def test_trackers_do_not_leak_between_turns() -> None:
    with track_timings() as first, timed("understand"):
        pass
    with track_timings() as second:
        pass

    assert "understand" in first
    assert second == {}


def test_timings_recorded_on_a_threadpool_thread_reach_the_tracker() -> None:
    def work() -> None:
        with timed("understand"):
            pass

    async def run() -> dict[str, float]:
        with track_timings() as timings:
            await run_in_threadpool(work)
        return timings

    assert "understand" in asyncio.run(run())
