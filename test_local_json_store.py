from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from yt_transcriber_api import JobRequest, JobStore


class LocalJsonJobStoreTests(unittest.TestCase):
    def test_jobs_survive_restart_and_processing_is_requeued(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            jobs_path = root / "results" / "api_jobs.json"
            with (
                patch("yt_transcriber_api.RESULTS_DIR", root / "results"),
                patch("yt_transcriber_api.WORK_DIR", root / "work"),
            ):
                store = JobStore(jobs_path)
                job = store.create(
                    JobRequest(
                        url="https://www.youtube.com/watch?v=localjson123",
                        label="kids",
                    )
                )
                store.update(job["job_id"], status="processing")
                restarted = JobStore(jobs_path)
                restored = restarted.get(job["job_id"])

            self.assertEqual(restored["status"], "queued")
            self.assertEqual(restored["stage_status"], "restart_queued")
            self.assertTrue(jobs_path.exists())
            self.assertEqual(restarted.data["summary"]["queued"], 1)


if __name__ == "__main__":
    unittest.main()
