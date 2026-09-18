from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from batch_pipeline import (
    default_checkpoint_path,
    load_or_create_checkpoint,
    shard_manifest,
)


class BatchShardingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.videos = [
            {"url": f"https://www.youtube.com/watch?v=video{i:02d}", "label": "kids"}
            for i in range(1, 11)
        ]

    def test_round_robin_shards_are_complete_and_disjoint(self) -> None:
        shards = [shard_manifest(self.videos, 4, index) for index in range(4)]

        self.assertEqual(
            [[video["manifest_index"] for video in shard] for shard in shards],
            [[1, 5, 9], [2, 6, 10], [3, 7], [4, 8]],
        )
        assigned = [video["manifest_index"] for shard in shards for video in shard]
        self.assertEqual(sorted(assigned), list(range(1, 11)))
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_checkpoint_uses_global_indexes_for_output_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shard = shard_manifest(self.videos, 4, 2)
            checkpoint = load_or_create_checkpoint(
                root / "checkpoint.json",
                shard,
                root / "videos.txt",
                root / "results",
                root / "work",
            )

        self.assertEqual([entry["index"] for entry in checkpoint["videos"]], [3, 7])
        self.assertIn("video_0003_", checkpoint["videos"][0]["output_path"])
        self.assertIn("video_0007_", checkpoint["videos"][1]["output_path"])

    def test_invalid_shard_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "shard-count"):
            shard_manifest(self.videos, 0, 0)
        with self.assertRaisesRegex(ValueError, "shard-index"):
            shard_manifest(self.videos, 4, 4)
        with self.assertRaisesRegex(ValueError, "shard-index"):
            shard_manifest(self.videos, 4, -1)

    def test_parallel_shards_get_distinct_default_checkpoints(self) -> None:
        output = Path("results")
        paths = [default_checkpoint_path(output, 4, index) for index in range(4)]

        self.assertEqual(len(set(paths)), 4)
        self.assertEqual(paths[2], output / "batch_checkpoint_shard_2_of_4.json")
        self.assertEqual(
            default_checkpoint_path(output, 1, 0), output / "batch_checkpoint.json"
        )


if __name__ == "__main__":
    unittest.main()
