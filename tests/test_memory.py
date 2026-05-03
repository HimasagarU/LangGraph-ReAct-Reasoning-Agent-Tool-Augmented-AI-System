from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import memory


class MemoryTests(unittest.TestCase):
    def test_remember_interaction_prunes_old_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            memory_path = Path(temp_dir) / "memory.jsonl"

            with patch.object(memory, "_MEMORY_PATH", memory_path), patch.object(memory, "MAX_RECORDS", 2):
                memory.remember_interaction("query 1", "answer 1")
                memory.remember_interaction("query 2", "answer 2")
                memory.remember_interaction("query 3", "answer 3")

                contents = memory_path.read_text(encoding="utf-8").strip().splitlines()

        self.assertEqual(len(contents), 2)
        self.assertIn("query 2", contents[0])
        self.assertIn("query 3", contents[1])


if __name__ == "__main__":
    unittest.main()