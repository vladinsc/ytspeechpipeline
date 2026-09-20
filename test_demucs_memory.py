from __future__ import annotations

import ast
import unittest
from pathlib import Path


class DemucsMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        source = Path(__file__).with_name("speech_pipeline.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        isolator = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "VocalIsolator"
        )
        isolate = next(
            node
            for node in isolator.body
            if isinstance(node, ast.FunctionDef) and node.name == "isolate"
        )
        cls.apply_call = next(
            node
            for node in ast.walk(isolate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "apply_model"
        )

    def test_full_recording_stays_on_cpu_for_demucs_accumulation(self) -> None:
        mix = self.apply_call.args[1]
        self.assertIsInstance(mix, ast.Subscript)
        self.assertIsInstance(mix.value, ast.Name)
        self.assertEqual(mix.value.id, "wav")
        self.assertFalse(
            any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "to"
                for node in ast.walk(mix)
            )
        )

    def test_demucs_chunks_still_run_on_configured_device(self) -> None:
        keywords = {item.arg: item.value for item in self.apply_call.keywords}
        self.assertEqual(ast.unparse(keywords["device"]), "self.device")
        self.assertIsInstance(keywords["split"], ast.Constant)
        self.assertIs(keywords["split"].value, True)


if __name__ == "__main__":
    unittest.main()
