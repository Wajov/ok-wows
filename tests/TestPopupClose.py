import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from ok import Box, FeatureSet

from src.tasks.AutoPveBattleTask import AutoPveBattleTask


class TestPopupClose(unittest.TestCase):
    def setUp(self):
        self.executor = MagicMock()
        self.executor.scene = None
        self.task = AutoPveBattleTask(self.executor, None)
        self.task.config = dict(self.task.default_config)

    def test_popup_takes_priority_over_underlying_port_and_battle(self):
        for underlying in ("Join-Battle", "Control-Camera"):
            visible = {"Popup-ESC-Close", underlying, "Select-Battle-Mode"}
            with self.subTest(underlying=underlying), patch.object(self.task, "next_frame"), patch.object(
                    self.task, "find_one", side_effect=lambda name, **kwargs: Box(1, 1, 20, 20) if name in visible else None):
                self.assertEqual("popup", self.task._detect_scene())

    def test_startup_closes_popup_and_rechecks_port(self):
        with patch.object(self.task, "_detect_scene", side_effect=["popup", "main"]), patch.object(self.task, "send_key") as key:
            self.assertTrue(self.task._return_to_main())
        key.assert_called_once_with("esc", after_sleep=1)

    def test_battle_closes_popup_and_rechecks_result(self):
        with patch.object(self.task, "_detect_scene", side_effect=["popup", "result"]), patch.object(self.task, "send_key") as key, patch.object(self.task, "_send_battle_action") as fire:
            self.assertTrue(self.task._run_until_result())
        key.assert_called_once_with("esc", after_sleep=1)
        fire.assert_not_called()

    def test_template_matches_relocated_close_button_but_not_plain_esc(self):
        root = Path("assets/21x9")
        data = json.loads((root / "coco_annotations.json").read_text(encoding="utf-8"))
        cid = next(c["id"] for c in data["categories"] if c["name"] == "Popup-ESC-Close")
        ann = next(a for a in data["annotations"] if a["category_id"] == cid)
        image = next(i for i in data["images"] if i["id"] == ann["image_id"])
        original = cv2.imread(str(root / image["file_name"]))
        x, y, w, h = ann["bbox"]
        template = original[y:y+h, x:x+w]
        features = FeatureSet(False, str(root / "coco_annotations.json"), 0.01, 0.01, 0.9)
        for left, top in ((x, y), (3540, 820)):
            frame = np.zeros_like(original)
            frame[top:top+h, left:left+w] = template
            self.assertTrue(features.find_feature(frame, "Popup-ESC-Close", horizontal_variance=1, vertical_variance=1, threshold=0.9, limit=1))
        frame = np.zeros_like(original)
        frame[y:y+h, x:x+47] = template[:, :47]
        self.assertFalse(features.find_feature(frame, "Popup-ESC-Close", horizontal_variance=1, vertical_variance=1, threshold=0.9, limit=1))


if __name__ == "__main__":
    unittest.main()
