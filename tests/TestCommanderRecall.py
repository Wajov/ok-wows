"""验证选船后的悬停召回顺序和指挥官文字的原尺寸识别。"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import cv2
import numpy as np
from ok import Box

from src.tasks.AutoPveBattleTask import AutoPveBattleTask
from src.tasks.MyBaseTask import MyBaseTask
from src.tasks.feature_ocr import search_box
from tests.ocr_support import bind_ocr


class TestCommanderRecall(unittest.TestCase):
    def setUp(self):
        self.executor = MagicMock()
        self.executor.scene = None
        self.task = AutoPveBattleTask(self.executor, None)
        self.task.config = dict(self.task.default_config)

    def test_no_missing_commander_skips_mouse_and_recall(self):
        with patch.object(self.task, "find_one", return_value=None), \
                patch.object(self.task, "move") as move, \
                patch.object(self.task, "wait_click_feature") as click:
            self.assertTrue(self.task._recall_commander_if_needed())
        move.assert_not_called()
        click.assert_not_called()

    def test_hover_precedes_recall_and_timeout_reports_error(self):
        missing = Box(4755, 405, 134, 30, .99, "No-Commander")
        for clicked in (True, False):
            events = MagicMock()
            with self.subTest(clicked=clicked), \
                    patch.object(self.task, "find_one", return_value=missing), \
                    patch.object(self.task, "move", events.move), \
                    patch.object(self.task, "sleep", events.sleep), \
                    patch.object(self.task, "wait_click_feature", events.click), \
                    patch.object(self.task, "log_error") as error:
                events.click.return_value = clicked
                self.assertEqual(clicked, self.task._recall_commander_if_needed())
            self.assertEqual([call.move(*missing.center()), call.sleep(.5),
                              call.click("Recall-Commander", threshold=.8, time_out=5,
                                         raise_if_not_found=False, after_sleep=1)], events.mock_calls)
            self.assertEqual(not clicked, error.called)

    def test_recall_runs_after_ship_selection_and_before_other_preparation(self):
        for selected, recalled in ((False, True), (True, False), (True, True)):
            events = []
            def click(name, **kwargs):
                events.append(name)
                return selected if name == "Pick-First-Ship" else True
            def recall():
                events.append("recall")
                return recalled
            with self.subTest(selected=selected, recalled=recalled), \
                    patch.object(self.task, "wait_click_feature", side_effect=click), \
                    patch.object(self.task, "_recall_commander_if_needed", side_effect=recall), \
                    patch.object(self.task, "get_feature_by_name", return_value=object()), \
                    patch.object(self.task, "_return_to_main", return_value=True), \
                    patch.object(self.task, "_select_battle_mode", return_value=True), \
                    patch.object(self.task, "_prepare_loadout_page", return_value=True):
                self.assertEqual(selected and recalled, self.task._prepare_and_join_first_battle())
            self.assertEqual(["Pick-First-Ship"] if not selected else
                             ["Pick-First-Ship", "recall"] if not recalled else
                             ["Pick-First-Ship", "recall", "Join-Battle"], events)

    def test_commander_ocr_stays_near_annotation_and_does_not_resize_or_use_template(self):
        frame = np.zeros((2160, 5120, 3), dtype=np.uint8)
        self.executor.frame = frame
        bind_ocr(self.executor)
        engine = MagicMock()
        self.executor.ocr_lib.side_effect = lambda name="default": engine
        polygon = [[180, 50], [314, 50], [314, 80], [180, 80]]
        for name, text in (("No-Commander", "没有指挥官"), ("Recall-Commander", "召回指挥官")):
            engine.ocr.return_value = [[(polygon, (text, .99))]]
            region = search_box(name, frame)
            with self.subTest(name=name), \
                    patch.object(cv2, "resize", side_effect=AssertionError("No resizing")), \
                    patch.object(MyBaseTask, "find_one", side_effect=AssertionError("No template matching")):
                match = self.task.find_one(name)
            self.assertIsNotNone(match)
            crop = engine.ocr.call_args.args[0]
            self.assertTrue(np.shares_memory(crop, frame))
            self.assertEqual((region.height, region.width, 3), crop.shape)
            self.assertGreater(region.x, frame.shape[1] * .85)
            self.assertLess(region.y + region.height, frame.shape[0] * .3)

    @unittest.skipUnless(Path("ok_templates/21x9/30.png").is_file() and Path("ok_templates/21x9/31.png").is_file(),
                         "Commander reference screenshots unavailable")
    def test_real_screens_distinguish_missing_and_recall_from_assign(self):
        bind_ocr(self.executor)
        for filename, expected in (("30.png", "No-Commander"), ("31.png", "Recall-Commander")):
            self.executor.frame = cv2.imread("ok_templates/21x9/" + filename)
            for name in ("No-Commander", "Recall-Commander"):
                with self.subTest(filename=filename, name=name):
                    match = self.task.find_one(name)
                    self.assertEqual(name == expected, match is not None)
                    if match is not None:
                        self.assertGreaterEqual(match.confidence, .8)
                        self.assertGreater(match.x, 4700)


if __name__ == "__main__":
    unittest.main()
