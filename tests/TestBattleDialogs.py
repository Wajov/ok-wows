from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from ok import Box, FeatureSet

from src.tasks.AutoPveBattleTask import AutoPveBattleTask
from src.tasks.feature_ocr import MODAL_CLOSE_TITLES, search_box
from tests.ocr_support import bind_ocr


class TestBattleDialogs(unittest.TestCase):
    CASES = (("survey.png", "battle_survey", "Battle-Survey-Close"),
             ("inactivity.png", "inactivity_penalty", "Inactivity-Penalty-Close"))

    def setUp(self):
        self.executor = MagicMock()
        self.executor.scene = None
        self.task = AutoPveBattleTask(self.executor, None)
        self.task.config = dict(self.task.default_config)

    def bind_frame(self, frame):
        bind_ocr(self.executor)
        self.executor.frame = frame
        self.executor.method.height, self.executor.method.width = frame.shape[:2]
        self.executor.feature_set = FeatureSet(False, "assets/21x9/coco_annotations.json", .01, .01, .8)

    def test_real_dialogs_match_close_button_at_full_and_half_resolution(self):
        for filename, scene, feature in self.CASES:
            original = cv2.imread(str(Path("tests/fixtures/battle_dialogs") / filename))
            for scale in (1, .5):
                frame = cv2.resize(original, None, fx=scale, fy=scale) if scale != 1 else original
                with self.subTest(dialog=filename, scale=scale):
                    self.bind_frame(frame)
                    button = self.task.find_one(feature)
                    self.assertIsNotNone(button)
                    self.assertEqual(scene, self.task._detect_scene(refresh=False))
                    x, y = button.center()
                    h, w = frame.shape[:2]
                    self.assertTrue(.49 < x / w < .55)  # 问卷的提交按钮在左侧，禁止误点提交。
                    self.assertTrue(.49 < y / h < .54)
                    other = "Inactivity-Penalty-Close" if scene == "battle_survey" else "Battle-Survey-Close"
                    self.assertIsNone(self.task.find_one(other))

    def test_generic_close_without_title_is_not_a_dialog(self):
        for filename, _, feature in self.CASES:
            frame = cv2.imread(str(Path("tests/fixtures/battle_dialogs") / filename))
            region = search_box(MODAL_CLOSE_TITLES[feature], frame)
            frame[region.y:region.y+region.height, region.x:region.x+region.width] = 0
            self.bind_frame(frame)
            self.assertIsNone(self.task.find_one(feature))

    def test_low_diagnostic_threshold_does_not_bypass_title_guard(self):
        frame = np.zeros((1080, 2560, 3), dtype=np.uint8)
        self.executor.frame = frame
        for feature in MODAL_CLOSE_TITLES:
            with patch("src.tasks.feature_ocr.find_text", return_value=None) as find:
                self.assertIsNone(self.task.find_one(feature, threshold=-1))
            self.assertEqual(.8, find.call_args.args[3])

    def test_overlay_takes_priority_over_underlying_result(self):
        for _, scene, feature in self.CASES:
            visible = {feature, "Continue-Battle", "Back-To-Port"}
            with self.subTest(scene=scene), patch.object(self.task, "find_one", side_effect=lambda name, **kw: Box(1, 1, 10, 10) if name in visible else None):
                self.assertEqual(scene, self.task._detect_scene(refresh=False))

    def test_startup_and_battle_dispatch_close_then_resume(self):
        for _, scene, _ in self.CASES:
            with self.subTest(scene=scene, phase="startup"), patch.object(self.task, "_detect_scene", side_effect=[scene, "main"]), patch.object(self.task, "_handle_screen_button", return_value=True) as handle:
                self.assertTrue(self.task._return_to_main())
                self.assertEqual(scene, handle.call_args.args[0])
                self.assertIsNotNone(handle.call_args.kwargs["deadline"])
            with self.subTest(scene=scene, phase="battle"), patch.object(self.task, "_detect_scene", side_effect=[scene, "result"]), patch.object(self.task, "_handle_screen_button", return_value=True) as handle, patch.object(self.task, "_send_battle_action") as fire:
                self.assertTrue(self.task._run_until_result())
                handle.assert_called_once_with(scene)
                fire.assert_not_called()

    def test_end_transition_handles_dialog_before_continuing(self):
        for _, scene, _ in self.CASES:
            with patch.object(self.task, "_detect_scene", side_effect=[scene, "result", "queue"]), patch.object(self.task, "_handle_screen_button", return_value=True) as handle, patch.object(self.task, "find_one", return_value=Box(1, 1, 10, 10)), patch.object(self.task, "sleep"), patch.object(self.task, "click") as click:
                self.assertEqual("continued", self.task._transition_battle_end(True))
            self.assertEqual(scene, handle.call_args.args[0])
            self.assertEqual(1, click.call_count)

    def test_close_handler_waits_through_unknown_without_clicking_submit(self):
        for _, scene, feature in self.CASES:
            def poll(predicate, **kwargs):
                self.assertFalse(predicate())  # 原弹窗仍在，首次关闭未立刻生效。
                self.assertFalse(predicate())  # 过渡帧不能直接报告已关闭。
                return predicate()
            with patch.object(self.task, "wait_click_feature", return_value=True) as click, patch.object(self.task, "_detect_scene", side_effect=[scene, "unknown", "result"]), patch.object(self.task, "wait_until", side_effect=poll):
                self.assertTrue(self.task._handle_screen_button(scene))
            self.assertEqual([feature], [c.args[0] for c in click.call_args_list])


if __name__ == "__main__":
    unittest.main()
