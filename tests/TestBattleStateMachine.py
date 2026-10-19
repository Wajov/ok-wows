"""用虚拟时间回放真实故障路径，验证状态跳转、有限重试及场次提交。"""

from contextlib import ExitStack
from pathlib import Path
import unittest
from unittest.mock import MagicMock, call, patch

import cv2
from ok import Box, FeatureSet

from src.tasks.AutoPveBattleTask import AutoPveBattleTask
from tests.ocr_support import bind_ocr


class TestBattleStateMachine(unittest.TestCase):
    def setUp(self):
        self.executor = MagicMock()
        self.executor.scene = None
        self.task = AutoPveBattleTask(self.executor, None)
        self.task.config = dict(self.task.default_config)
        self.task.log_info = MagicMock()
        self.task.log_error = MagicMock()
        self.task._save_failure_screenshot = MagicMock()
        self.task.click = MagicMock()
        self.task.send_key = MagicMock()
        self.clock = 0
        self.scene = "unknown"

    def replay(self, scenes, buttons=None):
        """每轮提供一个新页面；耗尽后保留最后一页，模拟点击无效。"""
        events = iter(scenes)
        default_buttons = {
            "result": {"Continue-Battle", "Back-To-Port"},
            "leave_confirm": {"Continue-Battle-After-Sunk", "Leave-Battle-Confirm"},
            "main": {"Join-Battle", "Select-Battle-Mode", "Addon-Selector", "Equipment"},
            "battle_mode": set(self.task.BATTLE_MODES),
        }
        default_buttons.update(buttons or {})

        def detect():
            self.scene = next(events, self.scene)
            return self.scene

        def find(name, **kwargs):
            return Box(10, 10, 30, 20, name=name) if name in default_buttons.get(self.scene, set()) else None

        def sleep(seconds):
            self.clock += seconds

        stack = ExitStack()
        stack.enter_context(patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: self.clock))
        stack.enter_context(patch.object(self.task, "_detect_scene", side_effect=detect))
        stack.enter_context(patch.object(self.task, "find_one", side_effect=find))
        stack.enter_context(patch.object(self.task, "sleep", side_effect=sleep))
        return stack

    def clicks(self):
        return [c.args[0].name for c in self.task.click.call_args_list]

    def test_sunk_can_jump_directly_to_result_after_escape(self):
        with self.replay(["leave_battle", "unknown", "result", "result", "queue"]):
            self.assertEqual("continued", self.task._handle_leave_battle(True))
        self.assertEqual(["Continue-Battle"], self.clicks())
        self.task.send_key.assert_called_once_with("esc", after_sleep=0)

    def test_confirmation_can_change_to_result_while_click_is_fading(self):
        with self.replay(["leave_confirm", "result", "unknown", "queue"]):
            self.assertEqual("continued", self.task._handle_leave_battle(True))
        self.assertEqual(["Continue-Battle-After-Sunk", "Continue-Battle"], self.clicks())

    def test_popup_during_end_transition_is_closed_before_continue(self):
        with self.replay(["popup", "leave_confirm", "unknown", "queue"]):
            self.assertEqual("continued", self.task._handle_leave_battle(True))
        self.assertEqual(["Continue-Battle-After-Sunk"], self.clicks())
        self.task.send_key.assert_called_once_with("esc", after_sleep=0)

    def test_last_battle_uses_confirm_and_waits_for_port(self):
        with self.replay(["leave_battle", "leave_confirm", "unknown", "main"]):
            self.assertEqual("left", self.task._handle_leave_battle(False))
        self.assertEqual(["Leave-Battle-Confirm"], self.clicks())

    def test_missing_sunk_continue_falls_back_to_port(self):
        with self.replay(["leave_confirm", "main"], {"leave_confirm": {"Leave-Battle-Confirm"}}):
            self.assertEqual("left", self.task._handle_leave_battle(True))
        self.assertEqual(["Leave-Battle-Confirm"], self.clicks())

    def test_missing_result_continue_falls_back_to_port(self):
        with self.replay(["result", "main"], {"result": {"Back-To-Port"}}):
            self.assertEqual("left", self.task._transition_battle_end(True))
        self.assertEqual(["Back-To-Port"], self.clicks())

    def test_last_result_does_not_finish_during_unknown_fade(self):
        with self.replay(["result", "unknown", "unknown", "main"]):
            self.assertEqual("left", self.task._transition_battle_end(False))
        self.assertEqual(["Back-To-Port"], self.clicks())
        self.assertEqual("main", self.scene)

    def test_lingering_result_retries_only_three_times_and_saves_failure(self):
        with self.replay(["result"]):
            self.assertFalse(self.task._transition_battle_end(True))
        self.assertEqual(["Continue-Battle"] * 3, self.clicks())
        self.assertEqual(15, self.clock)
        self.task._save_failure_screenshot.assert_called_once_with("战斗结束")

    def test_unknown_does_not_count_as_successful_departure(self):
        with self.replay(["result", "unknown"]):
            self.assertFalse(self.task._transition_battle_end(True))
        self.assertEqual(["Continue-Battle"], self.clicks())
        self.assertEqual(60, self.clock)

    def test_end_transition_has_total_deadline_even_if_pages_alternate(self):
        with self.replay(["popup", "unknown"] * 80):
            self.assertFalse(self.task._transition_battle_end(True))
        self.assertLessEqual(self.task.send_key.call_count, 3)
        self.assertLessEqual(self.clock, 60)

    def test_server_restart_stops_without_firing_or_clicking(self):
        with self.replay(["queue", "match_stopped"]), patch.object(self.task, "_send_battle_action") as fire:
            self.assertFalse(self.task._run_until_result())
        fire.assert_not_called()
        self.task.click.assert_not_called()
        self.assertIn("服务器重启", self.task.log_error.call_args.args[0])

    def test_port_join_failure_is_bounded(self):
        with self.replay(["main"]):
            self.assertFalse(self.task._run_until_result())
        self.assertEqual(["Join-Battle"] * 3, self.clicks())

    def test_queue_timeout_is_reported_as_queue_failure(self):
        with self.replay(["queue"]):
            self.assertFalse(self.task._run_until_result())
        self.assertEqual(600, self.clock)
        self.assertIn("queue", self.task.log_error.call_args.args[0])

    def test_failed_navigation_cannot_retry_forever(self):
        with self.replay(["battle"]), patch.object(self.task, "_initialize_battle_navigation", return_value=False) as navigate:
            self.assertFalse(self.task._run_until_result())
        self.assertEqual(3, navigate.call_count)
        self.assertIn("导航初始化", self.task.log_error.call_args.args[0])

    def test_startup_can_recover_last_runs_result_without_counting_it(self):
        with self.replay(["result", "result", "unknown", "main", "main"]):
            self.assertTrue(self.task._return_to_main(time_out=300))
        self.assertEqual(["Back-To-Port"], self.clicks())
        self.assertFalse(any("已完成" in c.args[0] for c in self.task.log_info.call_args_list))

    def test_startup_recovery_cannot_extend_its_total_deadline(self):
        with self.replay(["result", "unknown"]):
            self.assertFalse(self.task._return_to_main(time_out=4))
        self.assertEqual(4, self.clock)

    def test_recognition_expiring_deadline_does_not_send_input(self):
        with self.replay(["result"]):
            def detect():
                self.clock = 61
                return "result"
            with patch.object(self.task, "_detect_scene", side_effect=detect):
                self.assertFalse(self.task._transition_battle_end(True))
        self.task.click.assert_not_called()

    def test_missing_mode_template_fails_before_any_selection_click(self):
        with patch.object(self.task, "get_feature_by_name", return_value=None):
            self.assertFalse(self.task._select_battle_mode())
        self.task.click.assert_not_called()
        self.task._save_failure_screenshot.assert_called_once()

    def test_already_selected_mode_skips_all_mode_clicks(self):
        with self.replay(["main"]), patch.object(self.task, "get_feature_by_name", return_value=object()), patch.object(self.task, "_current_battle_mode", return_value="PVE-Battle"):
            self.assertTrue(self.task._select_battle_mode())
        self.task.click.assert_not_called()

    def test_mode_entry_miss_retries_then_confirms_selected_mode(self):
        with self.replay(["main"] * 11 + ["battle_mode", "main"]), patch.object(self.task, "get_feature_by_name", return_value=object()), patch.object(self.task, "_current_battle_mode", side_effect=lambda: "PVE-Battle" if self.clock >= 6 else "Asymmetry-Battle"):
            self.assertTrue(self.task._select_battle_mode())
        self.assertEqual(["Select-Battle-Mode", "Select-Battle-Mode", "PVE-Battle"], self.clicks())

    def test_mode_click_does_not_succeed_until_port_title_matches(self):
        with self.replay(["battle_mode", "main"]), patch.object(self.task, "get_feature_by_name", return_value=object()), patch.object(self.task, "_current_battle_mode", return_value="Asymmetry-Battle"):
            self.assertFalse(self.task._select_battle_mode())
        self.assertLessEqual(self.clicks().count("Select-Battle-Mode"), 3)

    def test_loadout_checks_removal_and_closes_intervening_popup(self):
        # 同一个 addon 状态的按钮随卸载动作变化。
        def find(name, **kwargs):
            visible = {"Addon-Selector"} if self.scene == "main" else ({"Install-Best-Buff"} if self.clicks().count("Remove-All-Buff") else {"Remove-All-Buff"}) if self.scene == "addon" else set()
            return Box(10, 10, 30, 20, name=name) if name in visible else None
        with self.replay(["main", "unknown", "addon", "addon", "popup", "main"]), patch.object(self.task, "find_one", side_effect=find):
            self.assertTrue(self.task._prepare_loadout_page("Addon-Selector", "addon", "Remove-All-Buff", "Install-Best-Buff"))
        self.assertEqual(["Addon-Selector", "Remove-All-Buff"], self.clicks())
        self.assertEqual(2, self.task.send_key.call_count)

    def test_run_counts_confirmed_transitions_once_and_finishes_after_last_port(self):
        self.task.config["Battle Count"] = 2
        with patch.object(self.task, "ensure_in_front", return_value=True), patch.object(self.task, "_return_to_main", return_value=True), patch.object(self.task, "_prepare_and_join_first_battle", return_value=True), patch.object(self.task, "_run_until_result", return_value=True) as battles, patch.object(self.task, "_transition_battle_end", side_effect=["continued", "left"]) as transition, patch.object(self.task, "_finish_successfully") as finish:
            self.task.run()
        self.assertEqual([call(True), call(False)], battles.call_args_list)
        self.assertEqual([call(True), call(False)], transition.call_args_list)
        finish.assert_called_once()
        self.assertEqual(2, sum("已完成" in c.args[0] for c in self.task.log_info.call_args_list))

    def test_run_does_not_count_failed_transition_or_start_second_battle(self):
        self.task.config["Battle Count"] = 2
        with patch.object(self.task, "ensure_in_front", return_value=True), patch.object(self.task, "_return_to_main", return_value=True), patch.object(self.task, "_prepare_and_join_first_battle", return_value=True), patch.object(self.task, "_run_until_result", return_value=True) as battles, patch.object(self.task, "_transition_battle_end", return_value=False), patch.object(self.task, "_finish_successfully") as finish:
            self.task.run()
        battles.assert_called_once_with(True)
        finish.assert_not_called()
        self.assertFalse(any("已完成" in c.args[0] for c in self.task.log_info.call_args_list))

    def test_returning_to_port_after_sinking_counts_before_next_battle(self):
        self.task.config["Battle Count"] = 2
        with patch.object(self.task, "ensure_in_front", return_value=True), patch.object(self.task, "_return_to_main", return_value=True), patch.object(self.task, "_prepare_and_join_first_battle", return_value=True), patch.object(self.task, "_run_until_result", side_effect=["left", "left"]) as battles, patch.object(self.task, "_transition_battle_end") as transition, patch.object(self.task, "_finish_successfully") as finish:
            self.task.run()
        self.assertEqual([call(True), call(False)], battles.call_args_list)
        transition.assert_not_called()
        finish.assert_called_once()

    @unittest.skipUnless(Path("logs/task_failures/2026-09-29/10-01-19-326405_Auto PVE Battle_204ae843.png").is_file(), "Local failure screenshot unavailable")
    def test_server_restart_failure_screenshot_has_explicit_scene(self):
        bind_ocr(self.executor)
        self.executor.feature_set = FeatureSet(False, "assets/21x9/coco_annotations.json", .01, .01, .8)
        self.executor.frame = cv2.imread("logs/task_failures/2026-09-29/10-01-19-326405_Auto PVE Battle_204ae843.png")
        self.executor.method.height, self.executor.method.width = self.executor.frame.shape[:2]
        self.assertEqual("match_stopped", self.task._detect_scene(refresh=False))

    @unittest.skipUnless(Path("logs/task_failures/2026-10-01/09-06-50-816544_Auto PVE Battle_60c0e4c5.png").is_file(), "Local port failure screenshot unavailable")
    def test_port_failure_screenshot_already_has_requested_mode(self):
        bind_ocr(self.executor)
        self.executor.feature_set = FeatureSet(False, "assets/21x9/coco_annotations.json", .01, .01, .8)
        self.executor.frame = cv2.imread("logs/task_failures/2026-10-01/09-06-50-816544_Auto PVE Battle_60c0e4c5.png")
        self.executor.method.height, self.executor.method.width = self.executor.frame.shape[:2]
        self.task.config["Battle Mode"] = "Asymmetry-Battle"
        self.assertEqual("Asymmetry-Battle", self.task._current_battle_mode())
        with patch.object(self.task, "next_frame"):
            self.assertTrue(self.task._select_battle_mode())
        self.task.click.assert_not_called()


if __name__ == "__main__":
    unittest.main()
