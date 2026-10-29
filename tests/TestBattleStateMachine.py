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

    def replay(self, scenes, buttons=None, frame=None):
        """每轮提供一个新页面；耗尽后保留最后一页，模拟点击无效。

        frame 用于模拟转场：每次场景识别推进一个画面，用来复现入口点击在加载动画中丢失的路径。
        """
        events = iter(scenes)
        frames = iter(frame) if frame is not None else None
        default_buttons = {
            "result": {"Continue-Battle", "Back-To-Port"},
            "leave_confirm": {"Continue-Battle-After-Sunk", "Leave-Battle-Confirm"},
            "main": {"Join-Battle", "Select-Battle-Mode", "Addon-Selector", "Equipment"},
            "battle_mode": set(self.task.BATTLE_MODES),
        }
        default_buttons.update(buttons or {})

        def detect():
            source = frames if frames is not None else events  # 转场测试由画面推进场景，普通测试由每轮调用推进。
            self.scene = next(source, self.scene)
            return self.scene

        def find(name, **kwargs):
            return Box(10, 10, 300, 120, name=name) if name in default_buttons.get(self.scene, set()) else None

        def sleep(seconds):
            self.clock += seconds

        stack = ExitStack()
        stack.enter_context(patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: self.clock))
        stack.enter_context(patch.object(self.task, "_detect_scene", side_effect=detect))
        stack.enter_context(patch.object(self.task, "find_one", side_effect=find))
        stack.enter_context(patch.object(self.task, "sleep", side_effect=sleep))
        return stack

    def clicks(self):
        return [getattr(c.args[0], "name", None) for c in self.task.click.call_args_list]

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
        self.assertEqual(2, self.task.send_key.call_count)  # 先按 ESC 离开已卸载的加成页，再关闭回港时的残留弹窗。

    def test_empty_addon_page_escapes_back_to_port_instead_of_stalling(self):
        # 2026-10-29 真机故障：加成页本来就是空的（只有“安装最佳”），重构后在 return 阶段不发 ESC，
        # 于加成页空转 60 秒后中断。空页面必须立即按 ESC 回港。
        port = {"Join-Battle", "Select-Battle-Mode", "Addon-Selector", "Equipment"}
        empty_addon = {"Install-Best-Buff"}
        with self.replay(["main", "addon", "addon", "addon", "main"], {"main": port, "addon": empty_addon}):
            self.assertTrue(self.task._prepare_loadout_page("Addon-Selector", "addon", "Remove-All-Buff", "Install-Best-Buff"))
        self.task.send_key.assert_called_with("esc", after_sleep=0)
        self.assertLess(self.clock, 60)  # 不再耗尽整个 60 秒额度。

    def test_loadout_entry_retries_after_dead_port_transition_without_escaping(self):
        # 复现 2026-10-28 故障：入口点击在转场中丢失，旧逻辑第二次点击被五分钟冷却挡住后超时。
        port = {"Join-Battle", "Select-Battle-Mode", "Addon-Selector", "Equipment"}
        addon = {"Remove-All-Buff"}
        scenes = ["main", "main", "main", "main", "main", "main", "main", "main", "addon", "addon", "main", "unknown", "main"]
        with self.replay(scenes, {"main": port, "addon": addon}):
            self.assertTrue(self.task._prepare_loadout_page("Addon-Selector", "addon", "Remove-All-Buff", "Install-Best-Buff"))
        self.assertEqual(2, self.clicks().count("Addon-Selector"))  # 首次点击帧丢失后按冷却重试一次。
        self.assertEqual(["Addon-Selector", "Addon-Selector", "Remove-All-Buff"], self.clicks())
        self.task.send_key.assert_not_called()  # 进入阶段一次 ESC 都不发，旧逻辑会在这里盲按 ESC。
        self.assertLess(self.clock, 60)

    def test_addon_page_wins_over_visible_battle_mode_label(self):
        # 2026-10-29 真机故障根因：加成页同时具备右上角 ESC 关闭标记和港口左下角的“非对称战斗”标签，
        # 叠加判断里 battle_mode 排在 addon 之前时，加成页永远识别不成 addon。
        def find(name, **kwargs):
            visible = {"Popup-ESC-Close", "Install-Best-Buff", "Asymmetry-Battle"}  # 与真机截图一致的同框元素。
            return Box(10, 10, 300, 120, name=name) if name in visible else None
        with patch.object(self.task, "find_one", side_effect=find):
            self.assertEqual("addon", self.task._detect_scene(refresh=False))

    def test_loadout_entry_is_not_clicked_while_addon_page_opens(self):
        # 加成页正在展开时被识别成 battle_mode，不能按 ESC 把刚打开的页面关掉。
        port = {"Join-Battle", "Select-Battle-Mode", "Addon-Selector", "Equipment"}
        addon = {"Remove-All-Buff"}
        with self.replay(["main", "battle_mode", "battle_mode", "addon", "addon", "main", "unknown", "main"], {"main": port, "addon": addon}):
            self.assertTrue(self.task._prepare_loadout_page("Addon-Selector", "addon", "Remove-All-Buff", "Install-Best-Buff"))
        self.assertEqual(["Addon-Selector", "Remove-All-Buff"], self.clicks())  # 打开页面期间不再重点入口。
        self.task.send_key.assert_not_called()  # 打开加成页的转场里一次 ESC 都不发，直接回港即视为完成。

    def test_addon_page_that_looks_like_battle_mode_is_reopened_without_stalling(self):
        # 2026-10-29 真机故障：加成页被持续误判成 battle_mode，旧逻辑只在港口外反复按 ESC，最终 60 秒超时中断。
        port = {"Join-Battle", "Select-Battle-Mode", "Addon-Selector", "Equipment"}
        addon = {"Remove-All-Buff"}
        scenes = ["main", "battle_mode", "battle_mode", "battle_mode", "main", "main", "addon", "addon", "main"]
        with self.replay(scenes, {"main": port, "addon": addon}):
            self.assertTrue(self.task._prepare_loadout_page("Addon-Selector", "addon", "Remove-All-Buff", "Install-Best-Buff"))
        self.assertIn("Addon-Selector", self.clicks())  # 必须真的重开入口，而不是空等。
        self.assertIn("Remove-All-Buff", self.clicks())
        self.assertLess(self.clock, 60)  # 不再耗尽整个 60 秒额度。

    def test_reopen_is_bounded_so_pages_cannot_flip_forever(self):
        # 页面反复掉出时必须有限次重开，然后给出可读的失败原因，而不是无限开关。
        port = {"Join-Battle", "Select-Battle-Mode", "Addon-Selector", "Equipment"}
        addon = {"Remove-All-Buff"}
        scenes = ["main", "addon", "battle_mode"] * 40
        with self.replay(scenes, {"main": port, "addon": addon}):
            self.task._prepare_loadout_page("Addon-Selector", "addon", "Remove-All-Buff", "Install-Best-Buff")
        self.assertLessEqual(self.clicks().count("Addon-Selector"), 1 + self.task.LOADOUT_REOPEN_LIMIT)

    def test_loadout_entry_click_targets_button_icon_not_panel_center(self):
        # 标注框包住悬停展开面板，点击必须落在框内按钮图标附近。
        with self.replay(["main", "addon", "addon", "main"], {"main": {"Addon-Selector"}, "addon": {"Remove-All-Buff"}}):
            self.task._prepare_loadout_page("Addon-Selector", "addon", "Remove-All-Buff", "Install-Best-Buff")
        point = self.task.click.call_args_list[0].args[0]
        self.assertEqual((40, 34), (point.x, point.y))  # 起点是标注框左上角 (10, 10)。
        self.assertLess(point.x, 160)  # 明显偏离面板中心 (160, 70)。

    def test_unreachable_loadout_entry_fails_with_actionable_reason(self):
        def find(name, **kwargs):
            return Box(10, 10, 300, 120, name=name) if name == "Addon-Selector" else None  # 加成页始终不出现。
        with self.replay(["main", "main"]), patch.object(self.task, "find_one", side_effect=find):
            self.assertFalse(self.task._prepare_loadout_page("Addon-Selector", "addon", "Remove-All-Buff", "Install-Best-Buff"))
        self.assertGreaterEqual(self.clicks().count("Addon-Selector"), 8)  # 旧逻辑只允许两次。
        self.assertIn("悬停态模板", self.task.log_error.call_args.args[0])

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
