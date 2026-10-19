import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, call, patch

import cv2
import numpy as np
from ok import Box, FeatureSet

from src.config import config, make_bottom_right_black
from src.tasks.AutoPveBattleTask import AutoPveBattleTask
from src.tasks.ScreenRecognitionTestTask import ScreenRecognitionTestTask
from tests.ocr_support import bind_ocr


class TestBattleEntryFlow(unittest.TestCase):
    def setUp(self):
        self.executor = MagicMock()
        self.executor.scene = None
        self.task = AutoPveBattleTask(self.executor, None)
        self.task.config = dict(self.task.default_config)

    def test_mode_dropdown_defaults_and_validation(self):
        self.assertEqual("PVE-Battle", self.task.config["Battle Mode"])
        self.assertEqual({"type": "drop_down", "options": ["PVE-Battle", "Asymmetry-Battle"]},
                         self.task.config_type["Battle Mode"])
        for mode in self.task.BATTLE_MODES:
            self.assertIsNone(self.task.validate_config("Battle Mode", mode))
        self.assertIsNotNone(self.task.validate_config("Battle Mode", "random"))
        diagnostic = ScreenRecognitionTestTask(self.executor, None)
        for metadata in (diagnostic.default_config, diagnostic.config_description, diagnostic.config_type):
            self.assertNotIn("Battle Mode", metadata)

    def test_preparation_clicks_only_selected_mode(self):
        for mode in self.task.BATTLE_MODES:
            with self.subTest(mode=mode):
                self.task.config["Battle Mode"] = mode
                with patch.object(self.task, "wait_click_feature", return_value=True) as click, \
                        patch.object(self.task, "get_feature_by_name", return_value=object()), \
                        patch.object(self.task, "_recall_commander_if_needed", return_value=True), \
                        patch.object(self.task, "_return_to_main", return_value=True), \
                        patch.object(self.task, "_select_battle_mode", return_value=True), \
                        patch.object(self.task, "_prepare_loadout_page", return_value=True):
                    self.assertTrue(self.task._prepare_and_join_first_battle())
                self.assertEqual(["Pick-First-Ship", "Join-Battle"], [c.args[0] for c in click.call_args_list])

    def test_missing_selected_mode_stops_without_falling_back(self):
        self.task.config["Battle Mode"] = "Asymmetry-Battle"
        with patch.object(self.task, "wait_click_feature", return_value=True) as click, \
                patch.object(self.task, "get_feature_by_name", return_value=None), \
                patch.object(self.task, "_recall_commander_if_needed", return_value=True), \
                patch.object(self.task, "log_error"), patch.object(self.task, "_save_failure_screenshot"):
            self.assertFalse(self.task._prepare_and_join_first_battle())
        self.assertEqual(["Pick-First-Ship"], [c.args[0] for c in click.call_args_list])

    def test_relocated_mode_icons_are_detected_and_clicked_across_the_screen(self):
        for mode in self.task.BATTLE_MODES:
            for x, y in ((80, 60), (4000, 1500)):
                with self.subTest(mode=mode, position=(x, y)):
                    frame = np.full((2160, 5120, 3), 30, dtype=np.uint8)
                    self.bind_frame(frame)
                    template = self.task.get_feature_by_name(mode).mat
                    height, width = template.shape[:2]
                    frame[y:y+height, x:x+width] = template
                    match = self.task.find_one(mode, threshold=self.task.threshold)
                    self.assertIsNotNone(match)
                    self.assertTrue(x <= match.center()[0] <= x + width)
                    self.assertTrue(y <= match.center()[1] <= y + height)
                    self.assertEqual("battle_mode", self.task._detect_scene(refresh=False))
                    # Exercise the framework's real wait/click path, replacing only waiting and input.
                    with patch.object(self.task, "wait_until", side_effect=lambda predicate, **kwargs: predicate()), \
                            patch.object(self.task, "click_box") as click:
                        self.assertTrue(self.task.wait_click_feature(mode, threshold=self.task.threshold))
                    clicked = click.call_args.args[0]
                    self.assertTrue(x <= clicked.center()[0] <= x + width)
                    self.assertTrue(y <= clicked.center()[1] <= y + height)

    def test_unavailable_selected_mode_does_not_click_the_other_visible_mode(self):
        for selected in self.task.BATTLE_MODES:
            with self.subTest(selected=selected):
                visible = next(mode for mode in self.task.BATTLE_MODES if mode != selected)
                frame = np.full((2160, 5120, 3), 30, dtype=np.uint8)
                self.bind_frame(frame)
                template = self.task.get_feature_by_name(visible).mat
                height, width = template.shape[:2]
                frame[60:60+height, 80:80+width] = template
                with patch.object(self.task, "wait_until", side_effect=lambda predicate, **kwargs: predicate()), \
                        patch.object(self.task, "click_box") as click:
                    self.assertFalse(self.task.wait_click_feature(selected, threshold=self.task.threshold,
                                                                 raise_if_not_found=False))
                click.assert_not_called()

    def test_ship_icon_only_matches_inside_left_team_panel(self):
        for x, y, expected in ((175, 353, True), (300, 950, True), (80, 60, False),
                               (4000, 1500, False), (175, 1500, False), (600, 353, False)):
            with self.subTest(position=(x, y), expected=expected):
                frame = np.full((2160, 5120, 3), 30, dtype=np.uint8)
                self.bind_frame(frame)
                template = self.task.get_feature_by_name("Ship-Icon").mat
                height, width = template.shape[:2]
                frame[y:y + height, x:x + width] = template
                match = self.task.find_one("Ship-Icon", threshold=self.task.map_threshold)
                if expected:
                    self.assertIsNotNone(match)
                    self.assertEqual((x, y), (match.x, match.y))
                else:
                    self.assertIsNone(match)

    def test_ship_icon_explicit_search_box_is_preserved(self):
        frame = np.full((2160, 5120, 3), 30, dtype=np.uint8)
        self.bind_frame(frame)
        template = self.task.get_feature_by_name("Ship-Icon").mat
        height, width = template.shape[:2]
        frame[1500:1500 + height, 4000:4000 + width] = template
        match = self.task.find_one("Ship-Icon", threshold=self.task.map_threshold,
                                   box=Box(3950, 1450, 400, 200))
        self.assertIsNotNone(match)
        self.assertEqual((4000, 1500), (match.x, match.y))

    @unittest.skipUnless(Path("ok_templates/21x9/28.png").is_file(), "Non-Libertad reference screenshot unavailable")
    def test_configured_ship_matches_port_screenshot_without_resizing(self):
        frame = cv2.imread("ok_templates/21x9/28.png")
        self.bind_frame(frame)
        match = self.task.find_one("Pick-First-Ship", threshold=self.task.threshold)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match.confidence, self.task.threshold)

    def test_ship_icon_score_ignores_background_and_rejects_gray_silhouette(self):
        scores = []
        for background in ((0, 0, 0), (255, 255, 255), (180, 90, 40)):
            with self.subTest(background=background):
                frame = np.full((2160, 5120, 3), background, dtype=np.uint8)
                self.bind_frame(frame)
                template = self.task.get_feature_by_name("Ship-Icon").mat
                mask = self.task._ship_icon_yellow_mask(template) > 0
                height, width = template.shape[:2]
                patch_image = frame[353:353 + height, 175:175 + width]
                patch_image[mask] = template[mask]
                match = self.task.find_one("Ship-Icon", threshold=self.task.map_threshold)
                self.assertIsNotNone(match)
                self.assertEqual((175, 353), (match.x, match.y))
                scores.append(match.confidence)
        self.assertLess(max(scores) - min(scores), 0.001)
        self.assertGreater(min(scores), 0.99)
        gray = cv2.cvtColor(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        frame[:] = 30
        frame[353:353 + height, 175:175 + width] = gray
        self.assertIsNone(self.task.find_one("Ship-Icon", threshold=self.task.map_threshold))

    @unittest.skipUnless(Path("ok_templates/21x9/14.png").is_file(), "Local reference screenshots unavailable")
    def test_battle_views_with_map_button_matching(self):
        for size in ((5120, 2160), (2560, 1080)):
            for threshold in (0.7, 0.8):
                self.task.config["Template Threshold"] = threshold
                for filename, expected in (("14.png", "map"), ("15.png", "battle"), ("16.png", "battle"), ("22.png", "map")):
                    with self.subTest(size=size, threshold=threshold, screenshot=filename):
                        frame = cv2.resize(cv2.imread(str(Path("ok_templates/21x9") / filename)), size)
                        self.bind_frame(make_bottom_right_black(frame))
                        self.assertEqual(expected, self.task._detect_battle_view())

    def test_new_buttons_and_sunk_icon_take_priority_over_background(self):
        cases = (("Login-Game", "login"), ("Claim-Reward", "claim_reward"),
                 ("Close-Reward-Screen", "reward_screen"), ("Control-Camera", "leave_battle"))
        for feature, scene in cases:
            with self.subTest(feature=feature):
                visible = {feature, "Join-Battle", "Select-Battle-Mode", "In-Battle-Compass",
                           "Libertad-Nameplate", "Map-Tutorial"}
                with patch.object(self.task, "next_frame"), \
                        patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: Box(1, 1, 10, 10, name=name) if name in visible else None):
                    self.assertEqual(scene, self.task._detect_scene())

    def test_claim_precedes_close_when_both_reward_buttons_are_visible(self):
        visible = {"Claim-Reward", "Close-Reward-Screen"}
        with patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: object() if name in visible else None):
            self.assertEqual("claim_reward", self.task._detect_scene(refresh=False))

    def test_asymmetry_button_identifies_mode_screen_without_pve_button(self):
        with patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: object() if name == "Asymmetry-Battle" else None):
            self.assertEqual("battle_mode", self.task._detect_scene(refresh=False))

    def test_login_claim_and_close_reach_port_without_esc(self):
        with patch.object(self.task, "_detect_scene", side_effect=["login", "claim_reward", "reward_screen", "main"]), \
                patch.object(self.task, "wait_click_feature", return_value=True) as click, \
                patch.object(self.task, "wait_until", return_value=True), \
                patch.object(self.task, "send_key") as keys:
            self.assertTrue(self.task._return_to_main())
        self.assertEqual(["Login-Game", "Claim-Reward", "Close-Reward-Screen"], [c.args[0] for c in click.call_args_list])
        keys.assert_not_called()

    def test_button_transition_waits_through_loading_for_next_known_screen(self):
        observations = []

        def poll(predicate, **kwargs):
            for _ in range(3):
                observations.append(predicate())
            return observations[-1]

        with patch.object(self.task, "wait_click_feature", return_value=True), \
                patch.object(self.task, "_detect_scene", side_effect=["login", "unknown", "claim_reward"]), \
                patch.object(self.task, "wait_until", side_effect=poll):
            self.assertTrue(self.task._handle_screen_button("login"))
        self.assertEqual([False, False, True], observations)

    def test_startup_waits_up_to_five_minutes_without_esc(self):
        for ready_at in (0, 299, None):
            with self.subTest(ready_at=ready_at):
                elapsed = [0]

                def sleep(seconds):
                    elapsed[0] += seconds

                def detect():
                    return "main" if ready_at is not None and elapsed[0] >= ready_at else "unknown"

                with patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: elapsed[0]), \
                        patch.object(self.task, "ensure_in_front", return_value=True), \
                        patch.object(self.task, "_detect_scene", side_effect=detect), \
                        patch.object(self.task, "sleep", side_effect=sleep), \
                        patch.object(self.task, "send_key") as keys, \
                        patch.object(self.task, "wait_click_feature") as click, \
                        patch.object(self.task, "_prepare_and_join_first_battle", return_value=False) as prepare, \
                        patch.object(self.task, "log_info"), patch.object(self.task, "log_error") as error:
                    self.task.run()
                keys.assert_not_called()
                click.assert_not_called()
                self.assertEqual(300 if ready_at is None else ready_at, elapsed[0])
                self.assertEqual(0 if ready_at is None else 1, prepare.call_count)
                if ready_at is None:
                    error.assert_any_call("等待游戏主界面超时（300 秒），任务停止。")

    def test_reward_retries_visible_button_but_never_clicks_loading_screen(self):
        for ready_at, expected_retries, expected_result in ((8, 1, True), (None, 2, False)):
            with self.subTest(ready_at=ready_at):
                elapsed = [0.0]
                retries = []
                observations = []

                def scene(refresh=False):
                    observations.append(elapsed[0])
                    if ready_at is not None and elapsed[0] >= ready_at:
                        return "reward_screen"
                    return "unknown" if 4 <= elapsed[0] < 7 else "claim_reward"

                def poll(predicate, time_out, post_action, **kwargs):
                    while elapsed[0] < time_out:
                        if predicate():
                            return True
                        post_action()
                    return None

                with patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: elapsed[0]), \
                        patch.object(self.task, "wait_click_feature", return_value=True), \
                        patch.object(self.task, "_detect_scene", side_effect=scene), \
                        patch.object(self.task, "find_one", return_value=Box(100, 200, 30, 40, name="Claim-Reward")), \
                        patch.object(self.task, "click", side_effect=lambda *args, **kwargs: retries.append(elapsed[0])), \
                        patch.object(self.task, "sleep", side_effect=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)), \
                        patch.object(self.task, "wait_until", side_effect=poll), \
                        patch.object(self.task, "_save_failure_screenshot") as save:
                    self.assertEqual(expected_result, self.task._handle_screen_button("claim_reward", deadline=300))
                self.assertEqual([7.0, 12.0][:expected_retries], retries)
                self.assertEqual(8.0 if expected_result else 17.0, elapsed[0])
                self.assertGreater(len(observations), 10)
                self.assertEqual(0 if expected_result else 1, save.call_count)

    def test_reward_retry_cannot_click_after_recognition_exceeds_deadline(self):
        elapsed = [0.0]

        def detect(refresh=False):
            elapsed[0] = 301
            return "claim_reward"

        with patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: elapsed[0]), \
                patch.object(self.task, "wait_click_feature", return_value=True), \
                patch.object(self.task, "_detect_scene", side_effect=detect), \
                patch.object(self.task, "wait_until", side_effect=lambda predicate, **kwargs: predicate()), \
                patch.object(self.task, "click") as click, \
                patch.object(self.task, "_save_failure_screenshot") as save:
            self.assertFalse(self.task._handle_screen_button("claim_reward", deadline=300))
        click.assert_not_called()
        save.assert_called_once_with("Claim-Reward")

    def test_failure_screenshots_are_unique_and_preserve_pixels(self):
        frame = np.full((12, 24, 3), (20, 40, 60), dtype=np.uint8)
        self.executor.frame = frame
        self.executor.nullable_frame.return_value = frame
        with TemporaryDirectory() as directory, \
                patch.object(self.task, "FAILURE_DIRECTORY", Path(directory)):
            self.task._save_failure_screenshot("Claim-Reward")
            self.task._save_failure_screenshot("Claim-Reward")
            paths = list(Path(directory).glob("*/*_Claim-Reward_*.png"))
            self.assertEqual(2, len(paths))
            for path in paths:
                restored = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(frame, restored)

    def test_startup_login_and_rewards_share_the_same_deadline(self):
        elapsed = [0]
        budgets = []

        def transition(predicate, **kwargs):
            budgets.append(kwargs["time_out"])
            elapsed[0] += min(120, kwargs["time_out"])
            return elapsed[0] < 300

        with patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: elapsed[0]), \
                patch.object(self.task, "_detect_scene", side_effect=["login", "claim_reward", "reward_screen"]), \
                patch.object(self.task, "wait_click_feature", return_value=True) as click, \
                patch.object(self.task, "wait_until", side_effect=transition), \
                patch.object(self.task, "send_key") as keys, patch.object(self.task, "log_error") as error:
            self.assertFalse(self.task._return_to_main(time_out=300))
        self.assertEqual([300, 180, 60], budgets)
        self.assertEqual(300, elapsed[0])
        self.assertEqual(["Login-Game", "Claim-Reward", "Close-Reward-Screen"],
                         [c.args[0] for c in click.call_args_list])
        keys.assert_not_called()
        error.assert_any_call("等待游戏主界面超时（300 秒），任务停止。")

    def test_loading_does_not_consume_return_attempts(self):
        scenes = ["unknown"] * 10 + ["menu", "addon", "equipment", "battle_mode", "main"]
        with patch.object(self.task, "_detect_scene", side_effect=scenes), \
                patch.object(self.task, "sleep"), patch.object(self.task, "log_info"), \
                patch.object(self.task, "send_key") as keys:
            self.assertTrue(self.task._return_to_main(max_attempts=4, time_out=300))
        self.assertEqual([call("esc", after_sleep=1)] * 4, keys.call_args_list)

    def test_stuck_known_page_keeps_return_attempt_limit(self):
        with patch.object(self.task, "_detect_scene", return_value="menu"), \
                patch.object(self.task, "send_key") as keys:
            self.assertFalse(self.task._return_to_main(time_out=300))
        self.assertEqual(8, keys.call_count)

    def test_startup_timeout_during_recognition_does_not_click(self):
        elapsed = [0]

        def detect():
            elapsed[0] = 301
            return "login"

        with patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: elapsed[0]), \
                patch.object(self.task, "_detect_scene", side_effect=detect), \
                patch.object(self.task, "wait_click_feature") as click, \
                patch.object(self.task, "send_key") as keys, patch.object(self.task, "log_error"):
            self.assertFalse(self.task._return_to_main(time_out=300))
        click.assert_not_called()
        keys.assert_not_called()

    def test_stuck_reward_page_stops_without_combat_input(self):
        with patch.object(self.task, "_detect_scene", return_value="claim_reward"), \
                patch.object(self.task, "wait_click_feature", return_value=True), \
                patch.object(self.task, "wait_until", return_value=None), \
                patch.object(self.task, "log_error"), \
                patch.object(self.task, "_send_battle_action") as fire:
            self.assertFalse(self.task._run_until_result())
        fire.assert_not_called()

    def test_reward_interrupt_during_battle_does_not_send_combat_input(self):
        with patch.object(self.task, "_detect_scene", side_effect=["claim_reward", "reward_screen", "result"]), \
                patch.object(self.task, "wait_click_feature", return_value=True) as click, \
                patch.object(self.task, "wait_until", return_value=True), \
                patch.object(self.task, "_send_battle_action") as fire:
            self.assertTrue(self.task._run_until_result())
        self.assertEqual(["Claim-Reward", "Close-Reward-Screen"], [c.args[0] for c in click.call_args_list])
        fire.assert_not_called()

    def test_control_camera_stops_fire_and_opens_leave_flow(self):
        for can_continue, expected in ((True, "continued"), (False, "left")):
            with self.subTest(can_continue=can_continue), patch.object(self.task, "next_frame"), patch.object(
                    self.task, "find_one", side_effect=lambda name, **kwargs: object() if name == "Control-Camera" else None), patch.object(
                    self.task, "_transition_battle_end", return_value=expected) as transition, patch.object(
                    self.task, "_send_battle_action") as fire, patch.object(self.task, "_initialize_battle_navigation") as navigate:
                self.assertEqual(expected, self.task._run_until_result(can_continue))
            transition.assert_called_once_with(can_continue, open_leave=True)
            fire.assert_not_called()
            navigate.assert_not_called()

    def test_missing_optional_templates_are_ignored_by_older_packs(self):
        frame = cv2.imread("assets/16x10/images/0.png")
        self.bind_frame(frame)
        for feature in self.task.OPTIONAL_FEATURES:
            with self.subTest(feature=feature):
                self.assertIsNone(self.task.find_one(feature, threshold=self.task.threshold))

    def bind_frame(self, frame):
        bind_ocr(self.executor)
        matching = config["template_matching"]
        self.executor.feature_set = FeatureSet(False, matching["coco_feature_json"],
                                              matching["default_horizontal_variance"],
                                              matching["default_vertical_variance"], matching["default_threshold"])
        self.executor.frame = frame
        self.executor.method.width = frame.shape[1]
        self.executor.method.height = frame.shape[0]

    @unittest.skipUnless(Path("ok_templates/21x9/coco_annotations.json").is_file(), "Local reference screenshots unavailable")
    def test_new_scenes_match_their_annotated_screenshots(self):
        root = Path("ok_templates/21x9")
        data = json.loads((root / "coco_annotations.json").read_text(encoding="utf-8"))
        categories = {c["name"]: c["id"] for c in data["categories"]}
        images = {i["id"]: i for i in data["images"]}
        cases = {"Login-Game": "login", "Claim-Reward": "claim_reward", "Close-Reward-Screen": "reward_screen",
                 "Control-Camera": "leave_battle", "Asymmetry-Battle": "battle_mode"}
        for feature, scene in cases.items():
            with self.subTest(feature=feature):
                ann = next(a for a in data["annotations"] if a["category_id"] == categories[feature])
                path = root / Path(images[ann["image_id"]]["file_name"]).name
                self.bind_frame(make_bottom_right_black(cv2.imread(str(path))))
                self.assertIsNotNone(self.task.find_one(feature, threshold=self.task.threshold))
                self.assertEqual(scene, self.task._detect_scene(refresh=False))


if __name__ == "__main__":
    unittest.main()
