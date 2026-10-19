import os  # 导入路径工具以判断本地忽略的原始模板目录是否存在。
import json  # 通过标注定位当前参考截图，避免复用编号后指向其他页面。
import unittest  # 导入标准库单元测试框架。
from pathlib import Path  # 导入路径对象以读取两套 coco 分类名。
from unittest.mock import MagicMock, call, patch  # 导入方法替身、调用记录和临时补丁工具。

import cv2  # 导入 OpenCV 以读取完整标注参考截图。
import numpy as np  # 导入数组工具以构造颜色模板测试数据。
from ok import Box  # 导入模板匹配结果使用的矩形框类型。
from ok import FeatureSet  # 导入真实模板引擎以验证场景标注。

from src.config import config, make_bottom_right_black  # 导入应用配置和正式截图预处理器。
from src.resolution_assets import asset_folder_for_size, coco_json_for_size, current_template_folder, pack_for_size, ratio_is_supported, redirect_asset_target, set_pack_override, template_folder_for_size  # 导入按窗口比例选择模板的辅助函数。
from src.tasks.AutoPveBattleTask import AutoPveBattleTask  # 导入本次新增的自动 PVE 任务。
from src.tasks.MyBaseTask import MyBaseTask  # 导入任务基类以验证全图搜索覆盖。
from src.tasks.feature_ocr import search_box  # 检查 OCR 保留的局部搜索边界。
from tests.ocr_support import bind_ocr  # 为真实截图验证连接应用 OCR。


class TestAutoPveBattleTask(unittest.TestCase):  # 定义不共享全局应用状态的自动 PVE 任务测试集合。

    def setUp(self):  # 为每个测试创建完全隔离的轻量任务实例。
        self.executor = MagicMock()  # 构造不连接真实游戏窗口的执行器替身并保留引用以检查关闭调用。
        self.executor.scene = None  # 提供任务基类初始化所需的场景属性。
        self.task = AutoPveBattleTask(self.executor, None)  # 使用替身执行器创建待测任务。
        self.task.config = dict(self.task.default_config)  # 使用普通字典模拟任务保存后的有效配置。
        set_pack_override(None)  # 每个测试从自动按窗口选目录开始，避免截图页手动比例残留。

    def test_task_is_registered_with_safe_defaults(self):  # 验证应用能发现任务且默认配置安全。
        self.assertIn(["src.tasks.AutoPveBattleTask", "AutoPveBattleTask"], config["onetime_tasks"])  # 确认任务已注册到一次性任务列表。
        self.assertEqual(1, self.task.default_config["Battle Count"])  # 确认默认只执行一场战斗。
        self.assertFalse(self.task.default_config["Close Game After Completion"])  # 确认默认不会在任务结束后关闭游戏。
        self.assertEqual(0.8, self.task.default_config["Template Threshold"])  # 确认默认匹配阈值与项目一致。
        self.assertEqual(0.75, self.task.map_threshold)  # 确认大地图识别使用更宽松但仍保守的专用阈值。

    def test_config_validation_rejects_invalid_values(self):  # 验证危险或无效输入会被配置界面拒绝。
        self.assertIsNotNone(self.task.validate_config("Battle Count", 0))  # 零场战斗必须视为无效。
        self.assertIsNotNone(self.task.validate_config("Battle Count", True))  # 布尔值不能被当作整数战斗场数。
        self.assertIsNotNone(self.task.validate_config("Template Threshold", 1.1))  # 大于一的匹配阈值必须视为无效。
        self.assertIsNone(self.task.validate_config("Battle Count", 2))  # 正整数战斗场数应该通过校验。
        self.assertIsNone(self.task.validate_config("Template Threshold", 0.85))  # 合法置信度应该通过校验。

    def test_scene_detection_uses_unique_main_screen_elements(self):  # 验证主界面需要两个独特元素共同确认。
        boxes = {  # 构造只包含主界面两个特有元素的匹配结果。
            "Join-Battle": Box(10, 10, 20, 20, name="Join-Battle"),  # 模拟加入战斗按钮。
            "Select-Battle-Mode": Box(40, 10, 20, 20, name="Select-Battle-Mode"),  # 模拟模式选择按钮。
        }  # 完成模拟匹配结果定义。
        with patch.object(self.task, "next_frame"), patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: boxes.get(name)):  # 使用同一张模拟截图执行场景识别。
            self.assertEqual("main", self.task._detect_scene())  # 两个主界面元素同时存在时应识别为主界面。

    def test_scene_detection_prioritizes_menu(self):  # 验证菜单覆盖其他页面时优先识别菜单。
        menu_box = Box(10, 10, 20, 20, name="Menu")  # 构造菜单特有元素的匹配结果。
        with patch.object(self.task, "next_frame"), patch.object(self.task, "find_one", return_value=menu_box):  # 模拟截图中所有查询都能看到菜单元素。
            self.assertEqual("menu", self.task._detect_scene())  # 菜单应以最高优先级返回。

    def test_scene_detection_prioritizes_leave_battle_over_battle(self):  # 验证舰船被击沉时优先处理离开战斗而不是继续开火。
        boxes = {  # 构造同时包含离开入口和普通战斗罗盘的匹配结果。
            "Leave-Battlefield": Box(10, 10, 20, 20, name="Leave-Battlefield"),  # 模拟十二号截图中的离开战斗入口。
            "In-Battle-Compass": Box(40, 10, 20, 20, name="In-Battle-Compass"),  # 模拟击沉页面仍然保留的战斗罗盘。
        }  # 完成模拟匹配结果定义。
        with patch.object(self.task, "next_frame"), patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: boxes.get(name)):  # 使用同一张模拟截图执行场景识别。
            self.assertEqual("leave_battle", self.task._detect_scene())  # 离开入口出现时必须进入离开处理分支。

    def test_map_requires_both_buttons_and_ignores_ship_icon(self):  # 覆盖两个地图按钮的全部组合，舰船图标不参与视图判断。
        for m_visible in (False, True):  # 覆盖 M 按钮存在和缺失。
            for b_visible in (False, True):  # 覆盖 B 按钮存在和缺失。
                for icon_visible in (False, True):  # 图标存在与否不应改变结果。
                    with self.subTest(m=m_visible, b=b_visible, icon=icon_visible):  # 标明失败组合。
                        visible = {"Libertad-Nameplate", "Map-Tutorial", "In-Battle-Compass"}  # 铭牌确认战斗，旧元素不参与决策。
                        visible.update(name for name, present in (("Map-M-Button", m_visible), ("Map-B-Button", b_visible), ("Ship-Icon", icon_visible)) if present)  # 设置当前帧按钮组合。
                        with patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: object() if name in visible else None) as find_one, patch.object(self.task, "get_feature_by_name", side_effect=lambda name: None if name == "Ship-Icon" else object()) as get_feature:  # 即使缺少舰船图标模板也能正确识别。
                            is_map = m_visible and b_visible  # 两个按钮缺一不可。
                            self.assertEqual("map" if is_map else "battle", self.task._detect_scene(refresh=False))  # 验证场景判断。
                            self.assertEqual(is_map, self.task._map_is_visible())  # 验证打开地图的确认。
                            self.assertEqual(not is_map, self.task._map_is_closed())  # 验证关闭地图的确认。
                            self.assertNotIn("Ship-Icon", [c.args[0] for c in find_one.call_args_list + get_feature.call_args_list])  # 确认不再读取或匹配舰船图标。

    def test_without_nameplate_does_not_enter_battle_or_close_map(self):  # 缺少铭牌时不将未知页面当成战斗。
        for visible in (set(), {"Ship-Icon"}, {"Map-M-Button", "Map-B-Button"}):  # 包括两个按钮误命中的情况。
            with self.subTest(visible=visible), patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: object() if name in visible else None):  # 控制可见元素。
                self.assertEqual("unknown", self.task._detect_scene(refresh=False))  # 不能绕过铭牌。
                self.assertFalse(self.task._map_is_visible())  # 非战斗页面不属于大地图。
                self.assertFalse(self.task._map_is_closed())  # 未知页面不代表关闭成功。

    def test_missing_map_button_template_does_not_infer_battle_view(self):  # 任一按钮资源缺失时不能把缺模板当成按钮消失。
        for missing in self.task.MAP_VIEW_FEATURES:  # 分别覆盖两个模板缺失。
            with self.subTest(missing=missing), patch.object(self.task, "find_one", return_value=object()), patch.object(self.task, "get_feature_by_name", side_effect=lambda name: None if name == missing else object()):  # 模拟资源不完整。
                self.assertIsNone(self.task._detect_battle_view())  # 保持未知状态。
                self.assertFalse(self.task._map_is_visible())  # 不误判大地图。
                self.assertFalse(self.task._map_is_closed())  # 不误判关闭成功。

    @unittest.skipUnless(os.path.isdir(os.path.join("ok_templates", "21x9")), "Local reference screenshots are not available.")  # 仅在本地原始模板目录存在时运行截图集成验证。
    def test_annotated_reference_screens_have_expected_scenes(self):  # 用完整标注截图验证全部页面都能被状态机识别。
        expected_scenes = {  # 按工作流截图编号定义预期页面状态。
            "0.png": "battle_mode",  # 零号截图是战斗模式选择页。
            "1.png": "queue",  # 一号截图是战斗排队页。
            "2.png": "main",  # 二号截图是游戏主界面。
            "3.png": "battle_start",  # 三号截图是等待战斗开始页。
            "4.png": "login",  # 四号截图现在是新标注的登录界面。
            "5.png": "claim_reward",  # 五号截图现在是领取奖励界面。
            "6.png": "result",  # 六号截图是战斗结束页。
            "7.png": "addon",  # 七号截图是未装备加成状态页。
            "8.png": "addon",  # 八号截图是已装备加成状态页。
            "9.png": "equipment",  # 九号截图是未装备旗子状态页。
            "10.png": "equipment",  # 十号截图是已装备旗子状态页。
            "11.png": "menu",  # 十一号截图是菜单页面。
            "12.png": "leave_battle",  # 十二号截图是舰船被击沉后的离开战斗页面。
            "14.png": "map",  # 十四号截图有舰船铭牌和两个地图按钮。
            "15.png": "battle",  # 十五号截图有舰船铭牌且没有地图按钮组合。
            "16.png": "battle",  # 十六号截图有舰船铭牌且没有地图按钮组合。
        }  # 完成截图与状态的对应关系定义。
        matching_config = config["template_matching"]  # 读取应用真实模板引擎参数。
        feature_set = FeatureSet(False, matching_config["coco_feature_json"], default_horizontal_variance=matching_config["default_horizontal_variance"], default_vertical_variance=matching_config["default_vertical_variance"], default_threshold=matching_config["default_threshold"])  # 创建不依赖 GUI 生命周期的真实模板引擎。
        current_frame = {"value": None}  # 使用可变容器保存当前子测试对应的截图。

        bind_ocr(self.executor)  # 所有场景均经过正式的 OCR/模板混合识别路径。
        self.executor.feature_set = feature_set  # 保留未迁移的地图和舰船模板引擎。

        for image_name, expected_scene in expected_scenes.items():  # 逐张加载用户已标注的参考截图。
            with self.subTest(image=image_name):  # 在失败信息中保留具体截图名称。
                current_frame["value"] = make_bottom_right_black(cv2.imread(os.path.join("ok_templates", "21x9", image_name)))  # 按正式截图处理方式加载参考图片。
                self.executor.frame = current_frame["value"]  # 设置任务实际读取的同一帧。
                self.executor.method.width = current_frame["value"].shape[1]  # 提供模板缩放所需的画面尺寸。
                self.executor.method.height = current_frame["value"].shape[0]
                with patch.object(self.task, "next_frame", return_value=current_frame["value"]):  # 仅替换截图来源，不绕过实际识别。
                    self.assertEqual(expected_scene, self.task._detect_scene())  # 确认特有元素能够判断出预期页面。

    @unittest.skipUnless(os.path.isdir(os.path.join("ok_templates", "21x9")), "Local reference screenshots are not available.")  # 仅在本地原始模板目录存在时运行确认按钮验证。
    def test_leave_confirmation_template_matches_reference_screen(self):  # 验证当前标注指向的确认离开按钮可以被真实模板识别。
        matching_config = config["template_matching"]  # 读取应用真实模板引擎参数。
        feature_set = FeatureSet(False, matching_config["coco_feature_json"], default_horizontal_variance=matching_config["default_horizontal_variance"], default_vertical_variance=matching_config["default_vertical_variance"], default_threshold=matching_config["default_threshold"])  # 创建不依赖 GUI 生命周期的真实模板引擎。
        source_root = Path("ok_templates/21x9")  # 定位完整参考截图目录。
        source_data = json.loads((source_root / "coco_annotations.json").read_text(encoding="utf-8"))  # 读取当前原始标注。
        category_id = next(item["id"] for item in source_data["categories"] if item["name"] == "Leave-Battle-Confirm")  # 找到确认离开按钮的分类。
        image_id = next(item["image_id"] for item in source_data["annotations"] if item["category_id"] == category_id)  # 按标注获取当前参考图 ID。
        image_name = next(item["file_name"] for item in source_data["images"] if item["id"] == image_id)  # 解析当前截图名称，不再使用已过时的十三号图。
        frame = make_bottom_right_black(cv2.imread(str(source_root / Path(image_name).name)))  # 按正式预处理方式加载正确参考图。
        boxes = feature_set.find_feature(frame, "Leave-Battle-Confirm", threshold=self.task.threshold, limit=1)  # 在确认页面中匹配确认离开按钮。
        self.assertTrue(boxes)  # 确认模板能够稳定找到用户标注的按钮。

    @unittest.skipUnless(os.path.isfile(os.path.join("ok_templates", "21x9", "23.png")), "Continue-battle confirmation screenshot is not available.")  # 仅在本地二十三号截图存在时验证局部搜索排除远处相似按钮。
    def test_continue_battle_button_excludes_confirmation_outside_local_region(self):  # 验证结算页模板不会跨区域命中确认框中心的相似按钮。
        bind_ocr(self.executor)  # 同名按钮排除逻辑必须走真实 OCR。
        matching_config = config["template_matching"]  # 读取应用真实模板引擎参数。
        feature_set = FeatureSet(False, matching_config["coco_feature_json"], default_horizontal_variance=matching_config["default_horizontal_variance"], default_vertical_variance=matching_config["default_vertical_variance"], default_threshold=matching_config["default_threshold"])  # 创建与正式任务一致的模板引擎。
        frame = make_bottom_right_black(cv2.imread(os.path.join("ok_templates", "21x9", "23.png")))  # 按正式截图预处理方式加载二十三号确认框截图。
        self.task.executor.feature_set = feature_set  # 将任务连接到真实模板引擎以执行局部匹配。
        self.task.executor.frame = frame  # 把二十三号截图设置为任务正在处理的最新帧。
        self.task.executor.method.width = frame.shape[1]  # 提供模板引擎所需的画面宽度。
        self.task.executor.method.height = frame.shape[0]  # 提供模板引擎所需的画面高度。
        full_screen_matches = feature_set.find_feature(frame, "Continue-Battle", horizontal_variance=1, vertical_variance=1, threshold=self.task.threshold, limit=1)  # 确认该截图确实存在全图搜索会命中的相似按钮。
        self.assertTrue(full_screen_matches)  # 避免因截图中没有相似按钮而让局部排除测试无效。
        button = self.task.find_one("Continue-Battle", threshold=self.task.threshold)  # 使用标注宽高各四倍的局部范围搜索。
        self.assertIsNone(button)  # 确认范围外的中心确认框不再被结算页模板命中。

    def test_map_shift_clicks_all_recognized_areas_from_nearest_to_farthest(self):  # 验证识别到的全部灰红占领区都会按住 Shift 按距离由近及远点击。
        map_overview = Box(0, 0, 300, 300, name="Map-Overview")  # 构造十九号截图标注对应的主地图范围。
        cursor = Box(0, 0, 10, 10, name="My-Ship-Cursor")  # 构造舰船光标位置。
        area_a = Box(10, 0, 10, 10, name="Area-A")  # 构造距离舰船最近但不应导航的绿色 A 区。
        area_b = Box(100, 0, 10, 10, name="Area-B")  # 构造距离舰船较远的红色 B 区。
        area_d = Box(20, 0, 10, 10, name="Area-D")  # 构造距离舰船最近的灰色 D 区。
        areas = {"Area-A": (area_a, "green"), "Area-B": (area_b, "red"), "Area-D": (area_d, "gray")}  # 提供绿色、红色和灰色结果以验证导航过滤规则。
        with patch.object(self.task, "wait_until", return_value=True), patch.object(self.task, "next_frame"), patch.object(self.task, "get_box_by_name", return_value=map_overview), patch.object(self.task, "_find_rotated_ship_cursor", return_value=cursor) as find_cursor, patch.object(self.task, "_find_area", side_effect=lambda name, box: areas.get(name, (None, None))) as find_area, patch.object(self.task, "find_one", return_value=None), patch.object(self.task, "send_key_down") as key_down, patch.object(self.task, "send_key_up") as key_up, patch.object(self.task, "click") as click, patch.object(self.task, "_close_map"), patch.object(self.task, "log_info") as log_info:  # 隔离地图处理中的截图和输入操作。
            self.task._handle_map()  # 执行一次地图航点选择。
        find_cursor.assert_called_once_with(map_overview)  # 确认舰船光标只在十九号截图框定的主地图范围内查找。
        for feature_name in ("Area-A", "Area-B", "Area-C", "Area-D"):  # 逐一检查四个区域的颜色识别范围。
            find_area.assert_any_call(feature_name, map_overview)  # 确认每个区域都限制在主地图范围内并返回颜色。
        key_down.assert_called_once_with("shift", after_sleep=0.5)  # 确认点击占领区前只按住一次 Shift。
        key_up.assert_called_once_with("shift", after_sleep=1)  # 确认全部目标点击完成后释放 Shift。
        self.assertEqual([call(area_d, after_sleep=3), call(area_b, after_sleep=3), call(295, 295, name="opposite-map-side", after_sleep=3)], click.call_args_list)  # 必须先按距离由近及远点击全部灰红占领区，再把地图对侧追加为最后的目标点。
        log_info.assert_any_call(f"按住 Shift 点击第 1/2 个占领区 Area-D，颜色为 gray，分数 {area_d.confidence * 100:.2f}%，阈值 75.00%，中心 (25, 5)。")  # 确认导航日志包含最近目标的顺序、分数、阈值和位置。
        log_info.assert_any_call(f"按住 Shift 点击第 2/2 个占领区 Area-B，颜色为 red，分数 {area_b.confidence * 100:.2f}%，阈值 75.00%，中心 (105, 5)。")  # 确认导航日志包含最远目标的顺序、分数、阈值和位置。
        log_info.assert_any_call("点击地图对侧 (295, 295) 作为最后的目标点。")  # 确认对侧航点在全部占领区之后追加。

    def test_map_releases_shift_when_hold_wait_or_queued_click_is_interrupted(self):  # 验证按住后的等待和追加航点中断时均释放 Shift。
        map_overview = Box(0, 0, 300, 300, name="Map-Overview")  # 使用固定地图范围隔离识别逻辑。
        cursor = Box(0, 0, 10, 10, name="My-Ship-Cursor")  # 提供计算对侧点所需的舰船位置。
        target = Box(20, 20, 10, 10, confidence=0.95, name="target")  # 分别模拟占领区与敌方基地。
        for route in ("area", "base"):  # 两种按住 Shift 的分支都需要保证释放。
            for interrupted_at in ("hold_wait", "queued_click"):  # 覆盖按下后等待及后续点选抛异常。
                def sleep(seconds):  # 保留真实按键入口，只中断指定的等待。
                    if interrupted_at == "hold_wait" and seconds == 0.5:
                        raise RuntimeError("navigation interrupted")

                def click(*args, **kwargs):  # 在 Shift 已按住时模拟追加航点失败。
                    if interrupted_at == "queued_click" and kwargs.get("name") == "opposite-map-side":
                        raise RuntimeError("navigation interrupted")

                with self.subTest(route=route, interrupted_at=interrupted_at), \
                        patch.object(self.task, "wait_until", return_value=True), \
                        patch.object(self.task, "next_frame"), \
                        patch.object(self.task, "get_box_by_name", return_value=map_overview), \
                        patch.object(self.task, "_find_rotated_ship_cursor", return_value=cursor), \
                        patch.object(self.task, "_find_area", side_effect=lambda name, box: (target, "gray") if route == "area" and name == "Area-A" else (None, None)), \
                        patch.object(self.task, "find_one", return_value=target if route == "base" else None), \
                        patch.object(self.task, "sleep", side_effect=sleep), \
                        patch.object(self.task, "click", side_effect=click), \
                        patch.object(self.task, "_close_map") as close_map, \
                        patch.object(self.task, "log_info"):
                    self.executor.interaction.reset_mock()  # 每个场景单独记录真实框架派发的按键。
                    with self.assertRaisesRegex(RuntimeError, "navigation interrupted"):
                        self.task._handle_map()  # 中断必须继续向调用方传播。
                    self.executor.interaction.send_key_down.assert_called_once_with("shift")  # 确认修饰键已实际派发按下。
                    self.executor.interaction.send_key_up.assert_called_once_with("shift")  # 无论在哪一步中断都必须派发释放。
                    close_map.assert_not_called()  # 中断后不再发送关闭地图操作。
                    self.assertFalse(self.task._preserve_input_timing)  # 异常退出后恢复原有全局等待模式。

    def test_area_recognition_returns_highest_scoring_color(self):  # 验证单个区域会比较三种颜色并返回最高分颜色。
        map_overview = Box(0, 0, 300, 300, name="Map-Overview")  # 构造颜色模板搜索使用的主地图范围。
        feature = MagicMock()  # 构造包含原始模板矩阵的正式特征替身。
        feature.mat = np.zeros((20, 12, 3), dtype=np.uint8)  # 提供可由 OpenCV 转换颜色的三通道模板。
        matches = [Box(10, 10, 12, 20, confidence=0.91, name="Area-A"), None, Box(10, 10, 12, 20, confidence=0.95, name="Area-A")]  # 模拟绿色、红色和灰色依次得到的结果。
        with patch.object(self.task, "get_feature_by_name", return_value=feature), patch.object(self.task, "find_one", side_effect=matches) as find_one:  # 隔离正式模板资源并控制三种颜色的匹配分数。
            area_box, area_color = self.task._find_area("Area-A", map_overview)  # 识别单个字母并取得其最高分颜色。
        self.assertIs(area_box, matches[2])  # 确认返回三种颜色候选中置信度最高的匹配框。
        self.assertEqual("gray", area_color)  # 确认返回值同时包含最高分模板对应的颜色。
        self.assertEqual(3, find_one.call_count)  # 确认绿色、红色和灰色模板都参与比较。
        for match_call in find_one.call_args_list:  # 检查每一种颜色调用都使用正式匹配范围和阈值。
            self.assertEqual("Area-A", match_call.args[0])  # 确认颜色变体仍使用原字母特征名称。
            self.assertEqual(-1.0, match_call.kwargs["threshold"])  # 先取得原始分数，再由任务按地图阈值过滤。
            self.assertIs(map_overview, match_call.kwargs["box"])  # 确认颜色识别限制在主地图范围内。
            self.assertIsInstance(match_call.kwargs["template"], np.ndarray)  # 确认每次调用都把转换后的模板交给框架原生接口。

    @unittest.skipUnless(os.path.isfile(os.path.join("ok_templates", "21x9", "14.png")), "Map reference screenshot is not available.")  # 仅在本地十四号地图截图存在时运行颜色识别集成验证。
    def test_area_colors_match_reference_map(self):  # 用十四号真实截图验证四个字母及颜色能够同时识别。
        matching_config = config["template_matching"]  # 读取应用真实模板引擎参数。
        feature_set = FeatureSet(False, matching_config["coco_feature_json"], default_horizontal_variance=matching_config["default_horizontal_variance"], default_vertical_variance=matching_config["default_vertical_variance"], default_threshold=matching_config["default_threshold"])  # 创建与正式任务一致的模板引擎。
        frame = make_bottom_right_black(cv2.imread(os.path.join("ok_templates", "21x9", "14.png")))  # 按正式截图预处理方式加载十四号大地图。
        self.task.executor.feature_set = feature_set  # 将任务连接到真实模板引擎以执行颜色变体匹配。
        self.task.executor.frame = frame  # 把十四号截图设置为任务正在处理的最新帧。
        self.task.executor.method.width = frame.shape[1]  # 提供框架全屏搜索框计算所需的画面宽度。
        self.task.executor.method.height = frame.shape[0]  # 提供框架全屏搜索框计算所需的画面高度。
        map_overview = self.task.get_box_by_name("Map-Overview")  # 读取会随截图分辨率缩放的正式主地图范围。
        results = {area_name: self.task._find_area(area_name, map_overview)[1] for area_name in ("Area-A", "Area-B", "Area-C", "Area-D")}  # 使用正式任务接口识别四个字母对应的颜色。
        self.assertEqual({"Area-A": "green", "Area-B": "red", "Area-C": "gray", "Area-D": "gray"}, results)  # 确认十四号截图返回绿、红、灰、灰四种状态。

    def test_map_selects_enemy_base_when_no_area_is_recognized(self):  # 验证没有占领区时会使用识别到的敌方基地。
        map_overview = Box(100, 100, 800, 600, name="Map-Overview")  # 构造主地图范围。
        enemy_base = Box(700, 150, 20, 20, name="Enemy-Base")  # 构造敌方基地匹配结果。
        with patch.object(self.task, "wait_until", return_value=True), patch.object(self.task, "next_frame"), patch.object(self.task, "get_box_by_name", return_value=map_overview), patch.object(self.task, "_find_rotated_ship_cursor", return_value=None), patch.object(self.task, "_find_area", return_value=(None, None)), patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: enemy_base if name == "Enemy-Base" else None), patch.object(self.task, "click") as click, patch.object(self.task, "_close_map"):  # 模拟只识别到敌方基地的地图。
            self.task._handle_map()  # 执行一次地图航点选择。
        click.assert_called_once_with(enemy_base, after_sleep=3)  # 确认任务点击敌方基地后等待三秒再关闭地图。

    def test_map_prefers_higher_scoring_enemy_base_over_false_areas(self):  # 验证占领区和敌方基地同时命中时选择更高分的一类。
        map_overview = Box(100, 100, 800, 600, name="Map-Overview")  # 构造主地图范围。
        cursor = Box(120, 500, 10, 10, name="My-Ship-Cursor")  # 构造舰船光标以便走占领区分支时也能点击。
        area_a = Box(200, 500, 10, 10, confidence=0.83, name="Area-A")  # 构造较低分的灰色占领区误识别。
        enemy_base = Box(700, 150, 20, 20, confidence=0.93, name="Enemy-Base")  # 构造更高分的敌方基地匹配。
        with patch.object(self.task, "wait_until", return_value=True), patch.object(self.task, "next_frame"), patch.object(self.task, "get_box_by_name", return_value=map_overview), patch.object(self.task, "_find_rotated_ship_cursor", return_value=cursor), patch.object(self.task, "_find_area", side_effect=lambda name, box: (area_a, "gray") if name == "Area-A" else (None, None)), patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: enemy_base if name == "Enemy-Base" else None), patch.object(self.task, "send_key_down") as key_down, patch.object(self.task, "send_key_up") as key_up, patch.object(self.task, "click") as click, patch.object(self.task, "_close_map"), patch.object(self.task, "log_info"):  # 模拟二十二号截图这类两点图同时出现占领区误识别。
            self.task._handle_map()  # 执行一次地图航点选择。
        self.assertEqual([call(enemy_base, after_sleep=3), call(875, 295, name="opposite-map-side", after_sleep=3)], click.call_args_list)  # 先点击更高分的敌方基地，再按住 Shift 追加地图对侧目标点。
        key_down.assert_called_once_with("shift", after_sleep=0.5)  # 确认追加对侧航点前按住 Shift。
        key_up.assert_called_once_with("shift", after_sleep=1)  # 确认追加完成后释放 Shift。

    def test_map_prefers_higher_scoring_areas_over_enemy_base(self):  # 验证占领区分数更高时忽略同时命中的敌方基地。
        map_overview = Box(0, 0, 300, 300, name="Map-Overview")  # 构造主地图范围。
        cursor = Box(0, 0, 10, 10, name="My-Ship-Cursor")  # 构造舰船光标位置。
        area_d = Box(20, 0, 10, 10, confidence=0.96, name="Area-D")  # 构造更高分的灰色占领区。
        enemy_base = Box(200, 200, 20, 20, confidence=0.80, name="Enemy-Base")  # 构造较低分的敌方基地误识别。
        with patch.object(self.task, "wait_until", return_value=True), patch.object(self.task, "next_frame"), patch.object(self.task, "get_box_by_name", return_value=map_overview), patch.object(self.task, "_find_rotated_ship_cursor", return_value=cursor), patch.object(self.task, "_find_area", side_effect=lambda name, box: (area_d, "gray") if name == "Area-D" else (None, None)), patch.object(self.task, "find_one", side_effect=lambda name, **kwargs: enemy_base if name == "Enemy-Base" else None), patch.object(self.task, "send_key_down") as key_down, patch.object(self.task, "send_key_up") as key_up, patch.object(self.task, "click") as click, patch.object(self.task, "_close_map"), patch.object(self.task, "log_info"):  # 模拟四点图同时出现敌方基地误识别。
            self.task._handle_map()  # 执行一次地图航点选择。
        self.assertEqual([call(area_d, after_sleep=3), call(295, 295, name="opposite-map-side", after_sleep=3)], click.call_args_list)  # 先按住 Shift 点击保留的更高分占领区，再把地图对侧追加为最后的目标点。
        key_down.assert_called_once_with("shift", after_sleep=0.5)  # 确认选中占领区时按住 Shift 追加航点。
        key_up.assert_called_once_with("shift", after_sleep=1)  # 确认点击完成后释放 Shift。

    def test_map_appends_opposite_point_after_rejected_area_click(self):  # 模拟陆地占领点不接受航点时仍把对侧点追加为最后的目标点。
        map_overview = Box(100, 100, 800, 600, name="Map-Overview")  # 使用非零地图起点验证对侧坐标计算。
        cursor = Box(195, 195, 10, 10, name="My-Ship-Cursor")  # 本舰中心为二百乘二百，对侧应为八百乘六百。
        for letter in "ABCD":  # 四种占领区都验证追加顺序。
            with self.subTest(area=letter):  # 标出出现回归的占领区字母。
                area = Box(300, 300, 10, 10, confidence=0.95, name=f"Area-{letter}")  # 模拟被识别出来但实际位于陆地的目标。
                route = []  # 保存模拟游戏当前接受的航路。
                events = MagicMock()  # 记录按住 Shift、点击与关闭地图的先后顺序。
                def accept_water_only(*args, **kwargs):  # 模拟游戏仅接受对侧水面位置，忽略陆地点选。
                    if kwargs.get("name") == "opposite-map-side":  # 只有对侧点模拟为可通航水面。
                        route[:] = args  # 设置航路，验证陆地点选没有覆盖它。
                with patch.object(self.task, "wait_until", return_value=True), patch.object(self.task, "next_frame"), patch.object(self.task, "get_box_by_name", return_value=map_overview), patch.object(self.task, "_find_rotated_ship_cursor", return_value=cursor), patch.object(self.task, "_find_area", side_effect=lambda name, box: (area, "gray") if name == area.name else (None, None)), patch.object(self.task, "find_one", return_value=None), patch.object(self.task, "send_key_down") as key_down, patch.object(self.task, "send_key_up") as key_up, patch.object(self.task, "click", side_effect=accept_water_only) as click, patch.object(self.task, "_close_map") as close, patch.object(self.task, "log_info"):  # 仅运行真实导航决策，不向游戏发送输入。
                    events.attach_mock(key_down, "send_key_down")  # 捕获按住 Shift 的顺序。
                    events.attach_mock(click, "click")  # 捕获点击顺序及等待参数。
                    events.attach_mock(key_up, "send_key_up")  # 捕获释放 Shift 的顺序。
                    events.attach_mock(close, "close")  # 确认按住 Shift 的选点完成后才关图。
                    self.assertTrue(self.task._handle_map())  # 导航流程仍应正常完成。
                self.assertEqual([800, 600], route)  # 陆地目标无效时仍由对侧点提供有效航路。
                self.assertEqual([call.send_key_down("shift", after_sleep=0.5), call.click(area, after_sleep=3), call.click(800, 600, name="opposite-map-side", after_sleep=3), call.send_key_up("shift", after_sleep=1), call.close()], events.mock_calls)  # 严格验证先按住 Shift 点占领区、再追加对侧、最后释放并关图。

    @unittest.skipUnless(all(os.path.isfile(os.path.join("ok_templates", "21x9", f"{name}.png")) for name in (14, 17, 19)), "Rotated cursor reference screenshots are not available.")  # 仅在三张地图参考截图齐全时运行旋转匹配验证。
    def test_rotated_ship_cursor_matches_reference_maps(self):  # 验证不同朝向的舰船光标都能通过旋转模板识别。
        matching_config = config["template_matching"]  # 读取应用真实模板引擎参数。
        feature_set = FeatureSet(False, matching_config["coco_feature_json"], default_horizontal_variance=matching_config["default_horizontal_variance"], default_vertical_variance=matching_config["default_vertical_variance"], default_threshold=matching_config["default_threshold"])  # 创建与正式任务一致的模板引擎。
        expected_centers = {14: (1852, 827), 17: (2118, 1430)}  # 十九号图已替换为奖励页面，保留两张现有地图的独立光标位置验证。
        self.task.executor.feature_set = feature_set  # 将任务连接到真实模板引擎以执行旋转匹配。
        self.task.executor.device_manager.supported_ratio = 5120 / 2160  # 模拟正式超宽屏窗口比例。
        for image_number, expected_center in expected_centers.items():  # 逐张验证现有不同朝向的地图截图。
            with self.subTest(image=f"{image_number}.png"):  # 在测试失败信息中保留具体截图编号。
                frame = make_bottom_right_black(cv2.imread(os.path.join("ok_templates", "21x9", f"{image_number}.png")))  # 按正式截图预处理方式读取完整参考画面。
                self.task.executor.frame = frame  # 把当前参考截图设置为任务正在处理的最新帧。
                self.task.executor.method.width = frame.shape[1]  # 提供全屏搜索框计算所需的画面宽度。
                self.task.executor.method.height = frame.shape[0]  # 提供全屏搜索框计算所需的画面高度。
                cursor = self.task._find_rotated_ship_cursor()  # 使用正式旋转模板逻辑查找舰船光标。
                self.assertIsNotNone(cursor)  # 每张参考截图都必须成功识别出一个舰船光标。
                self.assertGreaterEqual(cursor.confidence, 0.7)  # 确认原始分辨率匹配结果达到旋转光标的专用阈值。
                center_x, center_y = cursor.center()  # 读取匹配框中心以排除右下角小地图中的相似图标。
                self.assertAlmostEqual(expected_center[0], center_x, delta=40)  # 确认匹配结果位于主地图光标的水平位置附近。
                self.assertAlmostEqual(expected_center[1], center_y, delta=40)  # 确认匹配结果位于主地图光标的垂直位置附近。

    def test_map_uses_center_opposite_point_when_no_target_is_recognized(self):  # 验证没有区域或基地时会导航到舰船所在位置的地图另一侧。
        map_overview = Box(100, 100, 800, 600, name="Map-Overview")  # 构造左上角为一百且大小为八百乘六百的主地图范围。
        cases = ((Box(495, 590, 10, 10, name="bottom"), (500, 205)), (Box(195, 195, 10, 10, name="top-left"), (800, 600)))  # 定义下方去上方以及左上去右下的中心对称坐标。
        for cursor, expected_point in cases:  # 逐一验证用户指定的两个方向示例。
            with self.subTest(cursor=cursor.name):  # 在失败信息中标明当前舰船起始方位。
                with patch.object(self.task, "wait_until", return_value=True), patch.object(self.task, "next_frame"), patch.object(self.task, "get_box_by_name", return_value=map_overview), patch.object(self.task, "_find_rotated_ship_cursor", return_value=cursor), patch.object(self.task, "_find_area", return_value=(None, None)), patch.object(self.task, "find_one", return_value=None), patch.object(self.task, "click") as click, patch.object(self.task, "_close_map"):  # 模拟没有任何区域或基地匹配的地图。
                    self.task._handle_map()  # 执行地图对侧航点选择。
                click.assert_called_once_with(*expected_point, name="opposite-map-side", after_sleep=3)  # 确认点击中心对称位置后等待三秒再关闭地图。

    def test_close_map_uses_m_when_escape_does_not_close_it(self):  # 验证 ESC 未关闭大地图时使用 M 键兜底。
        with patch.object(self.task, "send_key") as send_key, patch.object(self.task, "wait_until", side_effect=(None, True)) as wait_until, patch.object(self.task, "log_warning"), patch.object(self.task, "log_error"):  # 模拟 ESC 等待超时而 M 键成功关闭地图。
            self.assertTrue(self.task._close_map())  # 执行带两阶段等待确认的地图关闭流程。
        self.assertEqual([call("esc", after_sleep=1), call("m", after_sleep=1)], send_key.call_args_list)  # 确认先按 ESC 等待再用 M 恢复。
        self.assertEqual(2, wait_until.call_count)  # 确认 ESC 和 M 之后都执行了最长八秒的状态等待。

    def test_close_map_does_not_send_m_when_escape_returns_to_battle(self):  # 验证 ESC 在延长等待内生效时不会多按 M 重新打开地图。
        with patch.object(self.task, "send_key") as send_key, patch.object(self.task, "wait_until", return_value=True), patch.object(self.task, "log_warning"):  # 模拟 ESC 后地图特征在等待过程中正常消失。
            self.assertTrue(self.task._close_map())  # 执行正常地图关闭流程。
        send_key.assert_called_once_with("esc", after_sleep=1)  # 确认正常返回战斗后不会发送多余的地图切换键。








    def test_navigation_sends_exactly_ten_forward_keys_once(self):  # 验证单次航行初始化只发送十次前进键。
        events = MagicMock()  # 记录等待、重新识别和地图操作的先后顺序。
        with patch.object(self.task, "sleep") as sleep, patch.object(self.task, "_detect_scene", return_value="battle") as detect, patch.object(self.task, "log_info"), patch.object(self.task, "send_key") as send_key, patch.object(self.task, "_handle_map", return_value=True) as navigate:  # 隔离真实等待、截图和输入。
            for name, mock in (("sleep", sleep), ("detect", detect), ("key", send_key), ("navigate", navigate)):  # 将各个操作接到同一个有序记录器。
                events.attach_mock(mock, name)  # 后续断言可检查是否在二十五秒前误开地图。
            self.assertTrue(self.task._initialize_battle_navigation())  # 执行一次战斗航行初始化。
        forward_calls = [call for call in send_key.call_args_list if call.args == ("w",)]  # 筛选所有发送 W 键的调用。
        self.assertEqual(10, len(forward_calls))  # 确认前进键严格发送十次。
        send_key.assert_any_call("m", after_sleep=2)  # 确认十次前进后仍会发送 M 键打开地图。
        self.assertEqual([call.detect()] + [call.key("w", after_sleep=0.05)] * 10 + [call.key("m", after_sleep=2), call.navigate()], events.mock_calls)  # 每次导航重试都先确认画面；本场等待由外层循环统一处理。

    def test_navigation_wait_does_not_send_input_after_scene_changes(self):
        for scene in ("unknown", "leave_battle", "result", "menu", "map"):
            with self.subTest(scene=scene), patch.object(self.task, "sleep") as sleep, patch.object(self.task, "_detect_scene", return_value=scene), patch.object(self.task, "log_info"), patch.object(self.task, "send_key") as keys, patch.object(self.task, "_handle_map") as navigate:
                self.assertFalse(self.task._initialize_battle_navigation())
            sleep.assert_not_called()
            keys.assert_not_called()
            navigate.assert_not_called()

    def test_initial_map_returns_to_battle_before_delayed_navigation(self):
        events = MagicMock()
        with patch("src.tasks.AutoPveBattleTask.time.monotonic", return_value=0), patch.object(self.task, "_detect_scene", side_effect=("map", "battle", "battle", "result")), patch.object(self.task, "_close_map", return_value=True) as close, patch.object(self.task, "sleep") as sleep, patch.object(self.task, "log_info"), patch.object(self.task, "send_key") as keys, patch.object(self.task, "_handle_map", return_value=True) as navigate:
            for name, mock in (("close", close), ("sleep", sleep), ("key", keys), ("navigate", navigate)):
                events.attach_mock(mock, name)
            self.assertTrue(self.task._run_until_result())
        self.assertEqual([call.close(), call.sleep(25)] + [call.key("w", after_sleep=0.05)] * 10 + [call.key("m", after_sleep=2), call.navigate()], events.mock_calls)

    def test_rejoined_battle_waits_twenty_five_seconds_again(self):
        with patch("src.tasks.AutoPveBattleTask.time.monotonic", return_value=0), patch.object(self.task, "_detect_scene", side_effect=("battle", "battle", "leave_battle", "battle", "battle", "result")), patch.object(self.task, "_handle_leave_battle", return_value="left"), patch.object(self.task, "sleep") as sleep, patch.object(self.task, "log_info"), patch.object(self.task, "send_key"), patch.object(self.task, "_handle_map", return_value=True) as navigate:
            self.assertEqual("left", self.task._run_until_result())
            self.assertTrue(self.task._run_until_result())
        self.assertEqual([call(25), call(25)], sleep.call_args_list)
        self.assertEqual(2, navigate.call_count)

    def test_repeated_scene_misses_do_not_restart_entry_delay(self):  # 复现日志中等待结束后三次铭牌漏识别。
        elapsed = [100.0]
        scenes = iter(("battle", "unknown", "unknown", "battle", "unknown", "battle", "unknown", "battle", "battle", "result"))
        def sleep(seconds):
            elapsed[0] += seconds
        def detect():
            elapsed[0] += .2  # 将识别耗时计入本场已经经过的时间。
            return next(scenes)
        with patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: elapsed[0]), \
                patch.object(self.task, "_detect_scene", side_effect=detect), \
                patch.object(self.task, "sleep", side_effect=sleep) as waits, \
                patch.object(self.task, "send_key") as keys, \
                patch.object(self.task, "_handle_map", return_value=True) as navigate, \
                patch.object(self.task, "log_info") as logs:
            self.assertTrue(self.task._run_until_result())
        self.assertEqual([call(25), call(1)], waits.call_args_list)
        self.assertLess(elapsed[0] - 100, 30)  # 多次漏识别也不会变成四轮共一百秒的等待。
        self.assertEqual(1, sum("本场等待" in c.args[0] for c in logs.call_args_list))
        self.assertEqual([call("w", after_sleep=.05)] * 10 + [call("m", after_sleep=2)], keys.call_args_list)
        navigate.assert_called_once()

    def test_failed_map_retry_keeps_expired_entry_deadline(self):
        elapsed = [0.0]
        with patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: elapsed[0]), \
                patch.object(self.task, "_detect_scene", side_effect=("battle", "battle", "battle", "battle", "result")), \
                patch.object(self.task, "sleep", side_effect=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)) as sleep, \
                patch.object(self.task, "send_key"), \
                patch.object(self.task, "_handle_map", side_effect=(False, True)) as navigate:
            self.assertTrue(self.task._run_until_result())
        sleep.assert_called_once_with(25)
        self.assertEqual(2, navigate.call_count)

    def test_each_new_run_until_result_gets_its_own_entry_delay(self):  # 结算续战和任务重启都不继承上一场截止时间。
        elapsed = [0.0]
        with patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: elapsed[0]), \
                patch.object(self.task, "_detect_scene", side_effect=("battle", "battle", "result") * 2), \
                patch.object(self.task, "sleep", side_effect=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds)) as sleep, \
                patch.object(self.task, "send_key"), patch.object(self.task, "_handle_map", return_value=True):
            self.assertTrue(self.task._run_until_result())
            self.assertTrue(self.task._run_until_result())
        self.assertEqual([call(25), call(25)], sleep.call_args_list)

    def test_entry_wait_still_blocks_input_if_battle_ends_during_delay(self):
        for scene in ("unknown", "leave_battle", "result", "menu", "map"):
            with self.subTest(scene=scene), patch("src.tasks.AutoPveBattleTask.time.monotonic", return_value=0), \
                    patch.object(self.task, "_detect_scene", side_effect=("battle", scene, "result")), \
                    patch.object(self.task, "sleep") as sleep, patch.object(self.task, "send_key") as keys, \
                    patch.object(self.task, "_handle_map") as navigate:
                self.assertTrue(self.task._run_until_result())
            sleep.assert_called_once_with(25)
            keys.assert_not_called()
            navigate.assert_not_called()

    def test_battle_actions_rotate_left_click_r_t_and_f(self):  # 验证战斗输入严格按鼠标左键、R、T、F 循环发送。
        action_index = 0  # 从循环中的鼠标左键位置开始。
        with patch.object(self.task, "move_relative") as move, patch("src.tasks.AutoPveBattleTask.random.uniform", side_effect=[0.4, 0.5] * 5), patch.object(self.task, "click_relative") as click_relative, patch.object(self.task, "send_key") as send_key:  # 隔离真实鼠标和键盘输入。
            for _ in range(5):  # 连续执行五次以覆盖一轮以及下一轮的首项。
                action_index = self.task._send_battle_action(action_index)  # 发送当前动作并保存下一循环位置。
        self.assertEqual([call(0.4, 0.5)] * 5, move.call_args_list)
        self.assertEqual(1, action_index)  # 第五次左键后下一项应再次轮到 R。
        self.assertEqual([call(0.5, 0.5, move=False, name="battle_fire"), call(0.5, 0.5, move=False, name="battle_fire")], click_relative.call_args_list)  # 确认第一和第五次输入都是屏幕中心左键。
        self.assertEqual([call("r"), call("t"), call("f")], send_key.call_args_list)  # 确认三个键盘输入按 R、T、F 的顺序各发送一次。

    def test_battle_action_loop_waits_one_second_between_inputs(self):  # 验证战斗状态机在每项轮换输入后固定等待一秒。
        scenes = iter(("battle", "battle", "battle", "battle", "battle", "result"))  # 模拟初始化后连续四个战斗输入周期并进入结算页。
        with patch.object(self.task, "_detect_scene", side_effect=lambda: next(scenes)), patch.object(self.task, "_initialize_battle_navigation"), patch.object(self.task, "_send_battle_action", side_effect=(1, 2, 3, 0)) as send_action, patch("src.tasks.AutoPveBattleTask.time.monotonic", return_value=0), patch.object(self.task, "sleep") as sleep:  # 隔离导航、输入和真实等待。
            self.assertTrue(self.task._run_until_result())  # 运行状态机直到模拟的结算页。
        self.assertEqual([call(0), call(1), call(2), call(3)], send_action.call_args_list)  # 确认状态机依次推进左键、R、T、F 四个输入位置。
        self.assertEqual([call(25), call(1), call(1), call(1), call(1)], sleep.call_args_list)  # 确认每个输入周期后都固定等待一秒。

    def test_battle_cycle_includes_recognition_time_and_never_catches_up(self):
        for recognition_seconds, expected_sleep in ((0.8, 0.2), (1.4, 0)):
            with self.subTest(recognition_seconds=recognition_seconds):
                clock = [0]
                scenes = iter(("battle", "battle", "result"))

                def detect():
                    clock[0] += recognition_seconds
                    return next(scenes)

                with patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: clock[0]), patch.object(self.task, "_detect_scene", side_effect=detect), patch.object(self.task, "_initialize_battle_navigation", return_value=True), patch.object(self.task, "_send_battle_action", return_value=1) as action, patch.object(self.task, "sleep") as sleep:
                    self.assertTrue(self.task._run_until_result())
                action.assert_called_once_with(0)
                self.assertEqual(2, sleep.call_count)
                self.assertAlmostEqual(expected_sleep, sleep.call_args.args[0])

    def test_random_mouse_movement_only_runs_on_confirmed_battle_frame(self):
        with patch.object(self.task, "_detect_scene", side_effect=("battle", "battle", "unknown", "map", "queue", "result")), patch.object(self.task, "_initialize_battle_navigation", return_value=True), patch.object(self.task, "_close_map", return_value=True), patch.object(self.task, "sleep"), patch.object(self.task, "click_relative"), patch.object(self.task, "move_relative") as move, patch("src.tasks.AutoPveBattleTask.random.uniform", side_effect=(0.4, 0.5)) as uniform:
            self.assertTrue(self.task._run_until_result())
        self.assertEqual([call(0.35, 0.65), call(0.35, 0.60)], uniform.call_args_list)
        move.assert_called_once_with(0.4, 0.5)

    def test_start_button_does_not_initialize_without_battle_markers(self):  # 验证点击开始后的未知画面不会靠计时初始化战斗。
        scenes = iter(("battle_start", "unknown", "result"))  # 模拟点击开始后 HUD 一直无法模板识别再进入结算的状态序列。
        with patch.object(self.task, "_detect_scene", side_effect=lambda: next(scenes)), patch.object(self.task, "find_one", return_value=Box(1, 1, 10, 10)), patch.object(self.task, "click"), patch.object(self.task, "sleep"), patch.object(self.task, "_initialize_battle_navigation") as initialize, patch("src.tasks.AutoPveBattleTask.time.monotonic", return_value=0):  # 隔离等待和真实输入。
            self.assertTrue(self.task._run_until_result())  # 确认状态机最终能够继续运行到结算页。
        initialize.assert_not_called()  # 未检测到战斗标志时不能初始化。

    def test_unknown_after_initialization_does_not_fire(self):  # 验证初始化后丢失战斗标志也会停止输入。
        with patch.object(self.task, "_detect_scene", side_effect=("battle", "unknown", "result")), patch.object(self.task, "_initialize_battle_navigation"), patch.object(self.task, "sleep"), patch.object(self.task, "_send_battle_action") as fire:  # 模拟初始化后转入未知画面。
            self.assertTrue(self.task._run_until_result())  # 等待后仍能处理结算。
        fire.assert_not_called()  # 不向未知画面发送战斗输入。

    def test_area_scores_are_logged_without_accepting_low_scores(self):  # 验证日志保留低分候选且阈值边界保持一致。
        self.task.config["Template Threshold"] = 0.7  # 使用用户当前配置验证实际生效阈值。
        feature = MagicMock(mat=np.zeros((20, 12, 3), dtype=np.uint8))  # 提供颜色转换所需模板。
        matches = [Box(10, 10, 12, 20, confidence=score, name="Area-D") for score in (0.69, 0.2, 0.7)]  # 只有灰色刚好达到阈值。
        with patch.object(self.task, "get_feature_by_name", return_value=feature), patch.object(self.task, "find_one", side_effect=matches), patch.object(self.task, "log_info") as log:  # 控制候选并收集真实日志文本。
            self.assertEqual((matches[2], "gray"), self.task._find_area("Area-D", Box(0, 0, 300, 300)))  # 低分候选不能改变识别结果。
        messages = [entry.args[0] for entry in log.call_args_list]  # 汇总三个颜色候选的日志。
        self.assertIn("[未命中] Area-D (green): 分数 69.00%，阈值 70.00%", messages[0])  # 保留未通过候选的分数。
        self.assertIn("[命中] Area-D (gray): 分数 70.00%，阈值 70.00%", messages[2])  # 记录成功候选及实际阈值。

    def test_leave_battle_stops_firing_and_does_not_click_leave_button(self):  # 验证识别到离开战斗入口后立即停止开火循环。
        scenes = iter(("leave_battle",))  # 模拟当前已经出现离开战斗入口。
        with patch.object(self.task, "_detect_scene", side_effect=lambda: next(scenes)), patch.object(self.task, "_handle_leave_battle", return_value="continued") as handle_leave, patch.object(self.task, "_send_battle_action") as send_action, patch.object(self.task, "wait_click_feature") as wait_click:  # 隔离击沉处理和战斗输入。
            self.assertEqual("continued", self.task._run_until_result(True))  # 场次未满时击沉后续战应直接结束本场循环。
        handle_leave.assert_called_once_with(True)  # 确认把未满场次的信息交给击沉处理。
        send_action.assert_not_called()  # 确认不会启动每秒鼠标左键、R、T 的战斗输入。
        wait_click.assert_not_called()  # 确认不会点击离开战斗入口本身。

    def test_rejoined_battle_runs_navigation_initialization_again(self):  # 验证确认离开后重新加入会初始化新一场战斗。
        scenes = iter(("battle", "leave_battle", "battle", "result"))  # 模拟当前战斗、击沉离开、新战斗和最终结算的状态序列。
        with patch.object(self.task, "sleep"), patch.object(self.task, "_detect_scene", side_effect=lambda: next(scenes)), patch.object(self.task, "_handle_leave_battle", return_value="left") as leave, patch.object(self.task, "_initialize_battle_navigation") as initialize:  # 隔离实际按钮点击和键盘地图操作。
            self.assertEqual("left", self.task._run_until_result(True))  # 本场离开后先计数，再建立下一场会话。
            self.assertTrue(self.task._run_until_result(True))
        leave.assert_called_once_with(True)  # 确认击沉页面只执行一次 ESC 后续处理。
        self.assertEqual(2, initialize.call_count)  # 确认新一场战斗不会沿用上一场的航行初始化状态。

    def test_run_brings_game_to_foreground(self):  # 验证启动任务后会先把游戏窗口切换到前台。
        with patch.object(self.task, "ensure_in_front") as bring_front, patch.object(self.task, "_return_to_main", return_value=False), patch.object(self.task, "log_info"), patch.object(self.task, "log_error"):  # 隔离窗口切换和后续准备流程，只验证启动动作。
            self.task.run()  # 执行会在无法回到主界面时提前结束的任务主流程。
        bring_front.assert_called_once()  # 确认任务一开始就会把游戏切到前台。




    def test_run_closes_game_after_success_when_enabled(self):  # 验证用户开启开关后仅在达到目标场数时关闭游戏。
        self.task.config["Close Game After Completion"] = True  # 模拟用户在任务配置中开启完成后关闭游戏。
        with patch.object(self.task, "ensure_in_front"), patch.object(self.task, "_return_to_main", return_value=True), patch.object(self.task, "_prepare_and_join_first_battle", return_value=True), patch.object(self.task, "_run_until_result", return_value="left"), patch.object(self.task, "wait_click_feature") as wait_click, patch.object(self.task, "log_info"), patch.object(self.task, "log_error"):  # 模拟最后一场击沉后已经确认离开并隔离真实输入。
            with patch.object(self.task, "_collect_containers", return_value=True) as collect:  # 模拟回港后的集装箱已经领完。
                self.task.run()  # 执行一场完整成功流程。
            collect.assert_called_once_with()  # 最后一场击沉离开也必须领取集装箱。
        wait_click.assert_not_called()  # 已经从击沉后续页返回港口时不再点击结算按钮。
        self.executor.device_manager.stop_hwnd.assert_called_once_with()  # 确认只通过框架关闭当前绑定的游戏一次。

    def test_run_does_not_close_game_when_task_fails(self):  # 验证即使开关已开启，任务未成功达到场数时也不会误关游戏。
        self.task.config["Close Game After Completion"] = True  # 模拟用户已经开启完成后关闭游戏。
        with patch.object(self.task, "ensure_in_front"), patch.object(self.task, "_return_to_main", return_value=False), patch.object(self.task, "log_info"), patch.object(self.task, "log_error"):  # 模拟任务在准备前无法返回主界面。
            self.task.run()  # 执行会提前失败的任务流程。
        self.executor.device_manager.stop_hwnd.assert_not_called()  # 未成功完成战斗时不得关闭游戏。

    def test_run_skips_result_clicks_when_sunk_flow_already_acted(self):  # 验证击沉后续战或离开后不会再点结算页按钮。
        self.task.config["Battle Count"] = 2  # 将本次测试目标设置为两场战斗。
        outcomes = iter(("continued", "left"))  # 模拟第一场击沉后续战、第二场确认离开返回港口。
        with patch.object(self.task, "ensure_in_front"), patch.object(self.task, "_return_to_main", return_value=True), patch.object(self.task, "_prepare_and_join_first_battle", return_value=True), patch.object(self.task, "_run_until_result", side_effect=lambda can_continue=True: next(outcomes)), patch.object(self.task, "wait_click_feature") as wait_click, patch.object(self.task, "log_info"), patch.object(self.task, "log_error"):  # 隔离真实游戏输入并运行击沉后的计数逻辑。
            with patch.object(self.task, "_collect_containers", return_value=True) as collect:  # 隔离两场结束后的领取输入。
                self.task.run()  # 执行设置为两场且都走击沉后续页面的任务主流程。
            collect.assert_called_once_with()  # 第一场续战不触发领取，最后一场离开后才领取。
        wait_click.assert_not_called()  # 确认已经点击过继续或确认离开后不会再点结算页按钮。


    def test_find_one_uses_fourfold_local_search_for_continue_buttons(self):  # 验证两个继续按钮均围绕标注中心搜索且保留显式范围。
        feature = Box(300, 200, 100, 40)  # 构造标注中心位于三百五十、二百二十的按钮。
        for feature_name in ("Continue-Battle", "Continue-Battle-After-Sunk"):  # 同时覆盖结算页与击沉后续页。
            with self.subTest(feature_name=feature_name), patch("src.tasks.feature_ocr.annotated_box", return_value=feature):  # 隔离坐标元数据，验证 OCR 的默认范围。
                region = search_box(feature_name, np.zeros((1080, 2560, 3), dtype=np.uint8))  # 根据实际截图尺寸裁剪搜索框。
                self.assertEqual((150, 140, 400, 160), (region.x, region.y, region.width, region.height))  # 宽高各四倍且中心不变。
                explicit_box = Box(10, 20, 200, 80)  # 模拟调用方指定更小的搜索范围。
                with patch("src.tasks.AutoPveBattleTask.find_ocr_feature", return_value=None) as find_ocr:  # 验证显式范围传给 OCR，不再进入模板匹配。
                    self.task.find_one(feature_name, threshold=0.8, box=explicit_box)
                self.assertIs(explicit_box, find_ocr.call_args.args[4])  # 显式范围原样保留。

    def test_resolution_packs_pick_coco_by_window_ratio(self):  # 验证 21:9、16:10、16:9 窗口会选中对应的子目录。
        self.assertEqual(os.path.join("assets", "21x9", "coco_annotations.json"), coco_json_for_size(5120, 2160))  # 超宽屏使用 21:9 标注。
        self.assertEqual(os.path.join("assets", "16x10", "coco_annotations.json"), coco_json_for_size(2560, 1600))  # 十六比十窗口使用对应标注。
        self.assertEqual(os.path.join("assets", "16x9", "coco_annotations.json"), coco_json_for_size(1920, 1080))  # 十六比九窗口使用对应标注。
        self.assertEqual(os.path.join("ok_templates", "21x9"), template_folder_for_size(5120, 2160))  # 超宽屏开发截图保存在 21x9 子目录。
        self.assertEqual(os.path.join("ok_templates", "16x10"), template_folder_for_size(2560, 1600))  # 十六比十开发截图保存在 16x10 子目录。
        self.assertEqual(os.path.join("ok_templates", "16x9"), template_folder_for_size(1920, 1080))  # 十六比九开发截图保存在 16x9 子目录。
        self.assertEqual(os.path.join("assets", "21x9"), asset_folder_for_size(5120, 2160))  # 超宽屏压缩保存写入 21x9。
        self.assertEqual(os.path.join("assets", "16x10"), asset_folder_for_size(2560, 1600))  # 十六比十压缩保存写入 16x10。
        self.assertEqual(os.path.join("assets", "16x9"), asset_folder_for_size(1920, 1080))  # 十六比九压缩保存写入 16x9。
        self.assertEqual("21:9", pack_for_size(2560, 1080)["ratio"])  # 常见超宽屏分辨率归入 21:9。
        self.assertEqual("16:9", pack_for_size(2560, 1440)["ratio"])  # 常见十六比九分辨率归入 16:9。
        self.assertTrue(ratio_is_supported(5120, 2160))  # 确认超宽屏比例被接受。
        self.assertTrue(ratio_is_supported(2560, 1600))  # 确认十六比十比例被接受。
        self.assertTrue(ratio_is_supported(1920, 1080))  # 确认十六比九比例被接受。
        self.assertFalse(ratio_is_supported(1280, 1024))  # 确认未支持的比例不会被当成已知比例。
        self.assertEqual(os.path.abspath(os.path.join("assets", "16x10")), redirect_asset_target("assets", os.path.join("ok_templates", "16x10")))  # 从十六比十模板目录保存时写入 16x10。
        self.assertEqual(os.path.abspath(os.path.join("assets", "21x9")), redirect_asset_target("assets", os.path.join("ok_templates", "21x9")))  # 从超宽屏模板目录保存时写入 21x9。
        self.assertEqual(os.path.abspath(os.path.join("assets", "16x9")), redirect_asset_target("assets", os.path.join("ok_templates", "16x9")))  # 从十六比九模板目录保存时写入 16x9。
        self.assertEqual(os.path.join("ok_tasks", "assets"), redirect_asset_target(os.path.join("ok_tasks", "assets"), os.path.join("ok_templates", "16x10")))  # 自定义脚本目录保持原样。
        set_pack_override("16:9")  # 模拟截图页手动选十六比九。
        self.assertEqual(os.path.join("ok_templates", "16x9"), current_template_folder())  # 确认开发截图目录跟随按钮选择。
        self.assertEqual(os.path.join("assets", "21x9", "coco_annotations.json"), coco_json_for_size(5120, 2160))  # 确认运行时匹配仍按真实窗口比例，不受截图页按钮影响。
        set_pack_override(None)  # 清掉手动选择以免影响后续测试。

    def test_feature_set_switches_coco_json_by_frame_ratio(self):  # 验证模板引擎会按当前帧比例切换 coco 文件。
        matching_config = config["template_matching"]  # 读取正式模板配置。
        feature_set = FeatureSet(False, matching_config["coco_feature_json"], default_horizontal_variance=matching_config["default_horizontal_variance"], default_vertical_variance=matching_config["default_vertical_variance"], default_threshold=matching_config["default_threshold"])  # 创建与正式任务一致的模板引擎。
        feature_set.check_size(np.zeros((2160, 5120, 3), dtype=np.uint8))  # 先用超宽屏尺寸触发一次加载路径选择。
        self.assertIn("21x9", feature_set.coco_json.replace("\\", "/"))  # 确认超宽屏使用 21x9 coco。
        feature_set.check_size(np.zeros((1600, 2560, 3), dtype=np.uint8))  # 再改用十六比十尺寸。
        self.assertIn("16x10", feature_set.coco_json.replace("\\", "/"))  # 确认已经切换到十六比十模板。
        feature_set.check_size(np.zeros((1080, 1920, 3), dtype=np.uint8))  # 再改用十六比九尺寸。
        self.assertIn("16x10", feature_set.coco_json.replace("\\", "/"))  # 十六比九标注还不存在时继续使用上一套可用模板。

    def test_both_asset_packs_use_current_feature_names(self):  # 验证已有两套标注使用同一套当前器件名。
        import json  # 仅在本测试中读取两份 coco 分类名。
        current_names = {category["name"] for category in json.loads(Path("assets/21x9/coco_annotations.json").read_text(encoding="utf-8"))["categories"]}  # 读取超宽屏分类名。
        wide_names = {category["name"] for category in json.loads(Path("assets/16x10/coco_annotations.json").read_text(encoding="utf-8"))["categories"]}  # 读取十六比十分类名。
        optional_names = set(AutoPveBattleTask.OPTIONAL_FEATURES) | set(AutoPveBattleTask.MAP_VIEW_FEATURES)  # 新入口和非对称模式允许逐比例补充标注。
        self.assertTrue(set(AutoPveBattleTask.MAP_VIEW_FEATURES) <= current_names)  # 当前超宽屏资源必须提供两个地图按钮。
        self.assertEqual(current_names - optional_names - {"Ship-Icon"}, wide_names - optional_names - {"Ship-Icon"})  # 旧比例尚未补充独立图标，其余共享元素保持一致。
        self.assertIn("Continue-Battle-After-Sunk", current_names)  # 确认击沉后续按钮使用新名字。
        self.assertNotIn("Continue-Battle-Button-After-Sunk", wide_names)  # 确认旧的击沉后续按钮名已经去掉。
        self.assertNotIn("leave-battle-button", wide_names)  # 确认旧的小写按钮名已经去掉。

    def test_sixteen_by_ten_pack_matches_join_battle_on_its_own_image(self):  # 验证十六比十模板能在自己的标注图上命中。
        matching_config = config["template_matching"]  # 读取正式模板配置。
        feature_set = FeatureSet(False, matching_config["coco_feature_json"], default_horizontal_variance=matching_config["default_horizontal_variance"], default_vertical_variance=matching_config["default_vertical_variance"], default_threshold=matching_config["default_threshold"])  # 创建与正式任务一致的模板引擎。
        frame = cv2.imread("assets/16x10/images/0.png")  # 读取十六比十主界面标注图。
        boxes = feature_set.find_feature(frame, "Join-Battle", threshold=0.8, limit=1)  # 按当前器件名查找加入战斗按钮。
        self.assertTrue(boxes)  # 确认十六比十资源已经可以按新命名匹配。
        self.assertGreaterEqual(boxes[0].confidence, 0.8)  # 确认匹配分数达到正式阈值。


if __name__ == "__main__":  # 支持直接运行本测试文件。
    unittest.main()  # 启动标准库单元测试运行器。
