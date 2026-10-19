import tempfile  # 在临时目录中验证截图保存，不污染真实数据集。
import unittest  # 使用项目现有的标准库测试框架。
from datetime import datetime  # 固定系统时间以验证重复时间戳不会覆盖样本。
from pathlib import Path  # 检查截图文件名和场次目录。
from unittest.mock import MagicMock, call, patch  # 隔离游戏输入、识别和时钟。

import cv2  # 验证生成的 PNG 能够无损解码。
import numpy as np  # 构造不同颜色的战斗和大地图测试帧。

from src.tasks.AutoPveBattleTask import AutoPveBattleTask  # 测试自动战斗中的数据集采集行为。


class TestBattleDataset(unittest.TestCase):  # 覆盖计时、成对保存和地图切换异常。
    def setUp(self):  # 每个测试使用独立任务和临时数据目录。
        self.executor = MagicMock()  # 不连接真实游戏窗口。
        self.executor.scene = None  # 满足任务基类初始化要求。
        self.task = AutoPveBattleTask(self.executor, None)  # 创建实际任务类以验证新增行为。
        self.task.config = dict(self.task.default_config)  # 使用默认任务配置。
        self.task.config["Capture Battle Dataset"] = True  # 原有采集行为测试显式开启开关。
        temporary = tempfile.TemporaryDirectory()  # 为本测试分配独立文件空间。
        self.addCleanup(temporary.cleanup)  # 测试结束后仅清理本测试创建的临时目录。
        self.root = Path(temporary.name) / "中文数据集"  # 覆盖 Windows 中文路径写入。
        self.task.DATASET_DIRECTORY = self.root  # 禁止测试写入真实项目数据集。
        self.scene = "battle"  # 默认从已确认的战斗画面开始。
        self.battle_frame = np.full((24, 48, 3), (20, 80, 150), dtype=np.uint8)  # 构造战斗截图并保留非灰度颜色。
        self.map_frame = np.full((24, 48, 3), (170, 50, 10), dtype=np.uint8)  # 构造可与战斗截图区分的地图截图。
        self.executor.frame = self.battle_frame  # 任务通过执行器读取当前画面。
        self.task._detect_scene = MagicMock(side_effect=lambda **kwargs: self.scene)  # 用可变场景模拟同一帧确认。
        self.task.send_key = MagicMock(side_effect=self._toggle_map)  # 模拟真实 M 键的打开与关闭行为。
        self.task.wait_until = MagicMock(side_effect=lambda condition, **kwargs: condition())  # 立即检查模拟页面的切换结果。
        self.task.log_info = MagicMock()  # 捕获成功日志而不依赖界面。
        self.task.log_warning = MagicMock()  # 捕获失败日志而不依赖界面。

    def _toggle_map(self, key, **kwargs):  # 模拟 M 键改变视图和截图来源。
        self.assertEqual("m", key)  # 采集只能切换地图，不能发送移动或开火按键。
        self.scene = "map" if self.scene == "battle" else "battle"  # 切换地图与普通战斗界面。
        self.executor.frame = self.map_frame if self.scene == "map" else self.battle_frame  # 切换到对应的原始帧。

    def test_capture_disabled_by_default_and_for_legacy_config(self):  # 默认关闭及旧配置缺少开关时都不能触发采集。
        self.assertFalse(self.task.default_config["Capture Battle Dataset"])  # 新用户默认不采集。
        for legacy in (False, True):  # 同时验证显式关闭和旧配置未设置字段。
            with self.subTest(legacy=legacy):  # 分别报告两种配置情况。
                self.task.config["Capture Battle Dataset"] = False  # 显式关闭本次测试的采集开关。
                if legacy:  # 模拟更新前保存的配置。
                    self.task.config.pop("Capture Battle Dataset")  # 删除新增字段。
                with patch("src.tasks.AutoPveBattleTask.uuid4") as session_id:  # 监测是否错误启动采集会话。
                    self.assertEqual([], self._run_timeline([(0, "battle"), (60, "battle"), (120, "battle"), (121, "result")]))  # 战斗持续两分钟也不能采集。
                session_id.assert_not_called()  # 不创建采集会话标识。
                self.assertFalse(self.task._capture_battle_dataset(self.root))  # 直接调用也应立即返回。
        self.task._detect_scene.assert_not_called()  # 关闭后不额外执行截图识别。
        self.task.send_key.assert_not_called()  # 不执行采集用的地图切换。
        self.assertFalse(self.root.exists())  # 不创建图片输出目录。

    def test_pair_preserves_pixels_names_and_returns_to_battle(self):  # 验证两张 PNG 内容正确且采集后恢复战斗。
        directory = self.root / "battle_1"  # 为本场采集指定输出目录。
        with patch.object(self.task, "_handle_map") as navigate, patch.object(self.task, "click") as click:  # 监测是否误调用选点或鼠标操作。
            self.assertTrue(self.task._capture_battle_dataset(directory))  # 执行真实 PNG 编码及文件写入。
        files = sorted(directory.glob("*.png"))  # 查找本组生成的图片。
        self.assertEqual(2, len(files))  # 每次成功采集只生成两张图片。
        self.assertEqual(files[0].stem.removesuffix("_battle"), files[1].stem.removesuffix("_map"))  # 同组图片必须共享时间戳。
        for filename, expected in zip(files, (self.battle_frame, self.map_frame)):  # 分别验证战斗和地图截图。
            actual = cv2.imdecode(np.fromfile(str(filename), dtype=np.uint8), cv2.IMREAD_COLOR)  # 从中文路径读取并解码图片。
            np.testing.assert_array_equal(expected, actual)  # 验证无损保存原分辨率及像素，不混用旧帧。
        self.assertEqual("battle", self.scene)  # 完成后必须已经回到战斗界面。
        self.assertEqual([call("m", after_sleep=1), call("m", after_sleep=1)], self.task.send_key.call_args_list)  # 仅打开和关闭地图各一次。
        navigate.assert_not_called()  # 采集不能重新选择航点。
        click.assert_not_called()  # 采集不能在地图上点击或开火。

    def test_non_battle_scenes_do_not_capture_or_send_keys(self):  # 排队、结算、击沉和未知画面都不允许启动采集。
        for scene in ("queue", "result", "leave_battle", "unknown", "menu"):  # 覆盖需要交给外层处理的界面。
            with self.subTest(scene=scene):  # 单独标记失败的场景。
                self.scene = scene  # 模拟采集前场景发生变化。
                self.assertFalse(self.task._capture_battle_dataset(self.root))  # 当前周期应被跳过。
        self.task.send_key.assert_not_called()  # 不应为非战斗场景切换地图。
        self.assertFalse(self.root.exists())  # 不保存错误标注或空数据目录。

    def test_repeated_wall_clock_does_not_overwrite_previous_pair(self):  # 系统时间重复时保留之前采集的两张图片。
        with patch("src.tasks.AutoPveBattleTask.datetime") as clock:  # 模拟同一时间戳被重复使用。
            clock.now.return_value = datetime(2026, 9, 10, 12, 0, 0)  # 为两次采样返回相同时间。
            self.assertTrue(self.task._capture_battle_dataset(self.root))  # 保存第一组。
            previous = {path: path.read_bytes() for path in self.root.glob("*.png")}  # 记录原有文件及内容。
            self.assertTrue(self.task._capture_battle_dataset(self.root))  # 在相同时间戳下保存第二组。
        self.assertEqual(4, len(list(self.root.glob("*.png"))))  # 两组均须保留，不能覆盖文件。
        for path, contents in previous.items():  # 验证第一组内容也没有被修改。
            self.assertEqual(contents, path.read_bytes())  # 旧样本必须逐字节保留。

    def test_map_open_failure_and_mid_capture_result_skip_pair(self):  # 验证地图未打开或途中结算时不会误标图片或盲按 M。
        for scene in ("battle", "result", "leave_battle", "unknown"):  # 模拟开图失败及战斗中途结束。
            with self.subTest(scene=scene):  # 独立验证不同异常场景。
                self.scene = "battle"  # 每次采集均从正常战斗开始。
                self.task.send_key.reset_mock()  # 清除上一组场景的按键记录。
                self.task.send_key.side_effect = lambda *args, **kwargs: setattr(self, "scene", scene)  # 模拟按 M 后进入目标场景。
                self.assertFalse(self.task._capture_battle_dataset(self.root))  # 未确认大地图时跳过整组。
                self.task.send_key.assert_called_once_with("m", after_sleep=1)  # 不在战斗或结算界面盲发第二次切换。
                self.assertFalse(self.root.exists())  # 不将战斗或结算误存为大地图。

    def test_save_failure_happens_after_map_is_restored(self):  # 磁盘不可写时也应先恢复战斗界面。
        with patch.object(Path, "mkdir", side_effect=OSError("disk full")):  # 模拟文件系统失败而非游戏异常。
            self.assertFalse(self.task._capture_battle_dataset(self.root))  # 保存失败不向外抛出异常中断战斗。
        self.assertEqual("battle", self.scene)  # 确认恢复地图在磁盘写入之前完成。
        self.task.log_warning.assert_called_once()  # 明确记录本组保存失败。

    def test_failed_close_keeps_valid_pair_and_reports_recovery(self):  # 地图无法关闭时保留有效截图并交给外层恢复。
        self.task.send_key.side_effect = lambda *args, **kwargs: (setattr(self, "scene", "map"), setattr(self.executor, "frame", self.map_frame))  # 模拟打开成功但关闭无效。
        self.assertTrue(self.task._capture_battle_dataset(self.root))  # 有效的两张截图仍然可以保存。
        self.assertEqual(2, len(list(self.root.glob("*.png"))))  # 不丢弃已经确认的画面。
        self.task.log_warning.assert_called_once()  # 提示由外层继续恢复地图。

    def _run_timeline(self, timeline):  # 使用虚拟时间验证战斗循环，不实际等待一分钟。
        elapsed = [0]  # 保存当前场景对应的单调时钟。
        events = iter(timeline)  # 按顺序提供时间和场景。
        samples = []  # 记录每次采集的时间与场次目录。
        def detect_scene():  # 模拟每一轮识别到的界面及经过时间。
            elapsed[0], scene = next(events)  # 推进到当前时间点。
            return scene  # 将模拟场景交给实际战斗状态机。
        with patch.object(self.task, "_detect_scene", side_effect=detect_scene), patch("src.tasks.AutoPveBattleTask.time.monotonic", side_effect=lambda: elapsed[0]), patch.object(self.task, "_initialize_battle_navigation"), patch.object(self.task, "_handle_map"), patch.object(self.task, "_close_map", return_value=True), patch.object(self.task, "_send_battle_action", return_value=0), patch.object(self.task, "sleep"), patch.object(self.task, "_handle_leave_battle", return_value="left"), patch.object(self.task, "_capture_battle_dataset", side_effect=lambda directory: samples.append((elapsed[0], directory))):  # 隔离所有游戏操作，只执行状态机计时和重置逻辑。
            outcome = self.task._run_until_result()
            if outcome == "left":
                outcome = self.task._run_until_result()  # 下一场单独建立采集和导航会话。
            self.assertTrue(outcome)
        return samples  # 返回实际触发的采样事件供断言。

    def test_first_capture_after_sixty_seconds_then_every_minute(self):  # 排队时间不计入采样周期，采集也不打乱分钟节奏。
        samples = self._run_timeline([(0, "queue"), (100, "battle"), (159, "battle"), (160, "battle"), (161, "battle"), (219, "battle"), (220, "battle"), (221, "result")])  # 模拟排队后两分钟的战斗。
        self.assertEqual([160, 220], [time for time, _ in samples])  # 第一组在入场六十秒，第二组在一百二十秒。
        self.assertEqual(samples[0][1], samples[1][1])  # 同场截图必须写到同一目录。

    def test_entry_on_map_also_starts_timer(self):  # 用户已处于大地图时也能正确开始本场采样。
        samples = self._run_timeline([(100, "map"), (159, "battle"), (160, "battle"), (161, "result")])  # 模拟从大地图开始的战斗。
        self.assertEqual([160], [time for time, _ in samples])  # 入场地图同样启动六十秒倒计时。

    def test_pause_does_not_trigger_burst_of_missed_captures(self):  # 长时间暂停后只采当前一组，不连续开关地图补拍。
        samples = self._run_timeline([(0, "battle"), (190, "battle"), (191, "battle"), (240, "battle"), (241, "result")])  # 模拟战斗开始后暂停三分钟多。
        self.assertEqual([190, 240], [time for time, _ in samples])  # 跳过积压周期并在下一个分钟边界恢复。

    def test_rejoined_battle_resets_timer_and_directory(self):  # 击沉离开后重新加入必须使用新的时间起点和目录。
        samples = self._run_timeline([(0, "battle"), (60, "battle"), (61, "leave_battle"), (90, "queue"), (200, "battle"), (259, "battle"), (260, "battle"), (261, "result")])  # 模拟第一场采集后击沉并重开。
        self.assertEqual([60, 260], [time for time, _ in samples])  # 第二场从自己的入场时间重新等待一分钟。
        self.assertNotEqual(samples[0][1], samples[1][1])  # 不混合两场的数据。
