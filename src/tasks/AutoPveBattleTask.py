import math  # 导入数学函数以计算旋转后模板的完整边界尺寸。
import random  # 在战斗画面中央随机选择鼠标位置，避免持续悬停在舰船列表上。
import time  # 导入时间模块以限制无法识别界面的等待时长。
from datetime import datetime  # 使用本地时间为每场战斗及成对截图命名。
from pathlib import Path  # 将数据集保存在项目目录并支持中文路径。
from uuid import uuid4  # 给时间戳追加随机后缀，避免时钟重复时覆盖已有样本。

import cv2  # 导入 OpenCV 以旋转舰船光标模板。
import numpy as np  # 导入数组工具以计算模板边缘的地图背景色。
from ok import Box  # 使用截图像素坐标限制舰船图标的搜索范围。
from qfluentwidgets import FluentIcon  # 导入任务列表中使用的内置图标。

from src.tasks.MyBaseTask import MyBaseTask, preserve_input_timing  # 导入任务基类与战斗原始时序保护。
from src.tasks.feature_ocr import DEFAULT_SHIP_NAME, OCR_TEXTS, find_ocr_feature, find_text, normalized  # 共用文字识别及用户指定舰名。
from src.tasks.battle_flow import SceneProgress  # 保存状态转换、输入冷却和次数限制。
from src.task_failure_capture import save_failure_screenshot  # 专用按钮和统一退出钩子共用持久截图保存逻辑。


class AutoPveBattleTask(MyBaseTask):  # 定义自动完成 PVE 战斗的一次性任务。

    AREA_COLOR_HSV = {"green": (78, 171), "red": (6, 255), "gray": (0, 0)}  # 定义占领区绿色、红色和灰色模板使用的 OpenCV 色相与饱和度。
    MAP_TEMPLATE_THRESHOLD = 0.75  # 为大地图元素使用略低于全局默认值的专用阈值以减少动态画面漏识别。
    MAP_VIEW_FEATURES = ("Map-M-Button", "Map-B-Button")  # 两个地图按钮同时出现时确认大地图页面。
    MAP_VIEW_THRESHOLD = 0.70  # 小按钮缩小一半后受像素取整影响，配合双按钮共同确认降低漏识别。
    MAP_POINT_SETTLE_SECONDS = 3  # 点击地图航点后等待标记和路线动画稳定再尝试关闭地图。
    MAP_RETURN_TIMEOUT = 8  # 按返回键后最多等待八秒确认大地图已经消失。
    NAVIGATION_START_DELAY = 25  # 入场后等待开局提示文字消失，再打开大地图导航。
    STARTUP_TIMEOUT = 300  # 从任务开始识别游戏画面起，最多等待五分钟完成加载并到达港口。
    SCREEN_BUTTON_RETRY_INTERVAL = 5  # 同一入口按钮点击后至少等待五秒再尝试，避免加载动画期间连续点击。
    SCREEN_BUTTON_MAX_ATTEMPTS = 3  # 同一页面最多点击三次，持续无效时提前停止并保存现场。
    FAILURE_DIRECTORY = Path(__file__).resolve().parents[2] / "logs" / "task_failures"  # 失败现场独立保存，不受调试启动清空 screenshots 的影响。
    DATASET_INTERVAL = 60  # 每场战斗从确认入场起每六十秒采集一组截图。
    DATASET_DIRECTORY = Path(__file__).resolve().parents[2] / "dataset"  # 将数据集固定保存在项目根目录，避免随工作目录变化。
    BATTLE_MODES = ("PVE-Battle", "Asymmetry-Battle")  # 下拉选项直接对应正式标注名称。
    SUB_PAGES = ("battle_mode", "addon", "equipment", "container")  # 港口内可被 ESC 关闭的准备子页，转场中常被短暂误判。
    LOADOUT_CLICK_INTERVAL = 3  # 准备入口点击冷却，保证 60 秒额度内至少有十余次重试机会。
    LOADOUT_ESCAPE_INTERVAL = 3  # 准备子页 ESC 冷却，避免连续按键把两层页面一起关掉。
    LOADOUT_REOPEN_LIMIT = 2  # 加成页中途退出后最多重新打开的轮数，防止反复开关。
    SCREEN_BUTTONS = {"battle_survey": "Battle-Survey-Close", "inactivity_penalty": "Inactivity-Penalty-Close", "login": "Login-Game", "claim_reward": "Claim-Reward", "reward_screen": "Close-Reward-Screen"}  # 先处理评价与处罚覆盖页，再处理登录和奖励。
    OPTIONAL_FEATURES = (*SCREEN_BUTTONS.values(), "Popup-ESC-Close", "Control-Camera", "Asymmetry-Battle", "Container-Menu", "Pick-Container", "Confirm-Container", "No-Commander", "Recall-Commander")  # 部分比例尚未提供这些新标注，识别允许跳过。

    def __init__(self, *args, **kwargs):  # 初始化任务元数据和可配置参数。
        super().__init__(*args, **kwargs)  # 首先初始化 ok-script 的基础任务能力。
        self.name = "Auto PVE Battle"  # 设置任务列表中显示的名称。
        self.support_schedule_task = True  # 在框架计划任务界面开放定时执行，并复用任务启动前的 UU 加速流程。
        self.description = "Automatically prepares the selected ship and completes PVE or Asymmetry battles in the selected mode."  # 说明任务支持两种可选战斗模式。
        self.icon = FluentIcon.GAME  # 使用游戏图标标识该自动战斗任务。
        self.default_config.update({  # 添加任务运行时可由用户调整的配置。
            "Battle Count": 1,  # 默认完成一场战斗后停止。
            "Battle Mode": "PVE-Battle",  # 默认沿用 PVE 模式以兼容原有配置。
            "Ship Name": DEFAULT_SHIP_NAME,  # 框架用字符串默认值生成可编辑的舰船名称输入框。
            "Close Game After Completion": False,  # 默认保留游戏运行，由用户按需开启完成后关闭功能。
            "Capture Battle Dataset": False,  # 默认关闭数据集采集，仅在用户开启后定时截图。
            "Template Threshold": 0.8,  # 默认使用与项目一致的模板匹配阈值。
        })  # 完成默认配置定义。
        self.config_description.update({  # 添加配置项在界面中的帮助说明。
            "Battle Count": "Number of completed battles before the task stops.",  # 说明战斗场数的含义。
            "Battle Mode": "Select PVE or Asymmetry battle mode.",  # 说明模式选择在首场准备时生效。
            "Ship Name": "Enter the ship name shown in the game. Used for both port selection and battle recognition. Keep the ship visible in the bottom 30% of the screen.",  # 说明按当前游戏语言输入，并让目标船显示在屏幕底部 30%。
            "Close Game After Completion": "Close the game after the configured number of battles is completed.",  # 说明开关只在成功达到目标场数后关闭游戏。
            "Capture Battle Dataset": "Save battle and tactical-map screenshots every minute during battle.",  # 说明开关控制每分钟的战斗和地图截图采集。
            "Template Threshold": "Minimum confidence required for template matching.",  # 说明匹配阈值的含义。
        })  # 完成配置说明定义。
        self.config_type["Battle Mode"] = {"type": "drop_down", "options": list(self.BATTLE_MODES)}  # 使用框架原生下拉框展示两种模式。
        self.config_type["Ship Name"] = {"type": "line_edit"}  # 舰船名称使用可编辑的单行字符串输入框。

    def validate_config(self, key, value):  # 在用户保存配置时检查输入是否合法。
        if key == "Ship Name" and (not isinstance(value, str) or not normalized(value) or "\n" in value or "\r" in value):  # 船名必须为单行有效文字，拒绝空白或只有标点。
            return "Ship Name must be a non-empty single-line name."
        if key == "Battle Mode" and value not in self.BATTLE_MODES:  # 限制选择已支持的模式。
            return "Select PVE or Asymmetry battle mode."  # 返回可翻译的配置校验提示。
        if key == "Battle Count" and (not isinstance(value, int) or isinstance(value, bool) or value < 1):  # 要求战斗场数是至少为一的整数。
            return "Battle Count must be an integer greater than or equal to 1."  # 返回战斗场数的校验提示。
        if key == "Template Threshold" and (not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 < value <= 1):  # 要求阈值位于有效置信度范围内。
            return "Template Threshold must be greater than 0 and less than or equal to 1."  # 返回匹配阈值的校验提示。
        return None  # 输入合法时不返回错误信息。

    @property  # 将匹配阈值暴露为便于内部调用的只读属性。
    def threshold(self):  # 读取并转换当前模板匹配阈值。
        return float(self.config.get("Template Threshold", 0.8))  # 从任务配置中取得匹配阈值。

    @property  # 将大地图专用匹配阈值暴露为便于内部复用的只读属性。
    def map_threshold(self):  # 读取不会高于零点七五的大地图匹配阈值。
        return min(self.threshold, self.MAP_TEMPLATE_THRESHOLD)  # 用户设置更低阈值时仍尊重其配置。

    def find_one(self, feature_name=None, horizontal_variance=0, vertical_variance=0, threshold=0, **kwargs):  # 按元素分别设置局部搜索范围、模板缩放和模式按钮全图搜索。
        if isinstance(feature_name, (list, tuple)):  # 框架 wait_feature 会把多个页面状态作为列表传入。
            matches = [self.find_one(name, horizontal_variance, vertical_variance, threshold, **kwargs) for name in feature_name]  # 每个元素分别路由至 OCR 或保留的模板识别。
            return max((match for match in matches if match is not None), key=lambda match: match.confidence, default=None)  # 保持框架原有最高分选择行为。
        if feature_name in OCR_TEXTS:  # 已迁移的文字元素只使用 OCR，不回退到像素模板匹配。
            frame = kwargs.get("frame") if kwargs.get("frame") is not None else self.frame  # 整次查询和弹窗上下文均使用同一截图。
            return find_ocr_feature(self, feature_name, frame, threshold, kwargs.get("box"))  # 返回带原元素名的文字框供等待、点击及诊断共用。
        if feature_name in self.OPTIONAL_FEATURES and self.get_feature_by_name(feature_name) is None:  # 缺少某个比例的新标注时跳过查询，避免框架抛出模板缺失异常。
            return None  # 缺少模板只表示不能识别该元素，不影响其余已有流程。
        if feature_name == "Ship-Icon":  # 黄色舰船轮廓单独评分，避免天空、海面和灰色队友图标影响判断。
            kwargs.setdefault("mask_function", self._ship_icon_yellow_mask)  # 模板中只有黄色像素参与相关性计算，背景完全排除。
            kwargs.setdefault("frame_processor", self._ship_icon_yellow_pixels)  # 候选区域也仅保留黄色，避免灰色轮廓因形状相同而命中。
        if feature_name == "Ship-Icon" and kwargs.get("box") is None:  # 独立舰船图标只在左侧队伍列表搜索，调用方显式指定范围时沿用其范围。
            frame = kwargs.get("frame")  # 显式传入截图时按该截图的实际尺寸计算搜索范围。
            if frame is None:  # 普通调用使用任务当前缓存的截图。
                frame = self.frame  # 不额外刷新截图，保持整轮场景识别使用同一帧。
            height, width = frame.shape[:2]  # 直接使用截图尺寸，避免窗口比例修正引入坐标偏移。
            kwargs["box"] = Box(0, round(height * 0.10), round(width * 0.10), round(height * 0.40), name="Ship-Icon-Search")  # 搜索左侧百分之十、垂直百分之十至五十的队伍列表区域。
        return super().find_one(feature_name, horizontal_variance=horizontal_variance, vertical_variance=vertical_variance, threshold=threshold, **kwargs)  # 其余元素仍走框架默认的局部模板匹配。

    @staticmethod  # 固定颜色规则供模板掩码和当前截图共用。
    def _ship_icon_yellow_mask(image):  # 按 HSV 色相、饱和度和亮度提取黄色轮廓。
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 将 BGR 图像转换为便于隔离黄色的颜色空间。
        return cv2.inRange(hsv, (15, 60, 100), (40, 255, 255))  # 保留缩放后饱和度较低的黄色边缘，排除无彩色背景。

    @classmethod  # 复用与模板一致的黄色识别范围。
    def _ship_icon_yellow_pixels(cls, image):  # 仅在已裁剪的左侧列表区域做颜色过滤。
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 实时画面的边缘颜色允许受缩放和动态背景轻微影响。
        mask = cv2.inRange(hsv, (10, 60, 40), (45, 255, 255))  # 候选侧放宽色相和亮度，但仍排除灰色舰船轮廓。
        return cv2.bitwise_and(image, image, mask=mask)  # 非黄色位置归零，模板背景仍由独立掩码完全排除。

    def run(self):  # 按工作流准备舰船并循环完成指定数量的战斗。
        if not self.ensure_in_front():  # 启动任务后先确认游戏窗口已经获得稳定焦点。
            self.log_error("无法将游戏窗口切换到前台，任务停止。")  # 激活失败时明确提示，不继续发送输入。
            return  # 未获得焦点时不开始识别及准备流程。
        target_count = int(self.config.get("Battle Count", 1))  # 读取本次任务需要完成的战斗场数。
        completed_count = 0  # 初始化本次任务已完成的战斗计数器。
        self.log_info(f"准备执行 {target_count} 场 {self.config.get('Battle Mode', 'PVE-Battle')} 战斗。")  # 记录本次选择的模式和目标场数。
        self.log_info("等待游戏启动并进入主界面，超时为 5 分钟。")  # 明确启动加载的等待额度。
        if not self._return_to_main(time_out=self.STARTUP_TIMEOUT):  # 启动加载和登录奖励流程共用五分钟额度。
            self.log_error("无法回到游戏主界面，任务停止。")  # 记录无法开始准备流程的原因。
            return  # 无法确认主界面时安全结束任务。
        if not self._prepare_and_join_first_battle():  # 为第一场战斗选择模式、处理加成和旗子并加入队列。
            self.log_error("第一场战斗准备失败，任务停止。")  # 记录准备流程失败。
            return  # 准备失败时安全结束任务。
        while completed_count < target_count:  # 持续运行战斗循环直到达到用户设置的场数。
            can_continue = completed_count + 1 < target_count  # 当前这一场完成后是否还需要继续战斗。
            outcome = self._run_until_result(can_continue)  # 处理排队、开战、地图导航和战斗直到本场结束。
            if not outcome:  # 检查本场是否正常结束。
                self.log_error("战斗状态机无法继续，任务停止，请查看前面的状态及超时原因。")  # 不再将所有失败误报为结算超时。
                return  # 无法完成本场时安全结束任务。
            if outcome is True:  # 普通结算也通过同一个结束状态机处理页面切换。
                outcome = self._transition_battle_end(can_continue)  # 点击后确认离开结束页，期间允许弹窗和确认页切换。
                if not outcome:  # 未确认页面切换时不能计数或执行收尾。
                    return
            completed_count += 1  # 本场结束后把当前战斗计入已完成数量。
            self.log_info(f"已完成 {completed_count}/{target_count} 场战斗。")  # 更新两种模式通用的战斗进度。
            if completed_count >= target_count:  # 检查是否已经达到用户设定的战斗场数。
                self._finish_successfully()  # 通知任务完成，并按用户开关决定是否关闭游戏。
                return  # 已确认港口后结束任务。

    def _flow_failed(self, phase, progress, reason):  # 所有状态超时都记录具体页面及输入次数并保存现场。
        self.log_error(f"{phase}失败：状态={progress.scene}，原因={reason}，输入记录={progress.actions}。")
        self._save_failure_screenshot(phase)
        return False

    def _current_battle_mode(self):  # 只识别港口顶部模式标题，避开说明气泡及模式选择页的同名按钮。
        height, width = self.frame.shape[:2]
        region = Box(round(width * .52), 0, round(width * .15), round(height * .065))
        return next((mode for mode in self.BATTLE_MODES if find_text(self, mode, self.frame, self.threshold, region) is not None), None)

    def _select_battle_mode(self):  # 港口、模式选择和覆盖弹窗共用一个有限状态循环。
        wanted = self.config.get("Battle Mode", "PVE-Battle")
        progress = SceneProgress()
        if wanted not in self.BATTLE_MODES or self.get_feature_by_name(wanted) is None:
            return self._flow_failed("选择战斗模式", progress, f"不支持或缺少模板：{wanted}")
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            scene = self._detect_scene()
            now = time.monotonic()
            if progress.observe(scene, now):
                self.log_info(f"选择战斗模式状态：{scene}，目标={wanted}。")
            if now >= deadline:
                break
            if scene == "main" and self._current_battle_mode() == wanted:
                return True  # 已确认目标模式，不再重复点击模式入口。
            if scene == "match_stopped":
                return self._flow_failed("选择战斗模式", progress, "服务器重启，匹配已停止")
            feature = "Select-Battle-Mode" if scene == "main" else wanted if scene == "battle_mode" else None
            if feature is not None:
                if progress.exhausted(feature, now):
                    return self._flow_failed("选择战斗模式", progress, f"{feature} 点击三次后未完成切换")
                if progress.action_due(feature, now):
                    button = self.find_one(feature, threshold=self.threshold)
                    if button is not None:
                        self.click(button, down_time=0.1, after_sleep=0)
                        progress.record_action(feature, time.monotonic())
            elif scene in ("popup", "menu"):
                if progress.exhausted(scene, now):
                    return self._flow_failed("选择战斗模式", progress, "覆盖页无法关闭")
                if progress.action_due(scene, now):
                    self.send_key("esc", after_sleep=0)
                    progress.record_action(scene, time.monotonic())
            elif scene in self.SCREEN_BUTTONS:
                if not self._handle_screen_button(scene, deadline=deadline):
                    return False
            self.sleep(min(.5, max(0, deadline - time.monotonic())))
        return self._flow_failed("选择战斗模式", progress, "60 秒内未确认港口目标模式")

    def _loadout_click_target(self, feature, position):  # 准备入口的标注框包住“图标加悬停展开面板”。
        if feature not in ("Addon-Selector", "Equipment"):
            return position  # 其余入口直接使用识别框中心。
        return Box(position.x + round(position.width * .10), position.y + round(position.height * .20), 1, 1, name=feature)  # 取框内固定比例，始终命中按钮图标本身。

    def _click_loadout_entry(self, feature, position=None):  # 点击准备入口或页面按钮。
        button = position if position is not None else self.find_one(feature, threshold=self.threshold)
        if button is None:
            return False  # 未识别到目标时交给下一轮重新检查。
        self.click(self._loadout_click_target(feature, button), down_time=0.1, after_sleep=0)  # 使用稳定锚点，避免点击落在悬停面板上。
        return True

    def _prepare_loadout_page(self, entry, target_scene, remove, empty):  # 装备准备按进入、卸载、返回三个阶段推进。
        phase, opened = "enter", False  # opened 表示曾经确认打开过目标页，用来区分"从未进入"和"中途掉出"。
        reopens = 0  # 页面中途退出后允许重新打开入口的轮数，避免在子页之间无限 ESC。
        progress = SceneProgress()
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            scene = self._detect_scene()
            now = time.monotonic()
            if progress.observe(f"{phase}:{scene}", now):
                self.log_info(f"准备 {target_scene} 状态：{phase}:{scene}。")
            if now >= deadline:
                break
            if scene == "match_stopped":
                return self._flow_failed(f"准备 {target_scene}", progress, "服务器重启，匹配已停止")
            if scene == target_scene:
                opened = True  # 已确认打开目标页，之后掉出页面才允许重开入口。
                if phase != "return":
                    phase = "remove"  # 只有真正打开过页面才进入卸载阶段，重开后仍停留在卸载阶段。
                    position = self.find_one(remove, threshold=self.threshold)
                    if position is not None:
                        self._click_once(progress, remove, position, now)  # 点击卸载按钮，成功后下一帧应离开加成页。
                    elif self.find_one(empty, threshold=self.threshold) is not None:
                        phase = "return"  # 已无装备可卸载，转入回港确认。
                    else:
                        self._send_loadout_escape(progress, f"{phase}:{scene}", now)  # 两个按钮都不在时页面不可用，先回港重试。
                if phase == "return":
                    self._send_loadout_escape(progress, f"return:{scene}", now)  # 页面已空或卸载完成时必须按 ESC 离开，否则会空等 60 秒。
            elif scene == "main" and phase == "return":
                return True  # 卸载完成并确认回到港口，准备流程结束。
            elif scene in ("popup", "menu"):
                self._send_loadout_escape(progress, f"{phase}:{scene}", now)  # 覆盖弹窗先关闭再重新识别。
            elif scene in self.SCREEN_BUTTONS:
                if not self._handle_screen_button(scene, deadline=deadline):
                    return False
            elif phase == "return":
                self._send_loadout_escape(progress, f"return:{scene}", now)  # 回港过程中只关闭残留外壳，不再点击入口。
            elif opened and scene in self.SUB_PAGES:
                if reopens < self.LOADOUT_REOPEN_LIMIT:  # 已打开过的页面被转场或误判打断，按确认的策略重开入口。
                    reopens += 1
                    phase = "enter"
                    self.log_info(f"加成页中途退出，第 {reopens} 次重新打开 {entry}。")
                else:
                    self._send_loadout_escape(progress, f"{phase}:{scene}", now)  # 重开额度用完后只退出，避免无限循环。
            elif phase == "enter":
                self._click_once(progress, entry, None, now)  # 尚未打开目标页时，入口点击可能在转场中丢失，冷却后重试。
            elif scene == "main":
                return True  # 卸载后回到港口即视为完成，不再点击入口。
            self.sleep(min(.5, max(0, deadline - time.monotonic())))
        reason = f"{entry} 在 {progress.count(entry)} 次点击后仍未进入 {target_scene}，请抬升悬停态模板" if not opened else "60 秒内未完成准备并回港"
        return self._flow_failed(f"准备 {target_scene}", progress, reason)

    def _send_loadout_escape(self, progress, action, now):  # 冷却结束后发送一次 ESC，避免连续按键连关两层。
        if progress.action_due(action, now, interval=self.LOADOUT_ESCAPE_INTERVAL):
            self.send_key("esc", after_sleep=0)
            progress.record_action(action, time.monotonic())

    def _click_once(self, progress, feature, position, now):  # 冷却结束后点击一次入口或按钮。
        if progress.action_due(feature, now, interval=self.LOADOUT_CLICK_INTERVAL) and self._click_loadout_entry(feature, position):
            progress.record_action(feature, time.monotonic())  # 只有实际发出点击才计数，避免冷却窗口内重复累计。

    def _transition_battle_end(self, can_continue, open_leave=False, deadline=None):  # 结束状态机允许击沉确认、结算、弹窗和港口之间任意切换。
        progress = SceneProgress()
        deadline = min(deadline, time.monotonic() + 60) if deadline is not None else time.monotonic() + 60  # 启动恢复时不突破启动流程的总额度。
        pending = None  # 只有识别到点击后的目标状态才提交本场结果。
        seen_end = open_leave
        while time.monotonic() < deadline:
            scene = self._detect_scene()
            now = time.monotonic()
            if progress.observe(scene, now):
                self.log_info(f"战斗结束状态：{scene}，待确认动作={pending}。")
            if now >= deadline:
                break
            if scene == "main" and (pending is not None or seen_end):
                return "left"  # 回港后外层只计数一次，下一场从港口重新加入。
            if pending == "continued" and scene in ("queue", "battle_start", "battle", "map"):
                return "continued"  # 已确认进入下一场，不把残留结算帧计作另一场。
            feature = None
            if scene == "result":
                seen_end = True
                feature = "Continue-Battle" if can_continue else "Back-To-Port"
                button = self.find_one(feature, threshold=self.threshold)
                if button is None and can_continue:
                    feature = "Back-To-Port"  # 续战不可用时回港，再由下一场状态机加入。
            elif scene == "leave_confirm":
                seen_end = True
                feature = "Continue-Battle-After-Sunk" if can_continue else "Leave-Battle-Confirm"
                if self.find_one(feature, threshold=self.threshold) is None and can_continue:
                    feature = "Leave-Battle-Confirm"
            elif scene in ("leave_battle", "popup", "menu"):
                action = f"esc:{scene}"
                if progress.exhausted(action, now):
                    return self._flow_failed("战斗结束", progress, "按 ESC 三次后页面未切换")
                if progress.action_due(action, now):
                    self.send_key("esc", after_sleep=0)
                    progress.record_action(action, time.monotonic())
            elif scene == "match_stopped":
                return self._flow_failed("战斗结束", progress, "服务器重启，匹配已停止")
            elif scene in self.SCREEN_BUTTONS:
                if not self._handle_screen_button(scene, deadline=deadline):
                    return False
            if feature is not None:
                if progress.exhausted(feature, now):
                    return self._flow_failed("战斗结束", progress, f"{feature} 点击三次后页面未切换")
                if progress.action_due(feature, now):
                    button = self.find_one(feature, threshold=self.threshold)  # 每次点击使用本轮新坐标。
                    if button is not None:
                        self.click(button, down_time=0.1, after_sleep=0)
                        progress.record_action(feature, time.monotonic())
                        pending = "continued" if feature.startswith("Continue-") else "left"
            self.sleep(min(.5, max(0, deadline - time.monotonic())))
        return self._flow_failed("战斗结束", progress, "等待额度内未确认续战或返回港口")

    def _finish_successfully(self):  # 达到设定场数后先回港领取集装箱，再通知完成并按配置关闭游戏。
        if not self._return_to_main():  # 正常结算和击沉离开都必须等待主界面加载完成。
            self.log_error("战斗已完成，但未能回到主界面领取集装箱，任务停止。")  # 明确区分战斗完成和收尾失败。
            return  # 未确认主界面时不点击领取入口或关闭游戏。
        if not self._collect_containers():  # 领取所有可用集装箱后才允许完成任务。
            return  # 领取流程异常时保留现场和游戏窗口。
        self.log_info("已达到设定战斗场数，任务完成。", notify=True)  # 先通知用户任务已经成功完成。
        if self.config.get("Close Game After Completion", False):  # 仅在用户主动开启开关时关闭当前绑定的游戏进程。
            self.log_info("已开启完成后关闭游戏，正在关闭游戏。")  # 记录即将执行的可见结束动作。
            self.executor.device_manager.stop_hwnd()  # 复用框架当前窗口管理器，只关闭已绑定的游戏而不退出自动化程序。

    def _collect_containers(self):  # 从主界面进入集装箱页面，循环领取直到点击领取不再出现确认按钮。
        for feature_name in ("Container-Menu", "Pick-Container", "Confirm-Container"):  # 使用用户已有标注，确认按钮的正式名称带连字符。
            if self.get_feature_by_name(feature_name) is None:  # 部分窗口比例可能尚未提供集装箱标注。
                self.log_error(f"当前比例缺少集装箱模板 {feature_name}，请补充标注，任务停止。")  # 不把缺少模板误判为没有可领取集装箱。
                return False  # 在任何领取点击前报告资源缺失。
        if not self.wait_click_feature("Container-Menu", threshold=self.threshold, time_out=30, raise_if_not_found=False, after_sleep=2):  # 点击主界面的集装箱入口并等待转场。
            self.log_error("没有找到集装箱入口，任务停止。")  # 报告未能进入领取页面。
            return False  # 入口失败时不报告领取完成。
        collected_count = 0  # 只统计确认按钮已消失的成功领取次数。
        while True:  # 按用户要求持续领取，直到领取按钮点击后没有反应。
            if not self.wait_click_feature("Pick-Container", threshold=self.threshold, time_out=15, raise_if_not_found=False, after_sleep=1):  # 每轮等待动画结束后再点击领取按钮。
                self.log_error("没有找到 Pick-Container，无法确认领取页面，任务停止。")  # 未点击到按钮属于识别异常，不当作领取耗尽。
                return False  # 保留页面供用户检查。
            if not self.wait_click_feature("Confirm-Container", threshold=self.threshold, time_out=10, raise_if_not_found=False, after_sleep=2):  # 点击领取后等待确认按钮出现并确认领取。
                self.log_info(f"点击 Pick-Container 后未出现确认按钮，已无更多集装箱可领取，本次共领取 {collected_count} 个。")  # 按用户指定的无反应条件结束循环。
                return True  # 领取耗尽属于正常完成。
            if not self.wait_until(lambda: self.find_one("Confirm-Container", threshold=self.threshold) is None, time_out=15, raise_if_not_found=False):  # 确认页面必须消失，避免同一个确认按钮反复被计数。
                self.log_error("确认领取后页面未关闭，任务停止。")  # 明确指出确认点击没有生效。
                return False  # 不在卡住的确认页面继续发送领取点击。
            collected_count += 1  # 记录本次确认领取成功。
            self.log_info(f"已领取 {collected_count} 个集装箱，继续检查可领取数量。")  # 提供可观察的领取进度。
            self.sleep(1)  # 刷新画面并响应用户停止，下一轮等待领取按钮就绪。

    def _prepare_and_join_first_battle(self):  # 在主界面完成首场战斗的全部准备动作。
        if not self.wait_click_feature("Pick-First-Ship", threshold=self.threshold, time_out=15, raise_if_not_found=False, after_sleep=1):  # 在屏幕底部 30% 范围内点击配置的舰名文字。
            return False  # 找不到舰船入口时报告准备失败。
        if not self._recall_commander_if_needed():  # 选船后先处理缺少指挥官，再进入战斗模式与装备准备。
            return False
        if not self._select_battle_mode():  # 港口已是目标模式时跳过选择，否则按识别状态执行有限重试。
            return False
        if not self._prepare_loadout_page("Addon-Selector", "addon", "Remove-All-Buff", "Install-Best-Buff"):
            return False  # 确认加成卸载完成并返回港口后再继续。
        if not self._prepare_loadout_page("Equipment", "equipment", "Remove-All-Flag", "Install-Recommended-Flag"):
            return False  # 确认旗子卸载完成并返回港口后再加入战斗。
        return self.wait_click_feature("Join-Battle", threshold=self.threshold, time_out=15, raise_if_not_found=False, after_sleep=2)  # 点击加入战斗并返回操作结果。

    def _recall_commander_if_needed(self):  # 只检查右上方当前舰船的指挥官栏，不搜索下方其他舰船卡片。
        missing = self.find_one("No-Commander", threshold=self.threshold)
        if missing is None:
            return True  # 没有识别到缺少指挥官时，沿用原来的准备流程。
        self.log_info("当前舰船没有指挥官，悬停并召回指挥官。")
        self.move(*missing.center())  # 将鼠标移到 OCR 命中的文字中心，触发指挥官操作按钮。
        self.sleep(.5)  # 等待悬停动画，并刷新下一轮识别画面。
        if not self.wait_click_feature("Recall-Commander", threshold=self.threshold, time_out=5, raise_if_not_found=False, after_sleep=1):  # 仅点击召回文字，不点击相邻的指派指挥官。
            self.log_error("当前舰船没有指挥官，悬停后未找到召回指挥官按钮，任务停止。")
            return False  # 交给统一错误退出流程保存现场。
        return True


    def _return_to_main(self, max_attempts=8, time_out=60):  # 未知画面等待加载，只对已确认可返回的页面使用 ESC。
        deadline = time.monotonic() + time_out  # 所有加载和按钮切换共用截止时间，页面变化不重置计时。
        attempts = 0  # 只统计实际发送 ESC 的次数，加载和登录不消耗返回次数。
        waiting_scene = None  # 避免同一加载阶段反复刷出等待日志。
        while time.monotonic() < deadline:  # 使用单调时钟控制总等待时长。
            scene = self._detect_scene()  # 识别当前截图对应的界面。
            remaining = deadline - time.monotonic()  # 模板匹配耗时也计入超时额度。
            if remaining <= 0:  # 识别完成时已经超时则不再发送输入。
                break  # 统一输出超时原因。
            if scene == "main":  # 主界面同时包含加入战斗和模式选择两个特有元素。
                return True  # 已回到主界面时完成返回流程。
            if scene == "match_stopped":
                progress = SceneProgress(scene=scene)
                return self._flow_failed("返回港口", progress, "服务器重启，匹配已停止")
            if scene in ("result", "leave_confirm", "leave_battle"):
                if not self._transition_battle_end(False, open_leave=True, deadline=deadline):
                    return False  # 启动时从上次残留结束页恢复，沿用总截止时间。
                continue
            if scene in self.SCREEN_BUTTONS:  # 登录和奖励页面需要点击对应按钮才能继续。
                if not self._handle_screen_button(scene, deadline=deadline):  # 按钮等待沿用本次返回流程的剩余额度。
                    if time.monotonic() >= deadline:  # 按钮切换耗尽总额度时补充整体超时原因。
                        break  # 统一输出等待主界面超时。
                    return False  # 页面没有正常切换时停止返回流程。
                waiting_scene = None  # 下一次加载阶段允许记录新的等待提示。
                continue  # 重新识别下一页，不对登录或奖励页面发送 ESC。
            if scene not in ("popup", "menu", "battle_mode", "addon", "equipment", "container"):  # 带 ESC 关闭按钮的覆盖弹窗也允许返回。
                if waiting_scene != scene:  # 仅在等待状态变化时记录原因。
                    self.log_info(f"等待游戏画面就绪：{scene}，剩余 {remaining:.0f} 秒。")  # 日志包含当前场景和剩余时间。
                    waiting_scene = scene  # 保存已输出的等待状态。
                self.sleep(min(1, remaining))  # 通过框架短暂等待，保留用户停止任务的能力。
                continue  # 刷新下一帧，不消耗返回次数。
            if attempts >= max_attempts:  # 保留已知页面连续返回失败的次数限制。
                return self._flow_failed("返回港口", SceneProgress(scene=scene), f"ESC 返回次数已达到 {max_attempts}")
            attempts += 1  # 只对本次实际返回操作计数。
            waiting_scene = None  # 返回后重新记录可能出现的加载状态。
            self.send_key("esc", after_sleep=1)  # 返回上一层；若是菜单页面则关闭菜单。
        self.log_error(f"等待游戏主界面超时（{time_out} 秒），任务停止。")  # 明确区分加载超时和返回次数用尽。
        return False  # 总额度耗尽后报告失败。

    def _handle_screen_button(self, scene, deadline=None):  # 启动时共享总截止时间，战斗中沿用原有按钮等待。
        feature = self.SCREEN_BUTTONS[scene]  # 获取当前页面的正式按钮名称。
        if scene in ("battle_survey", "inactivity_penalty"):
            self.log_info(f"识别到{'战斗满意度问卷' if scene == 'battle_survey' else '消极行为提示'}，点击关闭后重新识别页面。")
        remaining = 15 if deadline is None else deadline - time.monotonic()  # 启动流程不能重新申请独立等待额度。
        if remaining <= 0:  # 没有剩余时间时不再点击。
            return False  # 由调用方记录整体超时。
        if not self.wait_click_feature(feature, threshold=self.threshold, time_out=min(15, remaining), raise_if_not_found=False, after_sleep=min(2, remaining)):  # 点击前重新等待按钮并限制等待时间。
            self._save_failure_screenshot(feature)  # 首次查找或点击失败时也保留现场。
            return False  # 按钮消失或没有达到阈值时停止该操作。
        attempts = 1  # 首次点击已经发送，后续最多再尝试两次。
        retry_at = time.monotonic() + self.SCREEN_BUTTON_RETRY_INTERVAL  # 从首次点击后的稳定等待结束起计算重试间隔。
        exhausted = False  # 区分页面成功切换和重试耗尽导致的提前结束。
        remaining = 60 if deadline is None else deadline - time.monotonic()  # 启动后的加载使用五分钟总额度中尚未用完的时间。
        if remaining <= 0:  # 点击结束时额度耗尽则直接停止。
            self._save_failure_screenshot(feature)  # 保存最后一帧供排查总超时。
            return False  # 避免框架收到非正超时时间后继续等待。
        transition_deadline = time.monotonic() + remaining if deadline is None else deadline  # 战斗中限制为六十秒，启动时沿用同一个截止时间。

        def check_transition():  # 框架每轮先刷新画面，再检查切换或有条件重试。
            nonlocal attempts, retry_at, exhausted  # 保留本次按钮处理的次数和时间，不重置总超时。
            current_scene = self._detect_scene(refresh=False)  # 使用框架本轮的新截图，避免重复采集。
            now = time.monotonic()  # 将本轮模板匹配耗时计入总额度。
            if now >= transition_deadline:  # 截止时间之后不再发送任何重试点击。
                return False  # 由框架结束等待并报告超时。
            if current_scene not in (scene, "unknown"):  # 已到达下一个已知页面时立即交还外层流程。
                return True  # 后续领取或关闭按钮由下一轮处理。
            if current_scene == scene and now >= retry_at:  # 只有原按钮场景仍可见且冷却结束时才允许重试。
                if attempts >= self.SCREEN_BUTTON_MAX_ATTEMPTS:  # 最后一次点击也已经等待过完整间隔。
                    exhausted = True  # 提前停止无效点击，避免空耗五分钟。
                    return True  # 唤醒框架等待，由下方按失败处理。
                button = self.find_one(feature, threshold=self.threshold)  # 从本轮画面重新获取按钮坐标，不点击旧位置。
                if button is not None and time.monotonic() < transition_deadline:  # 按钮消失或识别耗尽时间时禁止发送输入。
                    attempts += 1  # 本次点击计入最多三次的限制。
                    self.log_info(f"{feature} 仍可见，重试点击（{attempts}/{self.SCREEN_BUTTON_MAX_ATTEMPTS}）。")  # 让日志明确显示正在重试。
                    self.click(button, down_time=0.1, after_sleep=0)  # 延长按下时间，并由轮询确认结果而非固定等待。
                    retry_at = time.monotonic() + self.SCREEN_BUTTON_RETRY_INTERVAL  # 每次实际点击后重新计时。
            return False  # 加载或按钮消失时只继续观察，不盲目重试。

        def pause_poll():  # 为失败的一轮识别留出短暂间隔，保持响应停止并避免忙轮询。
            self.sleep(max(0, min(0.5, transition_deadline - time.monotonic())))  # 等待不超过剩余总额度。

        changed = self.wait_until(check_transition, time_out=remaining, post_action=pause_poll, raise_if_not_found=False)  # 持续采集新画面直到成功、重试耗尽或总超时。
        if exhausted:  # 三次点击后按钮仍然存在时明确报告输入未生效。
            self.log_error(f"{feature} 点击 {attempts} 次后仍未消失，任务停止。")  # 与加载超时区分，便于检查输入方式。
        elif not changed:  # 页面迟迟没有变化时报告原因。
            self.log_error(f"点击 {feature} 后未能进入下一页面。")  # 记录哪个按钮之后发生超时。
        if exhausted or not changed:  # 失败时把现场保存到不会随调试启动清空的目录。
            self._save_failure_screenshot(feature)  # 日志会记录实际保存路径。
        return bool(changed) and not exhausted  # 只有切换到下一个已识别页面才允许继续流程。

    def _save_failure_screenshot(self, feature):  # 按日期持久保存最后识别到的失败画面。
        return save_failure_screenshot(self, feature, frame=self.executor.nullable_frame(), directory=self.FAILURE_DIRECTORY, label=feature)  # 使用已确认的错误帧，避免暂停后读取 frame 导致等待。

    def _run_until_result(self, can_continue=True):  # 从排队开始持续处理状态直到本场战斗结束。
        progress = SceneProgress()  # 明确保存本场状态和重试记录，下一场重新创建。
        battle_initialized = False  # 标记当前战斗是否已经完成前进和地图导航初始化。
        navigation_ready_at = None  # 本场首次确认战斗后记录截止时间，识别失败或导航重试不重新计时。
        battle_action_index = 0  # 从鼠标左键开始记录本场战斗下一项循环输入的位置。
        unknown_since = None  # 记录连续无法识别界面的起始时间。
        dataset_directory = None  # 每场战斗使用独立目录，排队和加载阶段不创建采集会话。
        next_dataset_capture = None  # 首次确认进入战斗后才开始一分钟计时。
        while True:  # 持续轮询战斗状态直到结算或超时。
            iteration_started = time.monotonic()  # 将识别耗时计入一秒输入周期，避免额外固定等待拉长鼠标移动间隔。
            scene = self._detect_scene()  # 使用当前最新截图判断所在界面。
            now = time.monotonic()
            if progress.observe(scene, now):
                self.log_info(f"战斗状态：{scene}。")  # 只在状态变化时记录，避免每帧刷屏。
            timeout = 600 if scene == "queue" else 60  # 排队允许十分钟，其他非战斗页面最多一分钟。
            if scene not in ("battle", "result", "leave_battle", "leave_confirm") and now - progress.entered_at >= timeout:
                return self._flow_failed("战斗流程", progress, f"状态持续 {timeout} 秒未推进")
            if scene == "match_stopped":
                return self._flow_failed("战斗流程", progress, "服务器重启，匹配已停止；本次未计入完成场次")
            if not self.config.get("Capture Battle Dataset", False):  # 开关关闭时不启动采集，旧配置缺少此字段时也默认关闭。
                dataset_directory = None  # 清除会话，重新开启后从当前战斗重新计时。
                next_dataset_capture = None  # 禁止关闭期间触发或积压采集。
            elif scene in ("battle", "map") and dataset_directory is None:  # 仅在开关开启且确认战斗后启动采集。
                dataset_directory = self.DATASET_DIRECTORY / f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"  # 用入场时间和随机后缀区分不同战斗与任务重启。
                next_dataset_capture = time.monotonic() + self.DATASET_INTERVAL  # 第一组截图在确认入场满一分钟后采集。
            if scene != "unknown":  # 成功识别任一已知界面时清除未知计时。
                unknown_since = None  # 重置连续未知界面计时器。
            if scene in self.SCREEN_BUTTONS:  # 处理运行途中出现的登录或奖励页面。
                if not self._handle_screen_button(scene):  # 复用启动时相同的按钮及页面切换逻辑。
                    return False  # 超时后结束本场处理，不向弹窗发送战斗输入。
                continue  # 弹窗处理完毕后重新识别画面。
            if scene == "result":  # 结算页包含继续战斗或返回港口按钮。
                return True  # 把结算页交回外层进行计数和续战判断。
            if scene in ("popup", "menu"):  # 覆盖弹窗和 ESC 菜单优先关闭，再识别底层页面。
                if progress.exhausted(scene, now):
                    return self._flow_failed("战斗流程", progress, "覆盖页按 ESC 三次后仍未关闭")
                if progress.action_due(scene, now):
                    self.send_key("esc", after_sleep=1)  # 再按一次 ESC 关闭最外层菜单。
                    progress.record_action(scene, time.monotonic())
                else:
                    self.sleep(.5)
                continue  # 关闭菜单后重新截图识别。
            if scene == "queue":  # 排队页面只需要等待系统匹配战斗。
                self.sleep(2)  # 等待两秒后再检查排队状态。
                continue  # 排队期间不执行任何游戏操作。
            if scene in ("leave_battle", "leave_confirm"):  # 从击沉或已打开的确认页都能进入结束状态机。
                outcome = self._handle_leave_battle(can_continue)  # 按 ESC 后根据场次选择继续战斗或确认离开。
                if outcome == "continued":  # 后续页面已经点击继续战斗。
                    return "continued"  # 本场计入完成且下一场已经开始排队。
                if outcome == "left":  # 离开本场也应计数；下一场重新建立导航和采集会话。
                    return "left"
                return False  # 后续页面处理失败时停止本场循环。
            if scene == "main":  # 加入战斗后仍停在主界面时视为按钮未成功生效。
                if progress.exhausted("join", now):
                    return self._flow_failed("战斗流程", progress, "加入战斗三次后仍停留在港口")
                if progress.action_due("join", now):
                    button = self.find_one("Join-Battle", threshold=self.threshold)
                    if button is not None:
                        self.click(button, down_time=0.1, after_sleep=0)
                        progress.record_action("join", time.monotonic())
                self.sleep(.5)
                continue  # 点击后重新截图识别游戏状态。
            if scene in ("battle_mode", "addon", "equipment"):  # 意外停留在准备子页面时按通用返回规则处理。
                self.send_key("esc", after_sleep=1)  # 按 ESC 返回上一层页面。
                continue  # 返回后重新截图识别游戏状态。
            if scene == "battle_start":  # 等待战斗开始页面出现开始按钮。
                if progress.exhausted("start", now):
                    return self._flow_failed("战斗流程", progress, "开始战斗三次后仍未入场")
                if progress.action_due("start", now):
                    button = self.find_one("Start-Battle", threshold=self.threshold)
                    if button is not None:
                        self.click(button, down_time=0.1, after_sleep=0)
                        progress.record_action("start", time.monotonic())
                self.sleep(.5)
                continue  # 点击后重新截图识别游戏状态。
            if scene == "battle" and not battle_initialized:  # 首次确认舰船铭牌和独立舰船图标后执行航行初始化。
                if navigation_ready_at is None:
                    navigation_ready_at = time.monotonic() + self.NAVIGATION_START_DELAY
                    self.log_info(f"已进入战斗界面，本场等待 {self.NAVIGATION_START_DELAY} 秒让开局提示消失后开始导航。")
                remaining = navigation_ready_at - time.monotonic()  # 只等待本场尚未经过的时间。
                if remaining > 0:
                    self.sleep(remaining)  # 保留框架的暂停和用户停止能力。
                if progress.exhausted("navigation", time.monotonic(), interval=0):
                    return self._flow_failed("战斗导航", progress, "导航初始化三次后仍未成功")
                battle_initialized = self._initialize_battle_navigation()  # 每次重试仍重新确认画面，只有导航成功才标记完成。
                if battle_initialized is False:  # 只有实际导航失败才计数，截图漏识别时尚未执行导航。
                    progress.record_action("navigation", time.monotonic())
                continue  # 回到战斗界面后重新识别状态。
            if scene == "map":  # 处理可能已经打开但尚未完成选择的地图页面。
                self._close_map()  # 首次导航前先回到战斗画面等待；已完成导航时只关闭地图，不重新选点。
                continue  # 返回战斗界面后重新识别状态。
            if scene == "battle":  # 仅在确认战斗界面时发送战斗输入。
                if next_dataset_capture is not None:  # 只有开关开启并启动计时后才执行采集相关逻辑。
                    capture_time = time.monotonic()  # 使用单调时间判断本场截图是否到期。
                    if capture_time >= next_dataset_capture:  # 满一分钟后先采集，避免向大地图发送战斗点击。
                        next_dataset_capture += (int((capture_time - next_dataset_capture) // self.DATASET_INTERVAL) + 1) * self.DATASET_INTERVAL  # 保持分钟节奏，暂停或加载后不补拍积压的时间点。
                        self._capture_battle_dataset(dataset_directory)  # 保存战斗和大地图的成对截图，并恢复战斗视图。
                        continue  # 采集后重新识别场景，避免在已经结算或击沉时继续开火。
                battle_action_index = self._send_battle_action(battle_action_index)  # 按鼠标左键、R、T、F 的顺序发送当前输入并推进轮换位置。
                self.sleep(max(0, 1 - (time.monotonic() - iteration_started)))  # 补足一秒周期；识别较慢时不追赶或额外等待。
                continue  # 继续检查战斗是否结束。
            if unknown_since is None:  # 第一次进入无法识别的过渡画面时开始计时。
                unknown_since = time.monotonic()  # 保存单调时钟时间以避免系统时间变化影响。
            elif time.monotonic() - unknown_since >= 60:  # 连续一分钟无法识别任何界面时判定异常。
                return self._flow_failed("战斗流程", progress, "连续 60 秒无法识别页面")
            self.sleep(1)  # 对加载画面和短暂动画留出一秒缓冲。


    @preserve_input_timing
    def _capture_battle_dataset(self, directory):  # 在同一任务线程内采集战斗和大地图，避免与战斗输入并发冲突。
        if not self.config.get("Capture Battle Dataset", False):  # 调用前再次检查开关，关闭时不额外截图或发送地图按键。
            return False  # 即使单独调用采集方法也必须尊重用户开关。
        if self._detect_scene() != "battle":  # 保存前刷新并确认仍在正常战斗，排除击沉、结算及覆盖页面。
            return False  # 当前画面已变化时跳过本组，交给外层状态机继续处理。
        battle_frame = self.frame.copy()  # 保留已确认场景的原始全尺寸截图，不画标注框也不缩放。
        sample_name = f"{datetime.now():%Y%m%d_%H%M%S_%f}_{uuid4().hex[:8]}"  # 同组共用时间戳及随机后缀，系统时钟回拨也不覆盖旧图。
        self.send_key("m", after_sleep=1)  # 仅打开大地图查看，不调用选点逻辑以免改变航线。
        self.wait_until(lambda: self._detect_scene(refresh=False) not in ("battle", "unknown"), time_out=5, raise_if_not_found=False)  # 等待地图出现，也允许结算或击沉立即中止等待。
        if self._detect_scene(refresh=False) != "map":  # 只保存确认的大地图，打开失败时不将其他界面误标为地图。
            self.log_warning("数据集采集：未进入大地图或战斗已结束，跳过本组截图。")  # 记录缺失原因供后续检查。
            return False  # 不盲按关闭键，以免在战斗界面重新打开地图或干扰结算。
        map_frame = self.frame.copy()  # 保存地图确认时的同一帧，保留原始分辨率和全部界面信息。
        self.send_key("m", after_sleep=1)  # 截图后使用地图切换键返回战斗，不重新设置航点。
        closed = self.wait_until(lambda: self._detect_scene(refresh=False) not in ("map", "unknown"), time_out=self.MAP_RETURN_TIMEOUT, raise_if_not_found=False)  # 确认已经离开地图，结算和击沉也视为地图流程结束。
        if not closed:  # 地图切换未成功时由外层状态机恢复，期间不发送开火输入。
            self.log_warning("数据集采集后仍未确认离开大地图，将由战斗流程继续恢复。")  # 保留地图恢复失败的诊断信息。
        try:  # 文件系统或 PNG 编码失败只影响当前采集，不中断自动战斗。
            directory.mkdir(parents=True, exist_ok=True)  # 首次成功采集时创建本场目录，后续追加保存。
            for view, frame in (("battle", battle_frame), ("map", map_frame)):  # 使用统一文件名前缀区分同组的两个视图。
                encoded, data = cv2.imencode(".png", frame)  # 无损保存截图以保留小地图图标和文字细节。
                if not encoded:  # 编码器没有返回有效图像时不能报告采集成功。
                    raise OSError(f"无法编码 {view} 数据集截图")  # 交给统一保存失败分支输出日志。
                data.tofile(str(directory / f"{sample_name}_{view}.png"))  # 通过数组写文件支持 Windows 中文目录。
        except (OSError, cv2.error) as error:  # 只捕获保存相关错误，不吞掉用户停止任务的异常。
            self.log_warning(f"数据集截图保存失败：{error}")  # 明确指出磁盘或编码问题。
            return False  # 保持战斗继续，下一分钟再尝试采集。
        self.log_info(f"已保存战斗与大地图截图：{directory / sample_name}")  # 报告本组文件的共同前缀以便定位。
        return True  # 两张 PNG 都成功写入后才报告成功。

    @preserve_input_timing
    def _send_battle_action(self, action_index):  # 发送当前轮换位置对应的战斗输入并返回下一位置。
        actions = ("left_click", "r", "t", "f")  # 定义鼠标左键、R、T、F 的固定循环顺序。
        action = actions[action_index % len(actions)]  # 把任意输入索引归一化到四项循环内。
        if action == "left_click":  # 当前轮到鼠标左键时在屏幕中心点击。
            self.click_relative(0.5, 0.5, move=False, name="battle_fire")  # 在屏幕中心发送一次鼠标左键点击。
        else:  # 当前轮到 R、T 或 F 时通过任务输入接口发送按键。
            self.send_key(action)  # 发送当前小写键名对应的 R、T 或 F 键。
        self.move_relative(random.uniform(0.35, 0.65), random.uniform(0.35, 0.60))  # 每轮战斗输入后把鼠标随机移到中央，避开左侧舰船图标和底部 HUD。
        return (action_index + 1) % len(actions)  # 推进并循环下一次输入的位置。

    def _handle_leave_battle(self, can_continue):  # 击沉后交给统一结束状态机，不阻塞等待某一种确认页面。
        return self._transition_battle_end(can_continue, open_leave=True)  # 允许按 ESC 后直接进入结算页或港口。


    @preserve_input_timing
    def _initialize_battle_navigation(self):  # 在战斗开始时完成前进输入并打开地图。
        if self._detect_scene() != "battle":  # 等待期间可能出现击沉、结算或弹窗，必须刷新截图重新确认。
            self.log_info("本场开局等待已结束，当前未能确认战斗界面；重新识别后继续导航，不重复等待。")
            return None  # 表示导航尚未执行，交回状态循环，短暂漏识别不消耗导航重试。
        for _ in range(10):  # 按工作流向游戏发送十次前进键。
            self.send_key("w", after_sleep=0.05)  # 短按一次 W 键并留出极短输入间隔。
        self.send_key("m", after_sleep=2)  # 按 M 键打开地图模式并等待地图绘制。
        return self._handle_map()  # 在地图上选择航点，并将初始化结果交回场景循环。

    @preserve_input_timing
    def _handle_map(self):  # 按住 Shift 依次追加占领区或敌方基地，最后追加地图对侧。
        map_anchor = self.wait_until(self._map_is_visible, time_out=20, raise_if_not_found=False)  # 等待舰船铭牌和两个地图按钮同时出现。
        if map_anchor is None:  # 检查 M 键是否成功打开了大地图。
            self.log_warning("没有识别到大地图锚点，跳过本次地图选点。")  # 记录地图未成功打开或仍处于加载中的情况。
            return False  # 未确认大地图时不发送 ESC 以免误开战斗菜单。
        self.next_frame()  # 获取同一时刻的新截图供全部地图元素共同判断。
        map_overview = self.get_box_by_name("Map-Overview")  # 使用十九号截图标注的矩形确定主地图实际边界。
        ship_cursor = self._find_rotated_ship_cursor(map_overview)  # 仅在主地图内旋转匹配当前舰船光标以排除右下角小地图。
        target_areas = []  # 收集主地图内实际识别到且不是绿色的占领区匹配框及其颜色。
        area_score = 0  # 记录任意占领区字母的最高置信度，用于和敌方基地互斥比较。
        for area_name in ("Area-A", "Area-B", "Area-C", "Area-D"):  # 逐个识别四个字母及其绿色、红色或灰色状态。
            area_box, area_color = self._find_area(area_name, map_overview)  # 同时取得当前字母的最佳匹配框和颜色。
            if area_box is None:  # 当前字母三种颜色都没有达到地图阈值时跳过。
                continue  # 继续检查下一个占领区字母。
            area_score = max(area_score, area_box.confidence)  # 用所有字母中的最高分代表占领区图类型。
            if area_color in ("gray", "red"):  # 仅把可占领的灰色或敌方红色区域作为导航候选。
                target_areas.append((area_box, area_color))  # 保存匹配框和颜色供选点及日志共同使用。
        enemy_base = self._find_scored_map_feature("Enemy-Base", box=map_overview)  # 记录敌方基地原始分数并按地图阈值过滤。
        if enemy_base is not None and area_score > 0:  # 占领区和敌方基地不可能同时属于同一张地图，因此按更高分只保留一类。
            if area_score >= enemy_base.confidence:  # 占领区最高分不低于敌方基地时按四点图处理。
                self.log_info(f"占领区最高分 {area_score * 100:.2f}% 高于敌方基地 {enemy_base.confidence * 100:.2f}%，忽略敌方基地。")  # 记录互斥判断结果以便核对误识别。
                enemy_base = None  # 丢弃较低分的敌方基地匹配。
            else:  # 敌方基地分数更高时按两点图处理。
                self.log_info(f"敌方基地 {enemy_base.confidence * 100:.2f}% 高于占领区最高分 {area_score * 100:.2f}%，忽略占领区。")  # 记录互斥判断结果以便核对误识别。
                target_areas = []  # 丢弃较低分的占领区匹配。
        if ship_cursor is not None and target_areas:  # 舰船位置和至少一个非绿色占领区存在时按住 Shift 依次选点。
            ordered_areas = sorted(target_areas, key=lambda area: ship_cursor.center_distance(area[0]))  # 把全部灰色和红色占领区按与舰船的距离由近及远排序。
            try:  # 保证任何异常或用户停止都能执行释放 Shift 的收尾动作。
                self.send_key_down("shift", after_sleep=0.5)  # 将按下后的等待也纳入保护，停止时仍会释放 Shift。
                for index, (area_box, area_color) in enumerate(ordered_areas, start=1):  # 按照由近及远的顺序逐个点击所有灰红占领区。
                    self.log_info(f"按住 Shift 点击第 {index}/{len(ordered_areas)} 个占领区 {area_box.name}，颜色为 {area_color}，分数 {area_box.confidence * 100:.2f}%，阈值 {self.map_threshold * 100:.2f}%，中心 ({area_box.center()[0]}, {area_box.center()[1]})。")  # 记录每个目标的顺序、颜色、分数、阈值与位置。
                    self.click(area_box, after_sleep=self.MAP_POINT_SETTLE_SECONDS)  # 点击当前区域并等待航点标记及路线动画稳定。
                self._click_opposite_map_point(ship_cursor, map_overview)  # 按住 Shift 把地图对侧追加为队列最后的目标点。
            finally:  # 释放修饰键必须早于关闭地图，避免 Shift 一直处于按下状态。
                self.send_key_up("shift", after_sleep=1)  # 全部目标点击完成后松开 Shift 并留出一帧处理时间。
        elif enemy_base is not None:  # 没有可选择的占领区但识别到敌方基地时先点击基地。
            self.click(enemy_base, after_sleep=self.MAP_POINT_SETTLE_SECONDS)  # 点击敌方基地并等待航点标记及路线动画稳定。
            if ship_cursor is not None:  # 只有识别到舰船位置时才能计算对侧点并追加到基地之后。
                try:  # 保证任何异常或用户停止都能执行释放 Shift 的收尾动作。
                    self.send_key_down("shift", after_sleep=0.5)  # 将按下后的等待也纳入保护，停止时仍会释放 Shift。
                    self._click_opposite_map_point(ship_cursor, map_overview)  # 把地图对侧追加为队列最后的目标点。
                finally:  # 释放修饰键必须早于关闭地图，避免 Shift 一直处于按下状态。
                    self.send_key_up("shift", after_sleep=1)  # 追加完成后松开 Shift 并留出一帧处理时间。
        elif ship_cursor is not None:  # 没有区域或基地但能定位舰船时仍导航到地图对侧。
            self._click_opposite_map_point(ship_cursor, map_overview)  # 未识别到任何目标时把对侧点作为唯一航点。
        else:  # 没有目标且舰船光标也缺失时无法设置地图对侧的备用航点。
            self.log_warning("没有识别到舰船光标、占领区或敌方基地，跳过本次地图选点。")  # 避免在无法确定当前位置时误点地图。
        self._close_map()  # 关闭大地图并在 ESC 未生效时使用地图切换键兜底。
        return True  # 报告本次大地图已经成功识别并关闭。

    def _click_opposite_map_point(self, ship_cursor, map_overview):  # 计算舰船位置的地图中心对称点并点击。
        cursor_x, cursor_y = ship_cursor.center()  # 读取当前舰船在主地图中的中心坐标。
        opposite_x = map_overview.x + map_overview.width - (cursor_x - map_overview.x)  # 以主地图中心为轴把舰船水平位置映射到另一侧。
        opposite_y = map_overview.y + map_overview.height - (cursor_y - map_overview.y)  # 以主地图中心为轴把舰船垂直位置映射到另一侧。
        self.log_info(f"点击地图对侧 ({opposite_x}, {opposite_y}) 作为最后的目标点。")  # 记录对侧航点坐标及追加时机。
        self.click(opposite_x, opposite_y, name="opposite-map-side", after_sleep=self.MAP_POINT_SETTLE_SECONDS)  # 点击对侧点并等待航点标记及路线动画稳定。

    def _find_scored_map_feature(self, feature_name, color=None, **kwargs):  # 获取原始匹配分数并在记录后应用地图阈值。
        match = self.find_one(feature_name, threshold=-1.0, **kwargs)  # 取得最佳候选以便低于阈值时也能诊断误匹配。
        label = f"{feature_name} ({color})" if color else feature_name  # 将颜色变体写入日志标签。
        if match is None:  # 模板引擎没有生成候选时不能提供分数。
            self.log_info(f"[地图匹配] {label}: 无分数，阈值 {self.map_threshold * 100:.2f}%。")  # 明确区分无候选和低分候选。
            return None  # 没有候选时不参与导航。
        accepted = match.confidence >= self.map_threshold  # 保持原有地图匹配阈值不变。
        status = "命中" if accepted else "未命中"  # 标记候选是否达到有效阈值。
        self.log_info(f"[地图匹配][{status}] {label}: 分数 {match.confidence * 100:.2f}%，阈值 {self.map_threshold * 100:.2f}%，中心 ({match.center()[0]}, {match.center()[1]})。")  # 保存每次比较的原始分数和位置。
        return match if accepted else None  # 只允许达到阈值的候选影响识别和选点。

    def _find_area(self, area_name, search_box=None):  # 在指定范围内识别占领区字母并同时返回它当前显示的颜色。
        area_feature = self.get_feature_by_name(area_name)  # 取得该字母正式标注生成的原始模板。
        if area_feature is None:  # 检查字母模板资源是否成功加载。
            return None, None  # 缺少模板时明确返回未识别到匹配框和颜色。
        if search_box is None:  # 独立调用没有传入范围时仍限制在正式主地图区域内搜索。
            search_box = self.get_box_by_name("Map-Overview")  # 从正式标注读取可随截图分辨率缩放的主地图边界。
        color_matches = []  # 收集三种颜色中达到正式阈值的匹配结果。
        for color_name, (hue, saturation) in self.AREA_COLOR_HSV.items():  # 分别生成绿色、红色和灰色模板进行完整比较。
            color_template = self._colorize_area_template(area_feature.mat, hue, saturation)  # 保留字母明暗结构并替换成当前候选颜色。
            color_box = self._find_scored_map_feature(area_name, color=color_name, box=search_box, template=color_template)  # 记录各颜色原始分数并应用地图阈值。
            if color_box is not None:  # 仅收集达到正式模板阈值的候选颜色。
                color_matches.append((color_box, color_name))  # 保存候选框和对应颜色供最终按置信度排序。
        return max(color_matches, key=lambda item: item[0].confidence, default=(None, None))  # 返回三种颜色中的最高分匹配框和颜色。

    @staticmethod  # 颜色转换只依赖原始模板与目标 HSV 参数。
    def _colorize_area_template(template, hue, saturation):  # 把占领区字母模板转换成指定颜色并保留原始亮度细节。
        template_hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)  # 将 BGR 模板转换为便于独立修改颜色的 HSV 空间。
        color_hsv = template_hsv.copy()  # 复制模板以避免修改 FeatureSet 缓存中的正式原图。
        color_hsv[:, :, 0] = hue  # 将模板全部像素统一设置为目标色相。
        color_hsv[:, :, 1] = saturation  # 将模板全部像素统一设置为目标饱和度。
        return cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)  # 转回框架模板匹配需要的 BGR 图像。

    def _find_rotated_ship_cursor(self, search_box=None):  # 在指定主地图范围内旋转模板以识别不同朝向的舰船光标。
        cursor_feature = self.get_feature_by_name("My-Ship-Cursor")  # 取得正式标注生成的原始舰船光标模板。
        if cursor_feature is None:  # 检查模板资源是否成功加载。
            return None  # 缺少模板时交给其他地图目标或安全跳过逻辑处理。
        if search_box is None:  # 独立调用没有传入范围时仍使用正式主地图标注。
            search_box = self.get_box_by_name("Map-Overview")  # 从十九号截图标注读取可随分辨率缩放的主地图边界。
        cursor_threshold = min(self.map_threshold, 0.7)  # 旋转插值会降低相关系数，因此为光标使用更宽松的专用上限阈值。
        matches = []  # 收集所有达到阈值的旋转模板匹配结果。
        for angle in range(0, 360, 45):  # 每隔四十五度匹配一次，仅检查八个主要方向以缩短导航识别耗时。
            rotated_template = self._rotate_template(cursor_feature.mat, angle)  # 生成当前朝向且不裁切边角的光标模板。
            match = self.find_one("My-Ship-Cursor", threshold=cursor_threshold, box=search_box, template=rotated_template)  # 使用原始截图分辨率在完整主地图区域匹配当前旋转方向。
            if match is not None:  # 仅保留达到旋转光标专用阈值的候选结果。
                matches.append(match)  # 保存候选结果供最终比较置信度。
        return max(matches, key=lambda box: box.confidence, default=None)  # 返回全部旋转方向中置信度最高的唯一光标。

    @staticmethod  # 该旋转操作只依赖输入模板，不读取任务状态。
    def _rotate_template(template, angle):  # 在不裁切内容的前提下按指定角度旋转模板。
        if angle == 0:  # 零度方向无需插值处理。
            return template  # 直接复用原始模板以保留全部像素细节。
        height, width = template.shape[:2]  # 读取原始模板尺寸以计算旋转边界。
        center = (width / 2, height / 2)  # 使用模板中心作为旋转中心。
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # 创建不缩放的二维旋转矩阵。
        cosine = abs(matrix[0, 0])  # 取得旋转矩阵中的余弦绝对值。
        sine = abs(matrix[0, 1])  # 取得旋转矩阵中的正弦绝对值。
        rotated_width = int(math.ceil(height * sine + width * cosine))  # 计算容纳完整旋转模板的新宽度。
        rotated_height = int(math.ceil(height * cosine + width * sine))  # 计算容纳完整旋转模板的新高度。
        matrix[0, 2] += rotated_width / 2 - center[0]  # 将旋转后的模板水平移动到新画布中心。
        matrix[1, 2] += rotated_height / 2 - center[1]  # 将旋转后的模板垂直移动到新画布中心。
        edge_pixels = np.concatenate((template[0], template[-1], template[:, 0], template[:, -1]))  # 汇总模板四条边上的地图背景像素。
        border_color = tuple(int(channel) for channel in np.median(edge_pixels, axis=0))  # 使用边缘中位色填充旋转产生的空白角落。
        return cv2.warpAffine(template, matrix, (rotated_width, rotated_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=border_color)  # 生成完整且背景连续的旋转模板。

    @preserve_input_timing
    def _close_map(self):  # 关闭大地图并等待确认已经回到动态战斗画面。
        self.send_key("esc", after_sleep=1)  # 在航点动画稳定后优先按工作流要求使用 ESC 返回战斗界面。
        map_closed = self.wait_until(self._map_is_closed, time_out=self.MAP_RETURN_TIMEOUT, raise_if_not_found=False)  # 持续刷新画面而不是固定两秒后只检查一次。
        if map_closed:  # 舰船铭牌存在且地图按钮组合已消失，确认回到战斗界面。
            return True  # 报告已回到战斗界面供调用方和测试确认。
        self.log_warning("ESC 后等待八秒仍未确认战斗界面，改用 M 键关闭地图。")  # 记录延长等待后仍需执行的恢复动作。
        self.send_key("m", after_sleep=1)  # 使用地图模式切换键兜底返回战斗界面。
        map_closed = self.wait_until(self._map_is_closed, time_out=self.MAP_RETURN_TIMEOUT, raise_if_not_found=False)  # 再等待一次并验证 M 键确实关闭了地图。
        if map_closed:  # M 键后确认舰船铭牌仍在且地图按钮组合已消失。
            return True  # 报告兜底关闭成功。
        self.log_error("M 键后等待八秒仍未回到战斗界面。")  # 两种关闭方式都失败时留下明确诊断日志。
        return False  # 报告地图关闭失败以便后续状态循环继续恢复。

    def _detect_scene(self, refresh=True):  # 根据当前截图中的特有元素判断游戏所处界面。
        if refresh:  # 普通调用需要在判断场景前刷新截图。
            self.next_frame()  # 主动获取一张最新截图供本次场景识别使用。
        if self.find_one("Menu", threshold=self.threshold) is not None:  # 菜单元素具有最高优先级以便正确关闭菜单。
            return "menu"  # 返回菜单场景。
        if self.find_one("Match-Stopped", threshold=self.threshold) is not None:
            return "match_stopped"  # 服务器维护属于明确的停止状态，不自动循环重排。
        for scene, feature in self.SCREEN_BUTTONS.items():  # 优先识别登录及奖励覆盖页，领取按钮优先于关闭按钮。
            if self.find_one(feature, threshold=self.threshold) is not None:  # 检查当前页面的按钮是否可见。
                return scene  # 仅返回状态，由执行流程决定点击；屏幕识别测试保持只读。
        if self._has_any(("Continue-Battle", "Back-To-Port")):  # 正常结算页也有 ESC 叉号，先交给计数及续战流程，避免跳过本场计数。
            return "result"  # 模糊覆盖弹窗没有可操作的结算按钮，仍由下面的关闭标记识别。
        if self.find_one("Leave-Battle-Confirm", threshold=self.threshold) is not None:
            return "leave_confirm"  # OCR 已要求离开标题上下文，避免把其他“是”误判为击沉确认页。
        if self.find_one("Popup-ESC-Close", threshold=max(self.threshold, 0.9), horizontal_variance=1, vertical_variance=1) is not None:  # 全图搜索 ESC 加关闭叉号，兼容居中问卷和右上角活动弹窗，不匹配普通 ESC 提示。
            for scene, features in (("addon", ("Remove-All-Buff", "Install-Best-Buff")), ("equipment", ("Remove-All-Flag", "Install-Recommended-Flag")), ("battle_mode", self.BATTLE_MODES)):  # 加成页和旗子页必须先于模式页判断：港口左下角的模式标签在子页仍然可见。
                if self._has_any(features):  # 已识别的准备页面由原流程管理，避免遮蔽状态。
                    return scene  # 启动返回和战斗恢复流程均会按 ESC 退出这些子页面。
            return "popup"  # 未知覆盖页交给执行流程按 ESC 后重新识别页面。
        if self.find_one("Control-Camera", threshold=self.threshold) is not None:  # 自由视角图标表示本舰已经被击沉。
            return "leave_battle"  # 优先停止战斗输入，并复用 ESC 打开离开战斗页面的流程。
        if self.find_one("Leave-Battlefield", threshold=self.threshold) is not None:  # 击沉页面使用底部的离开战斗入口判断。
            return "leave_battle"  # 返回需要离开当前战斗的场景。
        if self.find_one("In-Battle-Queue", threshold=self.threshold) is not None:  # 排队页使用专用状态元素判断。
            return "queue"  # 返回战斗排队场景。
        if self.find_one("Start-Battle", threshold=self.threshold) is not None:  # 等待开战页使用开始按钮判断。
            return "battle_start"  # 返回等待战斗开始场景。
        battle_view = self._detect_battle_view()  # 由铭牌确认进入战斗，再用两个地图按钮区分视图。
        if battle_view is not None:  # 只有舰船铭牌存在时组合判断才属于战斗生命周期页面。
            return battle_view  # 返回大地图或普通战斗场景。
        if self._has_any(("Remove-All-Buff", "Install-Best-Buff")):  # 加成页必须先于模式页判断：港口顶部的模式文字在加成页仍然可见。
            return "addon"  # 返回加成选择场景。
        if self._has_any(("Remove-All-Flag", "Install-Recommended-Flag")):  # 旗子页同样保留港口顶部模式文字。
            return "equipment"  # 返回旗子装备场景。
        if self._has_any(self.BATTLE_MODES):  # 任一支持的模式按钮均可确认模式选择页面。
            return "battle_mode"  # 返回战斗模式选择场景。
        if self.find_one("Join-Battle", threshold=self.threshold) is not None and self.find_one("Select-Battle-Mode", threshold=self.threshold) is not None:  # 主界面必须同时存在两个特有元素以减少误判。
            return "main"  # 返回游戏主界面场景。
        if self._has_any(("Pick-Container", "Confirm-Container")):  # 支持从上一次领取完成或手动停止的集装箱页面重新启动。
            return "container"  # 返回可通过 ESC 退出的集装箱页面。
        return "unknown"  # 没有匹配到已知特有元素时返回未知场景。

    def _has_any(self, feature_names):  # 判断当前缓存截图中是否存在任一指定元素。
        return any(self.find_one(feature_name, threshold=self.threshold) is not None for feature_name in feature_names)  # 依次匹配并在发现首个元素时返回真。

    def _detect_battle_view(self):  # 使用铭牌确认战斗状态，再用两个地图按钮共同确认大地图。
        nameplate_visible = self.find_one("Libertad-Nameplate", threshold=self.map_threshold) is not None  # 使用用户配置的舰名识别战斗和地图共有的铭牌。
        if not nameplate_visible:  # 没有舰船铭牌时不能确认已经进入战斗。
            return None  # 交给后续其他页面特征继续判断。
        if any(self.get_feature_by_name(name) is None for name in self.MAP_VIEW_FEATURES):  # 当前比例必须具备两个地图按钮模板才能可靠区分视图。
            return None  # 缺少模板时不推断已进入或关闭大地图。
        button_threshold = min(self.map_threshold, self.MAP_VIEW_THRESHOLD)  # 按钮使用缩放适配阈值，同时尊重用户配置的更低阈值。
        map_visible = all(self.find_one(name, threshold=button_threshold) is not None for name in self.MAP_VIEW_FEATURES)  # 同一帧中两个地图按钮必须同时命中。
        return "map" if map_visible else "battle"  # 两个按钮同时存在为大地图，否则为普通战斗界面。

    def _map_is_visible(self):  # 判断已进入战斗且两个地图按钮同时出现。
        return self._detect_battle_view() == "map"  # 仅组合判断结果为地图时返回真。

    def _map_is_closed(self):  # 必须重新确认战斗界面才能结束关闭地图的等待。
        return self._detect_battle_view() == "battle"  # 未知或加载画面不会被当成成功关闭地图。
