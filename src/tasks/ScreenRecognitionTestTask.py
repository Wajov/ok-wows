import json  # 导入 JSON 模块以动态读取正式模板中的全部元素名称。

from qfluentwidgets import FluentIcon  # 导入任务列表中使用的内置图标。

from src.config import config as app_config  # 导入应用正式模板配置以避免维护重复元素列表。
from src.tasks.AutoPveBattleTask import AutoPveBattleTask  # 复用自动 PVE 任务当前使用的完整场景判断规则。
from src.tasks.feature_ocr import OCR_TEXTS  # 诊断时区分文字识别分数与模板匹配分数。


class ScreenRecognitionTestTask(AutoPveBattleTask):  # 像诊断任务一样定义由用户启动并持续运行的只读测试任务。

    def __init__(self, *args, **kwargs):  # 初始化测试任务的界面元数据和模板阈值配置。
        super().__init__(*args, **kwargs)  # 首先初始化基础任务能力和共享场景判断方法。
        self.name = "Screen Recognition Test"  # 设置任务列表中显示的英文名称。
        self.support_schedule_task = False  # 持续识别测试不继承战斗任务的定时执行入口。
        self.description = "Continuously checks every template every 3 seconds and reports the detected screen without sending any input."  # 说明任务持续识别当前屏幕且不操作游戏。
        self.icon = FluentIcon.SEARCH  # 使用搜索图标表示该任务只执行检查。
        self.default_config.pop("Battle Count", None)  # 移除继承得到但测试任务不需要的战斗场数配置。
        self.config_description.pop("Battle Count", None)  # 移除测试任务不使用的战斗场数说明。
        self.default_config.pop("Battle Mode", None)  # 只读测试无需选择实际进入的战斗模式。
        self.config_description.pop("Battle Mode", None)  # 移除模式选择的帮助说明。
        self.config_type.pop("Battle Mode", None)  # 移除继承的模式下拉框元数据。
        self.default_config.pop("Close Game After Completion", None)  # 移除只属于自动战斗成功流程的关闭游戏开关。
        self.config_description.pop("Close Game After Completion", None)  # 移除测试任务不使用的关闭游戏说明。
        self.default_config.pop("Capture Battle Dataset", None)  # 只读识别任务不显示会切换地图的数据集采集开关。
        self.config_description.pop("Capture Battle Dataset", None)  # 同时移除数据集采集说明。

    def run(self):  # 参照 DiagnosisTask 在一次启动中持续运行，直到用户手动停止任务。
        if not self.ensure_in_front():  # 识别测试也先把游戏切换到前台，方便用户观察当前画面。
            self.log_error("无法将游戏窗口切换到前台，识别测试停止。")  # 启动失败时明确报告原因。
            return  # 不在未聚焦的窗口上开始识别循环。
        while True:  # 保持任务占用当前执行流程，避免一次检查结束后被一次性任务执行器禁用。
            self._inspect_once()  # 截取当前帧并完成一轮全部元素和界面判断。
            self.sleep(3)  # 使用任务睡眠等待三秒，并允许框架在用户停止任务时中断循环。

    def _inspect_once(self):  # 截取一帧并在同一帧中检查全部元素和当前界面。
        frame = self.next_frame()  # 只主动截取一次当前游戏画面。
        if frame is None:  # 检查截图接口是否成功返回画面。
            self.log_error("无法获取当前屏幕，本轮识别已跳过。")  # 记录当前轮次无法执行识别的原因。
            return  # 跳过本轮且不发送任何输入，外层循环仍会继续。
        feature_names = self._load_feature_names()  # 从正式 COCO 模板配置动态读取全部元素名称。
        matched_count = 0  # 初始化达到用户配置阈值的元素计数器。
        matched_boxes = []  # 收集超过阈值的候选框，供本轮结束时统一绘制。
        self.log_info(f"开始检查当前屏幕中的 {len(feature_names)} 个元素，阈值为 {self.threshold:.2f}。")  # 记录本次检查范围和匹配阈值。
        for feature_name in feature_names:  # 依次在同一张缓存截图中检查每个正式模板元素。
            method_label = "OCR" if feature_name in OCR_TEXTS else "模板"  # 显示当前元素实际使用的识别方法。
            if feature_name in ("Area-A", "Area-B", "Area-C", "Area-D"):  # 占领区需要同时判断字母和绿色、红色或灰色状态。
                feature_box, area_color = self._find_area(feature_name)  # 使用三种颜色模板中的最高分返回字母位置和颜色。
                if feature_box is None:  # 检查三种颜色是否都没有达到正式阈值。
                    self.log_info(f"[未命中] {feature_name}: 三种颜色均低于阈值")  # 记录当前字母没有可靠颜色匹配。
                    continue  # 跳过普通单模板识别并继续检查下一个正式元素。
                matched_count += 1  # 累加成功识别字母及颜色的元素数量。
                matched_boxes.append(feature_box)  # 保存最高分颜色对应的匹配框供统一绘制。
                self.log_info(f"[命中] {feature_name}: {feature_box.confidence * 100:.2f}% ({area_color})")  # 记录字母、置信度和识别颜色。
                continue  # 当前占领区已经完成颜色识别，不再重复使用原始灰色模板。
            feature_box = self.find_one(feature_name, threshold=-1.0)  # 使用最低阈值取得该元素在当前位置附近的最佳原始匹配结果。
            if feature_box is None:  # 检查模板引擎是否返回了可评分的候选区域。
                self.log_info(f"[无分数][{method_label}] {feature_name}")  # OCR 无匹配文字时不会提供模板相似度。
                continue  # 继续检查下一个模板元素。
            confidence = float(feature_box.confidence)  # 把模板引擎返回的置信度转换为普通浮点数。
            if confidence >= self.threshold:  # 判断最佳匹配是否达到任务配置的正式阈值。
                matched_count += 1  # 累加本帧成功命中的元素数量。
                matched_boxes.append(feature_box)  # 保存命中框，排除所有低于阈值的候选框。
                self.log_info(f"[命中][{method_label}] {feature_name}: {confidence * 100:.2f}%")  # 标明文字识别或模板匹配的置信度。
            else:  # 最佳匹配低于正式阈值时只作为诊断分数展示。
                self.log_info(f"[未命中][{method_label}] {feature_name}: {confidence * 100:.2f}%")  # 记录当前识别方法的候选分数。
        scene = self._detect_scene(refresh=False)  # 使用已经检查过的同一张缓存截图执行共享场景判断。
        self.clear_box()  # 清除模板匹配过程自动产生的低分候选框和蓝色搜索区域。
        if matched_boxes:  # 仅在本轮存在达到阈值的结果时重新启用覆盖层绘框。
            self.draw_boxes("screen_recognition_matches", matched_boxes, color="red")  # 只绘制达到正式阈值的命中框。
        scene_labels = {"main": "主界面", "battle_mode": "战斗模式选择", "addon": "加成页面", "equipment": "装备页面", "queue": "战斗排队", "battle_start": "等待战斗开始", "battle": "战斗界面", "map": "大地图", "result": "战斗结算", "menu": "菜单", "leave_battle": "离开战斗", "unknown": "未知界面"}  # 定义内部场景名称对应的中文显示文本。
        scene_labels.update({"battle_survey": "战斗满意度问卷", "inactivity_penalty": "消极行为提示", "match_stopped": "匹配已停止", "leave_confirm": "离开战斗确认", "popup": "可关闭弹窗", "login": "登录界面", "claim_reward": "领取奖励", "reward_screen": "奖励展示", "container": "集装箱页面"})  # 补充新增按钮对应的只读场景名称。
        scene_label = scene_labels.get(scene, scene)  # 获取可直接展示给用户的场景名称。
        self.info_set("Detected Scene", f"{scene_label} ({scene})")  # 在任务信息区域持续显示本次判断结果。
        self.log_info(f"元素检查完成：{matched_count}/{len(feature_names)} 个达到阈值；当前界面：{scene_label} ({scene})。", notify=True)  # 汇总命中数量并通知最终场景。

    @staticmethod  # 声明模板名称读取方法不依赖任务实例状态。
    def _load_feature_names():  # 从应用实际使用的 COCO 文件读取全部模板分类名称。
        feature_json = app_config["template_matching"]["coco_feature_json"]  # 读取正式模板 JSON 文件路径。
        with open(feature_json, "r", encoding="utf-8") as feature_file:  # 使用 UTF-8 打开 COCO 标注文件。
            feature_data = json.load(feature_file)  # 解析模板图片、标注和分类信息。
        return [category["name"] for category in feature_data["categories"]]  # 按正式配置顺序返回全部元素名称。
