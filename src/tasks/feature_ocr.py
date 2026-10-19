"""游戏文字元素的 OCR 规则；保留 COCO 坐标作为搜索区域，不做像素模板匹配。"""

import json  # 读取各比例现有标注的位置。
import re  # 规范空格、标点和英文大小写。
import unicodedata  # 统一 OCR 可能输出的全角舰船等级字符。
from functools import lru_cache  # 缓存只读标注，避免每次识别重复读取文件。
from pathlib import Path  # 从项目根目录解析资源路径。

from ok import Box  # 返回框架可直接点击及绘制的识别框。

from src.resolution_assets import coco_json_for_size  # 沿用当前分辨率的资源包选择。


DEFAULT_SHIP_NAME = "自由"  # 旧配置没有舰船名称时继续识别原来的船。
SHIP_NAME_FEATURES = frozenset(("Pick-First-Ship", "Libertad-Nameplate"))  # 港口卡片和战斗铭牌共用用户输入。


OCR_TEXTS = {  # 两项舰名动态读取配置，其余元素保留固定的中英文候选。
    "Pick-First-Ship": (),
    "Libertad-Nameplate": (),
    "No-Commander": ("没有指挥官", "No commander"),
    "Recall-Commander": ("召回指挥官", "Recall commander"),
    "Leave-Queue": ("离开队列", "Leave queue"),
    "Start-Battle": ("开始战斗", "Start battle"),
    "Back-To-Port": ("回到港口", "Back to port", "Return to port"),
    "Continue-Battle": ("继续战斗", "Battle on", "Continue battle"),
    "Equipment": ("装备", "Equipment"),
    "Join-Battle": ("加入战斗", "Battle", "To battle"),
    "Install-Best-Buff": ("安装最佳", "Mount best"),
    "Remove-All-Buff": ("全部移除", "Remove all"),
    "Select-Battle-Mode": ("更改战斗模式", "Change battle type", "Change battle mode"),
    "Install-Recommended-Flag": ("安装推荐", "Mount recommended"),
    "Remove-All-Flag": ("拆卸全部", "Demount all"),
    "PVE-Battle": ("联合作战", "Co-op battle", "Co-op battles"),
    "In-Battle-Queue": ("您已加入准备战斗的队列中", "You are in the queue for battle"),
    "Menu": ("菜单", "Menu"),
    "Leave-Battlefield": ("离开战斗", "Leave battle"),
    "Leave-Battle-Confirm": ("是", "Yes"),
    "Leave-Battle-Title": ("离开战斗", "Leave battle"),
    "Map-Tutorial": ("自动驾驶控制", "Autopilot controls"),
    "Continue-Battle-After-Sunk": ("继续战斗", "Battle on", "Continue battle"),
    "Claim-Reward": ("收集您的奖励", "Collect your reward", "Collect your rewards"),
    "Close-Reward-Screen": ("关闭", "Close"),
    "Login-Game": ("登录", "Log in", "Login"),
    "Control-Camera": ("镜头控制说明", "Camera controls"),
    "Asymmetry-Battle": ("非对称战斗", "Asymmetric battle", "Asymmetric battles"),
    "Confirm-Container": ("是", "Yes"),
    "Pick-Container": ("每日补给箱", "Daily containers", "Daily container"),
    "Container-Menu": ("补给箱", "Containers", "Container"),
    "Match-Stopped": ("玩家匹配已停止", "Matchmaking has stopped", "Matchmaking stopped"),
    "Battle-Survey-Title": ("您对刚刚进行的这场战斗满意度如何？", "How satisfied are you with the battle you just played?"),
    "Battle-Survey-Close": ("关闭", "Close"),
    "Inactivity-Penalty-Title": ("消极行为", "Unsporting conduct", "Unsportsmanlike conduct"),
    "Inactivity-Penalty-Close": ("关闭", "Close"),
}

MODAL_CLOSE_TITLES = {  # 同名关闭按钮必须在同一帧确认弹窗标题，不能只凭通用文字点击。
    "Battle-Survey-Close": "Battle-Survey-Title",
    "Inactivity-Penalty-Close": "Inactivity-Penalty-Title",
}

MODAL_REGIONS = {  # 相对区域适配不同分辨率，不依赖某一个比例的模板是否齐全。
    "Battle-Survey-Title": (.34, .34, .66, .44),
    "Inactivity-Penalty-Title": (.40, .34, .60, .48),
    "Battle-Survey-Close": (.43, .47, .58, .57),
    "Inactivity-Penalty-Close": (.43, .47, .58, .57),
}


def normalized(text):  # 完整文字匹配允许空白、大小写和标点差异，不使用单字模糊匹配。
    return re.sub(r"[\W_]+", "", text.casefold())


def ship_name(task):  # 配置缺失时兼容旧用户，空值或错误类型不能静默选回默认舰船。
    value = task.config.get("Ship Name", DEFAULT_SHIP_NAME)
    return value.strip() if isinstance(value, str) else ""


def matches_ship_name(text, wanted, *, allow_substring=False):  # 战斗铭牌使用船名子串匹配；港口仍完整匹配船名及可选等级。
    actual = normalized(unicodedata.normalize("NFKC", text))
    expected = normalized(unicodedata.normalize("NFKC", wanted))
    if not expected:
        return False
    if allow_substring:
        return expected in actual  # 容忍 OCR 将等级、图标或血量与船名合并。
    tiers = ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi")
    return actual == expected or any(actual == tier + expected for tier in tiers)  # 港口避免误选包含相同子串的其他船。


@lru_cache(maxsize=3)
def annotations(relative_path):  # 只加载坐标，不依赖模板图像内容。
    path = Path(__file__).resolve().parents[2] / relative_path
    if not path.is_file():  # 缺少该比例标注时不猜测位置。
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    names = {item["id"]: item["name"] for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}
    return {names[item["category_id"]]: (item["bbox"], images[item["image_id"]])
            for item in data["annotations"]}


def annotated_box(name, frame):  # 按实际传入截图缩放标注，支持原尺寸与半尺寸。
    height, width = frame.shape[:2]
    entry = annotations(coco_json_for_size(width, height)).get(name)
    if entry is None:
        return None
    (x, y, w, h), image = entry
    sx, sy = width / image["width"], height / image["height"]
    return Box(x * sx, y * sy, w * sx, h * sy, name=name)


def clipped_box(frame, left, top, right, bottom):  # 所有 OCR 区域先裁到画面内，避免负数切片。
    height, width = frame.shape[:2]
    left, top = max(0, round(left)), max(0, round(top))
    right, bottom = min(width, round(right)), min(height, round(bottom))
    return Box(left, top, right - left, bottom - top) if right > left and bottom > top else None


def search_box(name, frame):  # 采用实测过的邻近扩展范围，保留上下文文字。
    height, width = frame.shape[:2]
    if name in MODAL_REGIONS:
        left, top, right, bottom = MODAL_REGIONS[name]
        return clipped_box(frame, width * left, height * top, width * right, height * bottom)
    if name == "Match-Stopped":
        return clipped_box(frame, width * .38, height * .36, width * .62, height * .48)  # 只查找匹配停止弹窗的完整标题，避免匹配普通确定按钮。
    if name == "Pick-First-Ship":
        return clipped_box(frame, 0, height * .7, width, height)  # 港口选船搜索底部 30% 的全部行列。
    feature = annotated_box(name, frame)
    if feature is None:
        return None
    if name == "Pick-Container":
        return clipped_box(frame, 0, 0, width * .5, height)  # 每日补给箱随左侧活动列表上下移动，搜索整个左半屏。
    dx, dy = max(width * .035, feature.width * .35), max(height * .025, feature.height * .5)
    if name in ("Continue-Battle", "Continue-Battle-After-Sunk"):
        dx, dy = feature.width * 1.5, feature.height * 1.5  # 同名按钮仍使用原有四倍局部范围。
    right = feature.x + feature.width + dx
    if name == "Control-Camera":
        right += width * .05  # F1 右侧需要容纳完整的“镜头控制说明”。
    return clipped_box(frame, feature.x - dx, feature.y - dy, right, feature.y + feature.height + dy)


def read_text(task, frame, region):  # 通过框架 OCR 处理繁简转换和坐标恢复，不刷新截图。
    if region is None:
        return []
    return task.ocr(box=region, frame=frame, threshold=.5, target_height=0)  # 直接识别原始裁剪，不指定目标高度或重采样截图。


def read_port_ship_text(task, frame, region):  # 分块保留小字像素，避免整条宽图触发检测器的长边压缩。
    results = []
    for top in range(region.y, region.y + region.height, 720):
        for left in range(region.x, region.x + region.width, 720):
            tile = clipped_box(frame, left, top, min(left + 960, region.x + region.width),
                               min(top + 960, region.y + region.height))  # 相邻块重叠 240 像素，容纳边界处的舰名。
            results.extend(read_text(task, frame, tile))
    return results


def unique_ship_matches(matches):  # 同一舰名跨块出现只计一次；不同位置的同名船仍视为歧义。
    results = []
    for item in sorted(matches, key=lambda item: item.confidence, reverse=True):
        for other in results:
            width = max(0, min(item.x + item.width, other.x + other.width) - max(item.x, other.x))
            height = max(0, min(item.y + item.height, other.y + other.height) - max(item.y, other.y))
            if width * height > .5 * min(item.width * item.height, other.width * other.height):
                break  # 重叠过半的识别框来自同一文字，保留置信度较高者。
        else:
            results.append(item)
    return results


def find_text(task, name, frame, threshold, box=None):  # 将文字匹配结果转换回原元素名，兼容等待和点击流程。
    region = box or search_box(name, frame)
    if region is None:
        return None
    region = clipped_box(frame, region.x, region.y, region.x + region.width, region.y + region.height)
    if region is None:
        return None
    accepted = {normalized(text) for text in OCR_TEXTS[name]}
    wanted = ship_name(task) if name in SHIP_NAME_FEATURES else ""  # 每次查询读取当前配置，修改船名立即影响两个识别入口。
    if name in SHIP_NAME_FEATURES and not normalized(wanted):
        return None  # 空船名不能退化成匹配任意文字。
    texts = read_port_ship_text(task, frame, region) if name == "Pick-First-Ship" else read_text(task, frame, region)
    matches = [item for item in texts
               if item.confidence >= threshold and
               (normalized(item.name) in accepted or
                (name == "Battle-Survey-Title" and normalized(item.name).startswith("您对刚刚") and
                 "这场" in normalized(item.name) and "战" in normalized(item.name) and
                 normalized(item.name).endswith("满意度如何")) or  # 小尺寸 OCR 会漏掉“进行”“战斗”的个别字，仍要求完整问题的首尾及战斗上下文。
                (name in SHIP_NAME_FEATURES and matches_ship_name(item.name, wanted, allow_substring=name == "Libertad-Nameplate")) or
                (name == "Control-Camera" and any(text in normalized(item.name) for text in accepted)))]  # F1 可能与完整提示合成同一行，仅该状态提示允许包含匹配。
    if name == "Pick-First-Ship":
        matches = unique_ship_matches(matches)
    if len(matches) != 1:  # 同一区域存在多个同名文字时不选择可能错误的点击目标。
        return None
    result = matches[0]
    return Box(result.x, result.y, result.width, result.height, confidence=result.confidence, name=name)


def find_ocr_feature(task, name, frame, threshold=0, box=None):  # 对同文案按钮添加同一帧中的场景约束。
    if frame is None or (name not in ("Pick-First-Ship", "Match-Stopped") and name not in MODAL_REGIONS and annotated_box(name, frame) is None):
        return None
    threshold = threshold if threshold else task.threshold
    context_threshold = max(.8, threshold)  # 诊断的低阈值不能绕过确认弹窗的上下文检查。
    if name in MODAL_CLOSE_TITLES:
        if find_text(task, MODAL_CLOSE_TITLES[name], frame, context_threshold) is None:
            return None  # 问卷只关闭，不选择或提交评价；处罚通知只关闭，不改变处罚。
    if name in SHIP_NAME_FEATURES:
        in_port = find_text(task, "Join-Battle", frame, context_threshold) is not None  # 港口和战斗中的同名文字位置接近，必须区分页面。
        if (name == "Pick-First-Ship") != in_port:
            return None  # 港口只识别选船入口；加入战斗按钮存在时不把卡片当战斗铭牌。
    if name in ("Leave-Battle-Confirm", "Continue-Battle-After-Sunk"):
        if find_text(task, "Leave-Battle-Title", frame, context_threshold) is None:
            return None
    elif name == "Continue-Battle":
        if find_text(task, "Leave-Battle-Title", frame, context_threshold) is not None:
            return None
        if find_text(task, "Back-To-Port", frame, context_threshold) is None:
            return None
    elif name == "Confirm-Container":
        if find_text(task, "Leave-Battle-Title", frame, context_threshold) is not None:
            return None
        feature = annotated_box(name, frame)
        height, width = frame.shape[:2]
        context = clipped_box(frame, feature.x - width * .09, feature.y - height * .17,
                              feature.x + feature.width + width * .09, feature.y + feature.height + height * .03)
        if not any(item.confidence >= context_threshold and
                   ("补给箱" in item.name or "container" in item.name.casefold())
                   for item in read_text(task, frame, context)):
            return None
    if name in ("PVE-Battle", "Asymmetry-Battle"):
        if (find_text(task, "Join-Battle", frame, context_threshold) is not None and
                find_text(task, "Select-Battle-Mode", frame, context_threshold) is not None):
            return None  # 港口顶部当前模式同名文字不属于模式选择按钮。
        if box is None:
            match = find_text(task, name, frame, threshold)
            if match is not None:
                return match
            height, width = frame.shape[:2]
            for top in range(0, height, max(1, round(height / 4))):  # 分块搜索保持小字清晰，并兼容模式按钮换位。
                for left in range(0, width, max(1, round(width / 4))):
                    region = clipped_box(frame, left - width * .025, top - height * .025,
                                         left + width * .275, top + height * .275)
                    match = find_text(task, name, frame, threshold, region)
                    if match is not None:
                        return match
            return None
    return find_text(task, name, frame, threshold, box)
