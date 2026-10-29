"""页面状态及输入重试记录；不依赖截图、GUI 或真实游戏。"""

from dataclasses import dataclass, field  # 将状态、截止时间和重试次数集中保存。


@dataclass
class SceneProgress:  # 每段状态机使用独立记录，切换页面不清空已经发送的输入次数。
    scene: str | None = None
    entered_at: float = 0
    actions: dict = field(default_factory=dict)

    def observe(self, scene, now):  # 返回状态是否变化，供调用方记录转换日志。
        if scene == self.scene:
            return False
        self.scene, self.entered_at = scene, now
        return True

    def action_due(self, action, now, interval=5, limit=None):  # 只在冷却结束且次数未耗尽时允许输入。
        count, last_at = self.actions.get(action, (0, float("-inf")))
        return (limit is None or count < limit) and now - last_at >= interval  # 未指定次数上限时只受总截止时间约束。

    def record_action(self, action, now):  # 在实际发送输入后计入次数，防止已知页面无限重试。
        count, _ = self.actions.get(action, (0, 0))
        self.actions[action] = (count + 1, now)

    def count(self, action):  # 返回某个动作已经实际发送的次数，供日志和诊断使用。
        return self.actions.get(action, (0, 0))[0]

    def exhausted(self, action, now, interval=5, limit=3):  # 最后一次输入仍保留完整的页面切换时间。
        count, last_at = self.actions.get(action, (0, 0))
        return count >= limit and now - last_at >= interval
