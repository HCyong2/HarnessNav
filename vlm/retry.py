"""VLM 调用重试：解析或校验失败最多再试若干次。"""

VLM_MAX_TRIES = 5


class VlmRetryExhausted(RuntimeError):
    """连续多次不合格或通讯失败。"""

    def __init__(self, label, last):
        """记录最后一次错误。

        Args:
            label (str): 调用名。
            last: 最后一次异常。
        """
        self.label = label
        self.last = last
        super().__init__(f"{label} 连续 {VLM_MAX_TRIES} 次失败: {last}")


def retry_call(fn, label="VLM"):
    """执行 ``fn``，失败则原样再调，最多 ``VLM_MAX_TRIES`` 次。

    Args:
        fn (callable): 无参；成功返回结果，失败抛异常。
        label (str): 日志用名称。

    Returns:
        object: ``fn`` 的返回值。

    Raises:
        VlmRetryExhausted: 用尽次数仍失败。
    """
    last = None
    for _ in range(VLM_MAX_TRIES):
        try:
            return fn()
        except VlmRetryExhausted:
            raise
        except Exception as exc:
            last = exc
    raise VlmRetryExhausted(label, last)
