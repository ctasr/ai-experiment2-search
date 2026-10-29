"""实验图共用的中文字体选择，避免反复请求系统不存在的字体。"""


def configure_chinese_font() -> str:
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    candidates = (
        "SimSun",
        "Noto Sans CJK SC",
        "PingFang SC",
        "WenQuanYi Micro Hei",
    )
    for family in candidates:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=family), fallback_to_default=False
            )
        except ValueError:
            continue
        plt.rcParams["font.family"] = [family]
        plt.rcParams["axes.unicode_minus"] = False
        return family
    raise RuntimeError("绘制中文图表需要安装宋体或Noto Sans CJK SC等中文字体")
