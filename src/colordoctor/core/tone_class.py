"""影调分类（九种影调）：按直方图描述画面现在是什么影调，不评判对错。

摄影里的影调 = 基调 × 调长：
  基调（高 / 中 / 低）：画面的主要亮度落在直方图的哪一段——亮部、中间调还是暗部占得最多。
  调长（长 / 中 / 短）：从最暗到最亮跨了多大范围——跨度大是长调（反差大），跨度小是短调（柔和、平）。
直方图按 PS 的“明度”算（0.30R + 0.59G + 0.11B，0–255），分成 10 格（每格约 25.5 级）：
  第 0–3 格为暗部、4–6 格为中间调、7–9 格为亮部，哪一段像素最多就是什么基调；
  去掉两端各 2% 的零星像素后量跨度：≥ 170 级（约 2/3 个直方图）为长调、90–170 为中调、< 90 为短调。
例如：日系高调人像多是“高短调 / 高中调”，暗调情绪人像多是“低长调 / 低中调”，胶片多是“中短调”。

影调风格（TONE_STYLES）= 预制的目标影调，只管明暗和反差，和颜色模板互不绑定。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from colordoctor.core import color_spaces as cs

KEYS = ("高", "中", "低")
LENGTHS = ("长", "中", "短")
TONE_NAMES = tuple(f"{k}{n}调" for k in KEYS for n in LENGTHS)  # 高长调 … 低短调

# 每种影调的一句话说明（显示在九宫格里）
TONE_DESC = {
    "高长调": "明亮为主、带少量深色，清爽有力度",
    "高中调": "明亮、层次适中，干净通透",
    "高短调": "明亮柔和、反差小，轻盈朦胧（日系 / 高调人像）",
    "中长调": "亮暗都有、层次最丰富，最常见",
    "中中调": "中间调为主、反差适中，自然平和",
    "中短调": "中间调为主、反差小，柔和偏灰（胶片 / 莫兰迪）",
    "低长调": "暗为主、有亮点，强烈有戏剧性（暗调情绪 / 伦勃朗光）",
    "低中调": "暗为主、反差适中，沉稳含蓄",
    "低短调": "暗而平、几乎没有亮部，压抑神秘",
}

ZONES = 10
EDGE = 0.02  # 两端各去掉 2% 的零星像素（高光点、死黑）再量跨度
LONG_SPAN = 170.0  # 跨度（明度级）≥ 此值为长调
SHORT_SPAN = 90.0  # 跨度 < 此值为短调
NEAR = 12.0  # 跨度离分界不到这么多级时提示“接近 …”
LUMA = np.array([0.30, 0.59, 0.11])  # PS“明度”直方图的权重


def luminance(rgb: np.ndarray) -> np.ndarray:
    """sRGB 8 位像素 -> PS 明度（0–255，浮点）。"""
    return np.asarray(rgb, dtype=float)[..., :3] @ LUMA


@dataclass(frozen=True)
class ToneClass:
    name: str  # 例如 "低长调"
    key: str  # 高 / 中 / 低
    length: str  # 长 / 中 / 短
    shares: tuple[float, float, float]  # 暗部 / 中间调 / 亮部 各占多少
    lo: float  # 去掉 2% 后最暗处（明度 0–255）
    hi: float  # 去掉 2% 后最亮处
    span: float  # 跨度（明度级，0–255）
    histogram: tuple[float, ...]  # 32 段直方图（占比），用于显示
    near: str = ""  # 处在边界上时，相邻的影调（例如“低中调”；基调和调长都在分界附近时是“低中调、低长调”）

    @property
    def desc(self) -> str:
        return TONE_DESC[self.name]


def _colours(rgb: np.ndarray, valid: np.ndarray | None = None):
    """原图尺寸的像素 -> (不同的颜色, 每种的像素数)：直方图、分位数按像素数加权，和逐个像素算一样。"""
    from colordoctor.core.pixel_table import colour_counts

    px = np.asarray(rgb)
    if px.dtype != np.uint8:
        px = np.clip(np.rint(px[..., :3]), 0, 255).astype(np.uint8)
    cols, w = colour_counts(px, valid)
    return cols, w.astype(np.float64)


def classify(rgb: np.ndarray, valid: np.ndarray | None = None) -> ToneClass:
    """画面（或 valid 范围内）的影调：原图尺寸的每个像素都算。"""
    return classify_colours(*_colours(rgb, valid))


def classify_colours(colours: np.ndarray, weights) -> ToneClass:
    """colours：不同的颜色；weights：每种颜色的像素数。"""
    y = luminance(colours).ravel()
    w = np.asarray(weights, dtype=np.float64).ravel()
    if y.size == 0 or w.sum() <= 0:
        raise ValueError("没有像素")
    tot = w.sum()
    zones = np.minimum((y // (256 / ZONES)).astype(int), ZONES - 1)
    counts = np.bincount(zones, weights=w, minlength=ZONES) / tot
    shares = (float(counts[:4].sum()), float(counts[4:7].sum()), float(counts[7:].sum()))  # 暗 / 中 / 亮
    key = ("低", "中", "高")[int(np.argmax(shares))]
    lo, hi = (float(v) for v in cs.weighted_percentile(y, w, [EDGE * 100, 100 - EDGE * 100]))
    span = hi - lo
    length = "长" if span >= LONG_SPAN else ("中" if span >= SHORT_SPAN else "短")
    # 边界提示：主导的一段只比第二多一点，或跨度离分界很近
    # 基调和调长可能同时在分界附近：两个相邻的都列出来（以前只能说一个，另一边一点点变化就跳过去了却没提示）
    order = sorted(range(3), key=lambda i: -shares[i])
    nears = []
    if shares[order[0]] - shares[order[1]] < 0.08:
        nears.append(("低", "中", "高")[order[1]] + length + "调")
    if abs(span - LONG_SPAN) < NEAR:
        nears.append(key + ("中" if length == "长" else "长") + "调")
    elif abs(span - SHORT_SPAN) < NEAR:
        nears.append(key + ("短" if length == "中" else "中") + "调")
    near = "、".join(nears)
    hist = np.histogram(y, bins=32, range=(0, 256), weights=w)[0] / tot
    name = f"{key}{length}调"
    return ToneClass(name, key, length, tuple(round(s, 4) for s in shares), round(lo, 1), round(hi, 1), round(span, 1),
                     tuple(round(float(v), 4) for v in hist), near if near != name else "")


# ---------------------------------------------------------------- 目标影调的“标准直方图”（分位点）
# 每种影调的代表性分位点（明度 0–255）：P2 / P25 / P50 / P75 / P98。往目标影调调整时按分位点做直方图匹配。
_CENTER = {"高": 200.0, "中": 128.0, "低": 55.0}
_SPAN = {"长": 215.0, "中": 140.0, "短": 75.0}
MID_HALF = 22.0  # 中调：P25 / P75 离 P50 最多这么远


def target_quantiles(name: str) -> tuple[float, float, float, float, float]:
    key, length = name[0], name[1]
    c, s = _CENTER[key], _SPAN[length]
    if key == "低":  # 暗部为主：从接近黑开始往上延伸
        p2 = 8.0 if length == "长" else (14.0 if length == "中" else 20.0)
        p98 = min(250.0, p2 + s)
        p50 = c
    elif key == "高":  # 亮部为主：从接近白往下延伸
        p98 = 248.0 if length == "长" else (240.0 if length == "中" else 232.0)
        p2 = max(5.0, p98 - s)
        p50 = c
    else:
        p2, p98, p50 = c - s / 2, c + s / 2, c
    if key == "中":
        # 中间的一半像素（P25–P75）都落在中间调一段（102–179）里：中间调一定占多数，调完不会被判成低调 / 高调
        p25, p75 = p50 - min(MID_HALF, (p50 - p2) * 0.4), p50 + min(MID_HALF, (p98 - p50) * 0.4)
    else:
        p25 = p2 + (p50 - p2) * 0.6
        p75 = p50 + (p98 - p50) * 0.4
    return (round(p2, 1), round(p25, 1), round(p50, 1), round(p75, 1), round(p98, 1))


def quantiles(rgb: np.ndarray, valid: np.ndarray | None = None) -> tuple[float, float, float, float, float]:
    cols, w = _colours(rgb, valid)
    return tuple(round(float(v), 1) for v in cs.weighted_percentile(luminance(cols), w, [2, 25, 50, 75, 98]))


MAX_MOVE = 45.0  # 一次曲线每个点最多移动多少级（0–255），其余提示重新分析或回 RAW 调曝光


MIN_GAP = 12  # 曲线点之间至少相隔这么多级：太近的点会让曲线很陡，PS 的平滑曲线还会来回打折
MAX_SLOPE = 3.0  # 相邻两点之间的斜率限制在 1/3 – 3：再陡会出现断层、再平会把层次压没


def match_curve(current: tuple, target: tuple) -> tuple[list[list[int]], bool]:
    """按分位点做直方图匹配：把当前的 P2/P25/P50/P75/P98 映射到目标的分位点，得到 RGB 复合曲线的点。

    - 每个点一次最多移动 MAX_MOVE 级；
    - 最暗 / 最亮的分位离 0 / 255 不到 MIN_GAP 级（放不下一个点）时移动端点本身（抬黑场 / 压白场），否则调长改不了；
    - 点之间至少相隔 MIN_GAP 级，相邻斜率限制在 1/3 – 3，输出严格递增。
    返回 (曲线点（含输入 0 和 255 两个端点），是否被限幅)。
    """
    capped = False
    lo, hi = 0.0, 255.0
    mid: list[tuple[float, float]] = []
    for x, y in zip(current, target, strict=True):
        x, y = float(x), float(y)
        if abs(y - x) > MAX_MOVE:
            capped = True
            y = x + float(np.clip(y - x, -MAX_MOVE, MAX_MOVE))
        if x < MIN_GAP:
            # 离 0 不到 MIN_GAP 级，放不下一个点：整段按同样的量平移黑场端点（以前只在 ≤ 3 时移动，4–11 的点被悄悄丢掉）
            lo = float(np.clip(y - x, 0.0, MAX_MOVE))
        elif 255 - x < MIN_GAP:
            hi = float(np.clip(y + (255 - x), 255 - MAX_MOVE, 255.0))  # 离 255 太近：同样平移白场端点
        elif not mid or x - mid[-1][0] >= MIN_GAP:
            mid.append((x, y))
    pts, changed = limit_slopes([[0.0, lo], *[[x, y] for x, y in mid], [255.0, hi]])
    return round_points(pts), capped | changed


def round_points(pts) -> list[list[int]]:
    """曲线点取整：输出限制在 0–255 且严格递增（相邻两点取整后不会落在同一级上）。"""
    out, last = [], -1
    for x, y in pts:
        yi = int(np.clip(round(y), 0, 255))
        yi = max(yi, last + 1) if out else yi
        out.append([int(round(x)), min(yi, 255)])
        last = out[-1][1]
    return out


def limit_slopes(pts) -> tuple[list[list[float]], bool]:
    """相邻曲线点的斜率限制在 1/MAX_SLOPE – MAX_SLOPE：先从左往右、再从右往左，端点也会跟着让一点。

    pts：[[x, y], ...]（x 递增，含两个端点）。返回 (新的点, 是否有点被挪动超过 1 级)。
    """
    pts = [[float(x), float(y)] for x, y in pts]
    changed = False

    def clamp(i):  # 端点被推出 0–255 时拉回来，另一遍再把相邻的点让开（否则最后被截断，斜率又超限）
        nonlocal changed
        y = float(np.clip(pts[i][1], 0.0, 255.0))
        changed |= abs(y - pts[i][1]) > 1.0
        pts[i][1] = y

    for _ in range(2):
        clamp(0)
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            y = float(np.clip(pts[i][1], pts[i - 1][1] + dx / MAX_SLOPE, pts[i - 1][1] + dx * MAX_SLOPE))
            changed |= abs(y - pts[i][1]) > 1.0
            pts[i][1] = y
        clamp(len(pts) - 1)
        for i in range(len(pts) - 2, -1, -1):
            dx = pts[i + 1][0] - pts[i][0]
            y = float(np.clip(pts[i][1], pts[i + 1][1] - dx * MAX_SLOPE, pts[i + 1][1] - dx / MAX_SLOPE))
            changed |= abs(y - pts[i][1]) > 1.0
            pts[i][1] = y
    return pts, changed


def spline_lut(pts) -> np.ndarray:
    """PS 曲线的实际形状（256 级查找表）：PS 用平滑的三次样条穿过各点，不是点与点之间连直线。

    这里用自然三次样条近似。点之间的坡度一会儿很平一会儿很陡时，样条会冲过头（暗部被压成一片死黑、
    某一段反而往下走），用 spline_safe 加辅助点避免。
    """
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    xs = np.arange(256, dtype=float)
    n = len(x)
    if n < 3:
        return np.clip(np.interp(xs, x, y), 0, 255)
    h = np.diff(x)
    A = np.zeros((n, n))
    r = np.zeros(n)
    A[0, 0] = A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
        r[i] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, r)
    idx = np.clip(np.searchsorted(x, xs, side="right") - 1, 0, n - 2)
    hi = h[idx]
    a = (x[idx + 1] - xs) / hi
    b = (xs - x[idx]) / hi
    v = a * y[idx] + b * y[idx + 1] + ((a ** 3 - a) * M[idx] + (b ** 3 - b) * M[idx + 1]) * hi ** 2 / 6
    v = np.where(xs < x[0], y[0], np.where(xs > x[-1], y[-1], v))
    return np.clip(v, 0, 255)


SPLINE_TOL = 3.0  # PS 的平滑曲线离“点之间连直线”最多差这么多级
MAX_CURVE_POINTS = 14  # PS 曲线最多 16 个点（含两个端点）


def spline_safe(pts) -> list[list[int]]:
    """在 PS 的平滑曲线会冲过头或往回走的那一段中间加辅助点（取直线上的值），让实际曲线和设计的一致、单调。"""
    pts = [[int(x), int(y)] for x, y in pts]
    for _ in range(12):
        lut = spline_lut(pts)
        xs, ys = np.array(pts, dtype=float).T
        lin = np.interp(np.arange(256), xs, ys)
        bad = []
        for (x0, _), (x1, _) in zip(pts, pts[1:]):
            seg = slice(x0, x1 + 1)
            off = float(np.abs(lut[seg] - lin[seg]).max())
            back = bool(np.any(np.diff(lut[seg]) < -0.5))
            if (off > SPLINE_TOL or back) and x1 - x0 >= 8:
                bad.append((off + 100 * back, (x0 + x1) // 2))
        if not bad or len(pts) >= MAX_CURVE_POINTS:
            break
        for _, xm in sorted(bad, reverse=True)[: MAX_CURVE_POINTS - len(pts)]:
            pts.append([xm, int(round(np.interp(xm, xs, ys)))])
        pts.sort()
    return pts


def apply_curve_luminosity(rgb: np.ndarray, pts) -> np.ndarray:
    """曲线（线性插值近似）作用在 RGB 各通道上、图层混合模式“明度”时，结果的 PS 明度（0–255）。

    “明度”混合只取调整后画面的明度、保留原来的颜色，所以结果的明度 = Lum(曲线(R, G, B))，
    和“曲线(明度)”不一样（彩色像素差得更多）：生成曲线时用它来模拟、修正。
    """
    lut = spline_lut(pts)  # 按 PS 的平滑曲线算
    return luminance(lut[np.clip(np.asarray(rgb)[..., :3], 0, 255).astype(np.uint8)])


def fit_curve(rgb: np.ndarray, valid, current: tuple, target: tuple, rounds: int = 3) -> tuple[list[list[int]], bool]:
    """按目标分位点求曲线，并按“作用在 RGB 上、明度混合”的实际效果修正几次，让调完的分位点更接近目标。"""
    px, pw = _colours(rgb, valid)  # 每种颜色算一次，按像素数加权（原图尺寸的每个像素）
    goal = np.array(target, dtype=float)
    best = match_curve(current, target)
    best_err = None
    for _ in range(rounds):
        pts, capped = match_curve(current, tuple(goal))
        pts = spline_safe(pts)
        got = cs.weighted_percentile(apply_curve_luminosity(px, pts), pw, [2, 25, 50, 75, 98])
        err = np.array(target, dtype=float) - got
        score = float(np.abs(err).mean())
        if best_err is None or score < best_err:
            best, best_err = (pts, capped), score
        if np.abs(err).max() < 3:
            break
        goal = np.clip(goal + err, 0, 255)
    return best


# ---------------------------------------------------------------- 影调风格（预制的目标影调）
# 影调风格只决定“明暗和反差往哪里走”，和颜色无关：港风可以配青橙，也可以配黑金。
# target：主目标影调；also：也算符合的影调（已经是这些影调就不生成曲线）。
# look：看起来是什么样（给不熟悉这些叫法的人看）；scene：常见用在哪。
TONE_STYLES: dict[str, dict] = {
    # —— 偏亮
    "japanese": {"name": "日系", "target": "高短调", "also": ("高中调",),
                 "look": "明亮、柔和、反差小；暗部是浅灰而不是黑，像窗边的自然光",
                 "scene": "清新人像、生活照、街拍"},
    "candy": {"name": "糖水", "target": "高短调", "also": ("高中调",),
              "look": "很亮、很柔，几乎没有沉重的暗部，甜美梦幻（比日系更亮更软）",
              "scene": "少女写真、甜美人像、花海"},
    "korean": {"name": "韩系", "target": "高中调", "also": ("高短调",),
               "look": "明亮干净，比日系多一点层次和反差，皮肤白净",
               "scene": "精致写真、棚拍人像"},
    "cool": {"name": "清冷", "target": "高中调", "also": ("中中调",),
             "look": "干净通透、偏亮，暗部不沉、反差适中；配冷色调更显清冷疏离",
             "scene": "冬日雪景、白墙极简、清冷感人像"},
    "bright": {"name": "白亮通透", "target": "高长调", "also": ("高中调",),
               "look": "明亮为主，同时保留头发、眼睛这类少量深色，清爽有力度",
               "scene": "商业人像、白底、明亮室内"},
    "cos": {"name": "Cos", "target": "高长调", "also": ("中长调",),
            "look": "主体亮、轮廓清楚、反差大，光影和颜色都偏强",
            "scene": "Cosplay、角色写真、棚拍"},
    # —— 中间调
    "forest": {"name": "森系", "target": "中中调", "also": ("高中调", "中短调"),
               "look": "自然柔和、略偏亮，以中间调为主，不压黑",
               "scene": "草地树林、自然光人像"},
    "french": {"name": "法式", "target": "中中调", "also": ("中短调",),
               "look": "自然慵懒，反差适中，暗部柔和不死黑",
               "scene": "咖啡馆、街头、生活方式人像"},
    "chinese": {"name": "新中式", "target": "中中调", "also": ("低中调", "中短调"),
                "look": "雅致克制、略沉，反差适中；暗部像墨色但不死黑，亮部像宣纸不刺眼",
                "scene": "汉服、古风、茶室、园林"},
    "film": {"name": "胶片", "target": "中短调", "also": ("中中调",),
             "look": "柔和偏灰，黑位抬起（最暗处是深灰，不是纯黑），高光收住",
             "scene": "复古、旅行、生活记录"},
    "western": {"name": "欧美", "target": "中长调", "also": ("中中调",),
                "look": "反差足、层次厚实，暗部有分量，健康有质感",
                "scene": "欧美人像、时尚、户外阳光"},
    "cinematic": {"name": "电影感", "target": "中长调", "also": ("低长调",),
                  "look": "亮暗都有、反差大、层次丰富，有叙事感",
                  "scene": "街拍、夜景、风光人像"},
    # —— 偏暗
    "hongkong": {"name": "港风", "target": "低中调", "also": ("中中调",),
                 "look": "整体偏暗、浓郁，暗部厚重但不死黑，像 90 年代港片",
                 "scene": "霓虹夜景、复古人像、街头"},
    "moody": {"name": "暗调情绪", "target": "低长调", "also": ("低中调",),
              "look": "大面积暗部，只有主体或光源是亮的，戏剧性强",
              "scene": "情绪人像、伦勃朗光、窗光、夜景"},
}

KEY_IDX = {"低": 0, "中": 1, "高": 2}
LEN_IDX = {"短": 0, "中": 1, "长": 2}


def resolve_target(style: str = "", tone: str = "") -> tuple[str, tuple[str, ...], str]:
    """(目标影调, 也算符合的影调, 显示名)。tone（直接选的影调）优先于 style；都没有时目标为空（只看影调，不改）。"""
    if tone:
        if tone not in TONE_NAMES:
            raise ValueError(f"不认识的影调：{tone}")
        return tone, (), tone
    if style:
        if style not in TONE_STYLES:
            raise ValueError(f"不认识的影调风格：{style}")
        s = TONE_STYLES[style]
        return s["target"], tuple(s["also"]), s["name"]
    return "", (), ""


def pick_goal(cur: str, target: str, also: tuple[str, ...] = ()) -> str:
    """影调风格有几种可接受的影调时，往离现在最近的那一种走（改动最小）；一样近时用主目标。"""
    if not target:
        return ""
    return min((target, *also), key=lambda t: tone_distance(cur, t))


def key_code(name: str) -> str:
    """影调 → 颜色分析按哪种基调处理："low" / "high" / ""（中调）。"""
    return {"低": "low", "高": "high"}.get(name[:1], "") if name else ""


def path_words(cur: str, target: str) -> str:
    """从 cur 影调到 target 影调要做什么，例如“整体提亮一档、压缩反差”。"""
    if cur == target:
        return "已经是这种影调"
    dk = KEY_IDX[target[0]] - KEY_IDX[cur[0]]
    dl = LEN_IDX[target[1]] - LEN_IDX[cur[1]]
    out = []
    if dk:
        out.append(("整体提亮" if dk > 0 else "整体压暗") + ("一大截" if abs(dk) == 2 else ""))
    if dl > 0:
        out.append("拉开反差（亮暗拉开）")
    elif dl < 0:
        out.append("压缩反差（亮暗靠拢）")
    return "、".join(out)


def curve_words(pts) -> str:
    """曲线实际怎么动（按曲线点说，和图层一致）：哪些点提亮、哪些点压暗。"""
    def name(x):
        return "黑场" if x == 0 else ("白场" if x == 255 else str(x))
    up = [f"{name(x)} {x}→{y}" if x in (0, 255) else f"{x}→{y}" for x, y in pts if y - x >= 3]
    down = [f"{name(x)} {x}→{y}" if x in (0, 255) else f"{x}→{y}" for x, y in pts if x - y >= 3]
    parts = ([f"提亮 {'、'.join(up)}"] if up else []) + ([f"压暗 {'、'.join(down)}"] if down else [])
    return "；".join(parts)


def tone_distance(cur: str, target: str) -> int:
    return abs(KEY_IDX[target[0]] - KEY_IDX[cur[0]]) + abs(LEN_IDX[target[1]] - LEN_IDX[cur[1]])


def suggestions(cur: str) -> list[dict]:
    """每种影调风格离现在的影调有多远、要怎么改（按远近排序），给用户选方向时参考。"""
    out = []
    for k, s in TONE_STYLES.items():
        goal = pick_goal(cur, s["target"], tuple(s["also"]))
        out.append({"key": k, "name": s["name"], "target": goal, "distance": tone_distance(cur, goal),
                    "change": "已经符合" if goal == cur else path_words(cur, goal)})
    out.sort(key=lambda x: x["distance"])
    return out


def suggestion_text(cur: str) -> str:
    groups = {0: [], 1: [], 2: []}
    for s in suggestions(cur):
        groups[min(s["distance"], 2)].append(s["name"])
    parts = []
    if groups[0]:
        parts.append(f"已经符合：{'、'.join(groups[0])}")
    if groups[1]:
        parts.append(f"稍改就到：{'、'.join(groups[1])}")
    if groups[2]:
        parts.append(f"变化较大：{'、'.join(groups[2])}")
    return "；".join(parts)


def to_json(tc: ToneClass, target: str = "", also: tuple[str, ...] = (), label: str = "") -> dict:
    """给插件 / 界面画九宫格和直方图用。"""
    return {"name": tc.name, "key": tc.key, "length": tc.length, "desc": tc.desc, "near": tc.near,
            "shares": list(tc.shares), "lo": tc.lo, "hi": tc.hi, "span": tc.span, "histogram": list(tc.histogram),
            "target": target, "also": list(also), "label": label,
            "suggestions": suggestions(tc.name),
            "paths": {n: path_words(tc.name, n) for n in TONE_NAMES},  # 直接选某种影调时要怎么改
            "long_span": LONG_SPAN, "short_span": SHORT_SPAN}


BIG_JUMP_NOTE = "影调变化很大（跨了一个基调以上）：曲线只能拉近一部分，硬拉会让暗部噪点和断层变明显；有 RAW 时先在 Camera Raw 里调曝光 / 阴影 / 高光，再回来分析"


def tone_finding(rgb: np.ndarray, valid: np.ndarray | None, style: str, tone: str, confidence: float):
    """影调卡：现在是哪种影调；选了影调风格（或直接选了影调）时，给出往目标影调走的曲线图层。

    返回 (Finding, ToneClass, 目标影调)。
    """
    from colordoctor.core import native_layers as nl
    from colordoctor.core.color_models import Finding, Severity

    tc = classify(rgb, valid)
    main, also, label = resolve_target(style, tone)
    target = pick_goal(tc.name, main, also)
    shares = f"暗部 {tc.shares[0]:.0%} · 中间调 {tc.shares[1]:.0%} · 亮部 {tc.shares[2]:.0%}"
    evidence = [
        f"明度分布（PS 明度直方图）：{shares}",
        f"去掉两端各 2% 后从 {tc.lo:.0f} 到 {tc.hi:.0f}，跨度 {tc.span:.0f} 级（≥ {LONG_SPAN:.0f} 长调，< {SHORT_SPAN:.0f} 短调）",
    ]
    if tc.near:
        evidence.append(f"处在分界附近，也接近{tc.near}")
    head = f"影调：{tc.name}（{tc.desc}）"
    if not target:
        return Finding(
            problem=head, severity=Severity.NORMAL, evidence=tuple(evidence), causes=(), adjustments=(),
            confidence=confidence,
            notes=("只描述现在的影调，不改明暗。想往某种影调走，请在「影调风格」里选一个（PS 插件在「影调与颜色」页）",
                   "各影调风格离现在多远：" + suggestion_text(tc.name)),
        ), tc, ""
    who = label if label == target else f"「{label}」（{target}）"
    if tc.name == target:
        return Finding(
            problem=f"{head}：已经符合{who}",
            severity=Severity.NORMAL, evidence=tuple(evidence), causes=(), adjustments=(), confidence=confidence,
            notes=("影调已经到位，不生成曲线；颜色按「颜色」里选的模板单独判断",),
        ), tc, target
    cur_q = quantiles(rgb, valid)
    tgt_q = target_quantiles(target)
    pts, capped = fit_curve(rgb, valid, cur_q, tgt_q)
    dist = tone_distance(tc.name, target)
    evidence.append("分位点（P2 / P25 / P50 / P75 / P98）：现在 " + " / ".join(f"{v:.0f}" for v in cur_q)
                    + "，目标 " + " / ".join(f"{v:.0f}" for v in tgt_q))
    if max(abs(y - x) for x, y in pts) < 3:
        # 例如大部分像素已经是死黑 / 死白：曲线拉不动，不生成一条什么都不改的图层
        return Finding(
            problem=f"{head} → 目标{who}：曲线调不动",
            severity=Severity.NORMAL, evidence=tuple(evidence), causes=(), adjustments=(
                _raw_hint("直方图两端的像素已经是死黑 / 死白，曲线拉不动：请回到 RAW / Camera Raw 调曝光、阴影、高光后再分析"),),
            confidence=confidence, notes=("没有生成曲线图层",),
        ), tc, target
    params = nl.curves({"composite": pts})
    moved = nl.curves_text(params)
    text = (f"RGB 曲线（{curve_words(pts)}）：{moved}。按直方图匹配，把现在的明暗分布拉向{target}"
            + ("；一次每个点最多移动 45 级、坡度有限制，调完重新分析再看" if capped else ""))
    # 混合模式“明度”：曲线只改明暗，不会顺带把颜色变浓 / 变淡（影调风格和颜色互不干扰）
    adjs = [nl.adjustment("curves", params, text, "global", f"影调 {tc.name}→{target} · 曲线",
                          strength="中等" if dist >= 2 or capped else "小幅", blend="luminosity")]
    notes = []
    if abs(KEY_IDX[target[0]] - KEY_IDX[tc.name[0]]) == 2 or capped:
        notes.append(BIG_JUMP_NOTE)
        adjs.append(_raw_hint("有 RAW 时优先在 Camera Raw 里调曝光 / 阴影 / 高光：暗部噪点和断层更少"))
    notes.append("曲线图层的混合模式是“明度”：只改明暗、不改颜色（提亮 / 压暗时饱和度不会跟着变）")
    return Finding(
        problem=f"{head} → 目标{who}：{path_words(tc.name, target)}",
        severity=Severity.MILD if dist == 1 else Severity.NOTABLE,
        evidence=tuple(evidence), causes=(), adjustments=tuple(adjs), confidence=confidence, notes=tuple(notes),
    ), tc, target


def _raw_hint(text: str):
    from colordoctor.core.color_models import Adjustment

    return Adjustment("Camera Raw", text, None, True)
