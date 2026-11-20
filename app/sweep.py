"""整体灰度偏移 d 的判定保持区间求解（精确分段线性分析）。

问题
----
对图内归一化灰度统一加上连续偏移量 d ∈ [-0.25, 0.25]，再截到 [0, 1]，
按现有两层卷积计算（图外填充仍为零），分数 >= 0 记为“裂缝”。给定模型
输出网格上的 1～8 个复核点，求包含 d = 0 的最大连续区间，使区间内每个
复核点都保持 d = 0 时的原判；并说明左右端点能否取到、各由哪些点限制，
或是否已到达偏移范围边界。

为什么可以精确求解
------------------
固定权重与输入后，每个输出分数都是若干截断像素
``g_k(d) = clip(x_k + d, 0, 1)`` 的线性组合再加偏置：
  * clip 只在 d = -x_k（下截断开始）与 d = 1 - x_k（上截断开始）处折弯；
  * layer1 的 ReLU 在固定截断状态时仍是对 d 的线性函数，只可能多出
    ``alpha*d + beta = 0`` 这类折点；
  * layer2 无激活。
因此每个复核点的分数对 d 分段线性且连续，折点有限、可解析求出。把所有
复核点的折点与零点排序，得到有限个开子区间；每个子区间内所有判定不变。
从 d = 0 向两侧逐段扩展，遇到第一个“某点改变原判”的开子区间即停止——
即使该点之后恢复，中间也已越界。端点能否取到用端点处的精确分数按
“>= 0 即裂缝”判定。

实现采用 Python ``Fraction`` 精确表示 float 权重/像素（任何 float 都是
二进制有理数），折点、零点和端点比较均无浮点误差。页面展示的 d = 0
分数取自现有的 ``run_full_baseline``（float64）；仅在分数恰好落到
1e-17 量级的极端情形下两者符号可能不同，此时会在 notes 中提示。
"""

from __future__ import annotations

import bisect
from fractions import Fraction

import numpy as np

from .conv import (
    KERNEL,
    L1_OUT_CH,
    L2_DILATION,
    layer1_out_size,
    run_full_baseline,
    validate_weights,
)

D_MIN = Fraction(-1, 4)
D_MAX = Fraction(1, 4)
D_MIN_F = -0.25
D_MAX_F = 0.25
MAX_POINTS = 8
THRESHOLD = 0.0

_LABELS = {True: "裂缝", False: "非裂缝"}


class SweepError(ValueError):
    """输入或计算条件不满足时的可解释错误。"""


# --------------------------------------------------------------------------
# 单个复核点的精确分段线性模型
# --------------------------------------------------------------------------
class _PointModel:
    """输出点 (r, c) 的分数关于 d 的精确分段线性表示。

    只跟踪真正影响该输出的部分：layer2 的 9 个 tap 指向 3×3 个 layer1
    位置，各自又只读 3×3 个输入像素（共 9×9=81 个位置，图外为零填充）。
    layer1 预激活按像素截断状态的组合做缓存，折点扫描时只重算状态变化的项。
    """

    def __init__(self, image: np.ndarray, w1f, b1f, w2f, b2f,
                 r: int, c: int) -> None:
        self.r, self.c = r, c
        h, w = image.shape
        self.ho, self.wo = layer1_out_size(h, w)

        # ---- layer2 实际读取的 3x3 个 layer1 位置 ----
        self.targets: list[tuple[int, int]] = []
        target_idx: dict[tuple[int, int], int] = {}
        for kh in range(KERNEL):
            for kw in range(KERNEL):
                jr = r + L2_DILATION * (kh - 1)
                jc = c + L2_DILATION * (kw - 1)
                if 0 <= jr < self.ho and 0 <= jc < self.wo:
                    target_idx[(jr, jc)] = len(self.targets)
                    self.targets.append((jr, jc))

        # ---- 每个 layer1 位置的输入 tap（图外按零填充丢弃） ----
        raw_taps: list[list[tuple[tuple[int, int], list[Fraction]]]] = []
        for jr, jc in self.targets:
            taps = []
            for kh in range(KERNEL):
                for kw in range(KERNEL):
                    rr = 2 * jr + kh - 1
                    cc = 2 * jc + kw - 1
                    if 0 <= rr < h and 0 <= cc < w:
                        taps.append(((rr, cc),
                                     [w1f[o][kh][kw] for o in range(L1_OUT_CH)]))
            raw_taps.append(taps)

        # ---- 感受野内像素（精确有理数表示，去重） ----
        self.keys: list[tuple[int, int]] = []
        self.px: list[Fraction] = []
        key_pos: dict[tuple[int, int], int] = {}
        for taps in raw_taps:
            for key, _ in taps:
                if key not in key_pos:
                    key_pos[key] = len(self.keys)
                    self.keys.append(key)
                    self.px.append(Fraction(float(image[key[0], key[1]])))
        self.taps_idx = [[(key_pos[key], wvec) for key, wvec in taps]
                         for taps in raw_taps]

        # ---- layer2 的 4x3x3 个 tap 对应的 layer1 目标索引 ----
        self.l2: list[tuple[int, int, int, int | None]] = []
        for ic in range(L1_OUT_CH):
            for kh in range(KERNEL):
                for kw in range(KERNEL):
                    jr = r + L2_DILATION * (kh - 1)
                    jc = c + L2_DILATION * (kw - 1)
                    self.l2.append((ic, kh, kw, target_idx.get((jr, jc))))

        self.b1f = b1f
        self.w2f = w2f
        self.b2f = b2f
        self._preact_cache: list[dict[bytes, list[tuple[Fraction, Fraction]]]] = [
            {} for _ in self.targets]

        # ---- 折点：clip 折点（+ d 范围端点） ----
        clip_bps = {D_MIN, D_MAX}
        for x in self.px:
            for z in (-x, Fraction(1) - x):
                if D_MIN <= z <= D_MAX:
                    clip_bps.add(z)
        self.clip_bps = sorted(clip_bps)

        # ---- 每个 clip 子区间缓存截断状态与 layer1 预激活仿射式 ----
        self.clip_cache: list[tuple[Fraction, Fraction, list[int],
                                    list[list[tuple[Fraction, Fraction]]]]] = []
        for a, b in zip(self.clip_bps, self.clip_bps[1:]):
            gating = self._gating((a + b) / 2)
            self.clip_cache.append((a, b, gating, self._preacts(gating)))

        # ---- ReLU 折点：预激活 alpha*d+beta 在 clip 子区间内的零点 ----
        relu_bps: set[Fraction] = set()
        for a, b, _gating, pre in self.clip_cache:
            for entry in pre:
                for alpha, beta in entry:
                    if alpha != 0:
                        z = -beta / alpha
                        if a < z < b:
                            relu_bps.add(z)
        self.bps = sorted(set(self.clip_bps) | relu_bps)

        # ---- 每个最终开子区间上的分数仿射式与区间内零点 ----
        self.segments: list[tuple[Fraction, Fraction, Fraction, Fraction]] = []
        roots: set[Fraction] = set()
        clip_starts = [c[0] for c in self.clip_cache]
        for a, b in zip(self.bps, self.bps[1:]):
            mid = (a + b) / 2
            ci = max(0, min(bisect.bisect_right(clip_starts, mid) - 1,
                            len(self.clip_cache) - 1))
            pre = self.clip_cache[ci][3]
            active = [[(alpha * mid + beta) > 0 for alpha, beta in entry]
                      for entry in pre]
            slope = Fraction(0)
            intercept = self.b2f
            for ic, kh, kw, sidx in self.l2:
                if sidx is None or not active[sidx][ic]:
                    continue
                alpha, beta = pre[sidx][ic]
                wv = self.w2f[ic][kh][kw]
                slope += wv * alpha
                intercept += wv * beta
            self.segments.append((a, b, slope, intercept))
            if slope != 0:
                z = -intercept / slope
                if a < z < b:
                    roots.add(z)
        self.roots = sorted(roots)

    # ---- 内部工具 ----
    def _gating(self, d: Fraction) -> list[int]:
        """每个感受野像素在偏移 d 处的截断状态：0=下钳、1=上钳、2=线性。"""
        return [0 if x + d <= 0 else (1 if x + d >= 1 else 2)
                for x in self.px]

    def _preacts(self, gating) -> list[list[tuple[Fraction, Fraction]]]:
        """每个 layer1 目标/通道的 (alpha, beta)：preact = alpha*d + beta。"""
        out = []
        for t, taps in enumerate(self.taps_idx):
            key = bytes(gating[pos] for pos, _ in taps)
            cached = self._preact_cache[t].get(key)
            if cached is None:
                alphas = [Fraction(0)] * L1_OUT_CH
                betas = [Fraction(x) for x in self.b1f]
                for pos, wvec in taps:
                    st = gating[pos]
                    if st == 2:
                        x = self.px[pos]
                        for o in range(L1_OUT_CH):
                            alphas[o] += wvec[o]
                            betas[o] += wvec[o] * x
                    elif st == 1:
                        for o in range(L1_OUT_CH):
                            betas[o] += wvec[o]
                cached = list(zip(alphas, betas))
                self._preact_cache[t][key] = cached
            out.append(cached)
        return out

    # ---- 对外：精确分数 ----
    def score_at(self, d: Fraction) -> Fraction:
        i = bisect.bisect_right(self.bps, d) - 1
        i = max(0, min(i, len(self.segments) - 1))
        _, _, slope, intercept = self.segments[i]
        return slope * d + intercept


# --------------------------------------------------------------------------
# 展示辅助
# --------------------------------------------------------------------------
def _fmt_d(d: float) -> str:
    """端点展示：9 位小数（精确值见 exact_fraction / exact_decimal）。"""
    text = f"{d:+.9f}".rstrip("0").rstrip(".")
    return "0" if text in ("+0", "-0") else text


def _interval_text(lower: float, upper: float,
                   lower_incl: bool, upper_incl: bool) -> str:
    return (f"{'[' if lower_incl else '('}{_fmt_d(lower)}, "
            f"{_fmt_d(upper)}{']' if upper_incl else ')'}")


def _label(score: Fraction) -> bool:
    return score >= 0


def _names(rows: list[dict]) -> str:
    return "、".join(f"({p['row']}, {p['col']})" for p in rows)


def _exact_fraction(d: Fraction) -> str:
    """精确端点值。分段线性零点是有理数，分母不一定是 2 的幂。"""
    if d.denominator == 1:
        return str(d.numerator)
    return f"{d.numerator}/{d.denominator}"


def _approx_decimal(d: Fraction) -> str:
    """精确端点值的高精度十进制（约 30 位有效数字，非精确值）。"""
    from decimal import Decimal, localcontext
    with localcontext() as ctx:
        ctx.prec = 40
        value = Decimal(d.numerator) / Decimal(d.denominator)
    return format(value.normalize(), "f")


# --------------------------------------------------------------------------
# 主分析入口
# --------------------------------------------------------------------------
def analyze_sweep(image: np.ndarray, weights: dict,
                  points: list[tuple[int, int]]) -> dict:
    """求解包含 d=0 的最大原判保持区间。

    输入非法时抛 ``SweepError``；任何失败都不会返回半成品区间。
    """
    w1, b1, w2, b2 = validate_weights(weights)
    if not isinstance(image, np.ndarray) or image.ndim != 2:
        raise SweepError("图像必须是二维灰度数组")
    h, w = image.shape
    if h < 1 or w < 1:
        raise SweepError("图像尺寸为空")
    ho, wo = layer1_out_size(h, w)

    if not points:
        raise SweepError("请至少选择 1 个复核点")
    if len(points) > MAX_POINTS:
        raise SweepError(f"最多选择 {MAX_POINTS} 个复核点，收到 {len(points)} 个")
    seen: set[tuple[int, int]] = set()
    norm: list[tuple[int, int]] = []
    for p in points:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise SweepError("复核点格式应为 [行, 列]")
        r, c = p
        if (isinstance(r, bool) or isinstance(c, bool)
                or not isinstance(r, (int, np.integer))
                or not isinstance(c, (int, np.integer))):
            raise SweepError("复核点行列必须为整数")
        r, c = int(r), int(c)
        if not (0 <= r < ho and 0 <= c < wo):
            raise SweepError(
                f"复核点 ({r}, {c}) 超出输出网格范围："
                f"行 0..{ho - 1}，列 0..{wo - 1}")
        if (r, c) in seen:
            raise SweepError(f"复核点 ({r}, {c}) 重复")
        seen.add((r, c))
        norm.append((r, c))

    # 现有 float64 整图推理：作为 d=0 的基准分数与原判
    baseline = run_full_baseline(image, w1, b1, w2, b2)
    if not np.isfinite(baseline).all():
        raise SweepError("整图推理结果含 NaN/Inf，无法给出保持区间")

    # float 权重/偏置/像素 -> 精确有理数（float 本身即二进制有理数）
    w1f = [[[Fraction(float(w1[o, 0, kh, kw])) for kw in range(KERNEL)]
            for kh in range(KERNEL)] for o in range(L1_OUT_CH)]
    w2f = [[[Fraction(float(w2[0, ic, kh, kw])) for kw in range(KERNEL)]
            for kh in range(KERNEL)] for ic in range(L1_OUT_CH)]
    b1f = [Fraction(float(b1[o])) for o in range(L1_OUT_CH)]
    b2f = Fraction(float(b2[0]))

    models = [_PointModel(image, w1f, b1f, w2f, b2f, r, c) for r, c in norm]
    labels = [bool(float(baseline[0, r, c]) >= THRESHOLD) for r, c in norm]

    notes: list[str] = []
    for (r, c), lab, m in zip(norm, labels, models):
        ex = m.score_at(Fraction(0))
        if bool(ex >= 0) != lab:
            notes.append(
                f"点 ({r}, {c}) 在 d=0 处 float64 分数为 "
                f"{float(baseline[0, r, c]):.3e}（判为{_LABELS[lab]}），"
                f"精确模型为 {float(ex):.3e}；二者仅在浮点舍入量级不一致，"
                "本工具以 float64 原判为准。")

    # ---- 汇总所有折点/零点，定位 d=0 所在子区间 ----
    all_bps: set[Fraction] = {Fraction(0), D_MIN, D_MAX}
    for m in models:
        all_bps.update(m.bps)
        all_bps.update(m.roots)
    bps = sorted(all_bps)
    i0 = bps.index(Fraction(0))

    def seg_mid(i: int) -> Fraction:
        return (bps[i] + bps[i + 1]) / 2

    def failing_at(d: Fraction) -> list[int]:
        return [i for i, m in enumerate(models)
                if _label(m.score_at(d)) != labels[i]]

    def scan(direction: int):
        """向一侧逐段推进；返回 (端点, 是否范围边界, 触发越界的位置)。

        先看紧邻的**开子区间**（内部判定恒定），再看该侧的第一个折点。
        折点必须单独检查：分数可能在折点处恰好触零却不改变邻段内的符号
        （切触），按“>= 0 即裂缝”的规则，切触点本身也可能破坏原判。
        """
        if direction < 0:
            i = i0 - 1
            while i >= 0:
                mid = seg_mid(i)
                if failing_at(mid):
                    return bps[i + 1], False, mid
                if bps[i] > D_MIN and failing_at(bps[i]):
                    return bps[i], False, bps[i]
                i -= 1
            return D_MIN, True, None
        i = i0
        while i + 1 < len(bps):
            mid = seg_mid(i)
            if failing_at(mid):
                return bps[i], False, mid
            if bps[i + 1] < D_MAX and failing_at(bps[i + 1]):
                return bps[i + 1], False, bps[i + 1]
            i += 1
        return D_MAX, True, None

    def build_side(direction: int, endpoint: Fraction,
                   at_bound: bool, outside_mid: Fraction | None) -> dict:
        endpoint_fails = failing_at(endpoint)
        inclusive = True if endpoint == 0 else not endpoint_fails
        outside_fails = failing_at(outside_mid) if outside_mid is not None else []
        # 切触时越界点就是端点本身，没有“端点外”的样本
        tangent = outside_mid == endpoint

        def row(i: int) -> dict:
            m = models[i]
            return {
                "row": m.r,
                "col": m.c,
                "label_at_zero": _LABELS[labels[i]],
                "score_at_zero": float(baseline[0, m.r, m.c]),
                "score_at_endpoint": float(m.score_at(endpoint)),
                "score_just_outside": (float(m.score_at(outside_mid))
                                       if outside_mid is not None and not tangent
                                       else None),
            }

        limited_by = [row(i) for i in sorted(set(outside_fails) | set(endpoint_fails))]
        out_names = [row(i) for i in sorted(outside_fails)]
        end_names = [row(i) for i in sorted(endpoint_fails)]

        if at_bound:
            bound_txt = "下界 -0.25" if direction < 0 else "上界 +0.25"
            if not inclusive:
                reason = (f"已到偏移范围{bound_txt}；端点处 {_names(end_names)} "
                          "已不再保持原判，端点不可取。")
            else:
                reason = (f"已到偏移范围{bound_txt}；端点处所有复核点仍保持原判，"
                          "端点可取。")
        else:
            parts = []
            if tangent:
                parts.append(
                    f"点 {_names(end_names)} 恰在此折点处分数为 0，判定已改变"
                    "（折点切触同样算翻转）")
            else:
                if out_names:
                    parts.append(
                        f"点 {_names(out_names)} 在端点外的子区间内已改变判定")
                if end_names:
                    parts.append(
                        f"点 {_names(end_names)} 在端点处分数为 0，判定已改变")
            reason = "；".join(parts)
            if inclusive:
                reason += "；端点本身仍满足原判，故端点可取。"
            else:
                reason += "；故端点不可取。"

        return {
            "d": float(endpoint),
            "exact_fraction": _exact_fraction(endpoint),
            "exact_decimal": _approx_decimal(endpoint),
            "inclusive": bool(inclusive),
            "at_range_bound": bool(at_bound),
            "tangent": bool(tangent),
            "limited_by": limited_by,
            "reason": reason,
        }

    lower, lower_bound, lower_mid = scan(-1)
    upper, upper_bound, upper_mid = scan(1)
    lower_info = build_side(-1, lower, lower_bound, lower_mid)
    upper_info = build_side(1, upper, upper_bound, upper_mid)

    # ---- float64 复算：端点值四舍五入到 float64 后，现有流水线是否仍原判 ----
    def recheck(d_frac: Fraction) -> dict:
        d = float(d_frac)
        shifted = np.clip(image.astype(np.float64) + d, 0.0, 1.0)
        out = run_full_baseline(shifted, w1, b1, w2, b2)[0]
        flipped = [
            {"row": r, "col": c,
             "score": float(out[r, c]), "label_at_zero": _LABELS[labels[i]]}
            for i, (r, c) in enumerate(norm)
            if bool(out[r, c] >= THRESHOLD) != labels[i]
        ]
        return {"d": d, "all_hold": not flipped, "flipped": flipped}

    for info, endpoint in ((lower_info, lower), (upper_info, upper)):
        info["float64_recheck"] = recheck(endpoint)
        rk = info["float64_recheck"]
        if info["inclusive"] and not rk["all_hold"]:
            notes.append(
                f"端点 d={_fmt_d(info['d'])} 在精确模型中满足原判，但按 float64 "
                f"流水线复算时 {_names(rk['flipped'])} 已翻转；这是端点根无法用 "
                "float64 精确表示的舍入差异，实际操作请取略靠区间内侧的 d。")
        elif not info["inclusive"] and rk["all_hold"]:
            notes.append(
                f"端点 d={_fmt_d(info['d'])} 在精确模型中已改变判定，但 float64 "
                "复算尚未体现翻转（舍入量级），仍按精确模型记为不可取。")

    lower_ids = {p["row"] * wo + p["col"] for p in lower_info["limited_by"]}
    upper_ids = {p["row"] * wo + p["col"] for p in upper_info["limited_by"]}

    point_rows = []
    for i, (r, c) in enumerate(norm):
        m = models[i]
        point_rows.append({
            "row": r,
            "col": c,
            "score_at_zero": float(baseline[0, r, c]),
            "label_at_zero": _LABELS[labels[i]],
            "score_at_zero_exact": _approx_decimal(m.score_at(Fraction(0))),
            "score_at_lower": float(m.score_at(lower)),
            "score_at_upper": float(m.score_at(upper)),
            "limits_lower": (r * wo + c) in lower_ids,
            "limits_upper": (r * wo + c) in upper_ids,
        })

    if lower_bound and upper_bound:
        status = "两侧均已到达偏移范围边界"
    elif lower_bound:
        status = "左侧已到达偏移范围边界"
    elif upper_bound:
        status = "右侧已到达偏移范围边界"
    else:
        status = "两侧均由复核点的判定翻转限制"
    notes.append(
        f"区间内 {len(norm)} 个复核点均保持 d=0 时的判定；"
        f"端点为分段线性模型的精确折点/零点，{status}。")
    if lower == 0 and upper == 0:
        notes.append("两侧紧邻 d=0 即发生翻转，保持区间退化为单点 [0, 0]。")

    return {
        "ok": True,
        "error": None,
        "threshold": THRESHOLD,
        "d_range": [D_MIN_F, D_MAX_F],
        "grid": {"rows": ho, "cols": wo},
        "interval": {
            "lower": float(lower),
            "upper": float(upper),
            "lower_inclusive": bool(lower_info["inclusive"]),
            "upper_inclusive": bool(upper_info["inclusive"]),
            "text": _interval_text(float(lower), float(upper),
                                   lower_info["inclusive"],
                                   upper_info["inclusive"]),
        },
        "lower": lower_info,
        "upper": upper_info,
        "points": point_rows,
        "notes": notes,
    }


def parse_points_text(raw: str) -> list[tuple[int, int]]:
    """解析无 JavaScript 时的文本输入，如 ``3,4; 5,6``。"""
    if raw is None or not raw.strip():
        raise SweepError("请至少选择 1 个复核点")
    out: list[tuple[int, int]] = []
    for chunk in raw.replace("\n", ";").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.replace("，", ",").split(",")]
        if len(parts) != 2:
            raise SweepError(f"无法解析复核点“{chunk}”，格式应为 行,列")
        try:
            out.append((int(parts[0]), int(parts[1])))
        except ValueError as exc:
            raise SweepError(f"无法解析复核点“{chunk}”，行列须为整数") from exc
    if not out:
        raise SweepError("请至少选择 1 个复核点")
    return out


def failure_payload(reason: str, grid: dict | None = None) -> dict:
    """计算失败时的统一返回：区间显式清空。"""
    return {
        "ok": False,
        "error": reason,
        "threshold": THRESHOLD,
        "d_range": [D_MIN_F, D_MAX_F],
        "grid": grid,
        "interval": None,
        "lower": None,
        "upper": None,
        "points": [],
        "notes": [],
    }
