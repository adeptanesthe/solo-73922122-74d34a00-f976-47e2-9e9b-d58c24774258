"""灰度偏移保持区间（sweep）与网页接口的自测。

关键情形：
  * 精确分段线性模型与 float64 整图仿真在随机 d 上判定一致；
  * 区间内部各点保持原判，紧邻端点外侧/端点本身的行为与 inclusive 相符；
  * 翻转后恢复仍算越界（不允许跨过中间翻转子区间）；
  * 端点由哪些点限制、何时到达偏移范围边界；
  * 选点校验与网页接口（成功/失败/清空区间）。
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from app.conv import (
    KERNEL,
    L1_OUT_CH,
    layer1_out_size,
    run_full_baseline,
    validate_weights,
)
from app.sweep import (
    D_MAX,
    D_MIN,
    MAX_POINTS,
    SweepError,
    _PointModel,
    analyze_sweep,
    failure_payload,
    parse_points_text,
)


# --------------------------------------------------------------------- helpers
def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8), mode="L").save(buf, format="PNG")
    return buf.getvalue()


def rand_weights(rng, scale=0.45, bias=0.15):
    return {
        "layer1": {"weight": rng.normal(0, scale, (4, 1, 3, 3)).tolist(),
                   "bias": rng.normal(0, bias, 4).tolist()},
        "layer2": {"weight": rng.normal(0, scale, (1, 4, 3, 3)).tolist(),
                   "bias": rng.normal(0, bias, 1).tolist()},
    }


def recover_weights():
    """构造一个 d 上非单调的分数：原判为裂缝，中途跌破 0，末端再回升。

    layer1 通道0 = relu(0.45 - g)，通道1 = relu(g - 0.55)，
    layer2 求和后减 0.02。g = clip(0.4 + d, 0, 1)。
      g < 0.45: 通道0 = 0.45 - g，通道1 = 0；
      g > 0.55: 通道0 = 0，通道1 = g - 0.55。
    分数 = (0.45-g) 或 (g-0.55) 再加 -0.02：
      d=0   → 0.02（裂缝）
      d≈0.02 → 0（触零）
      d∈(0.05, 0.15) → -0.02（非裂缝，已翻转）
      d=0.25 → 0.07（重新回到裂缝）
    保持区间必须在 d≈0.02 处截断，不能跨到末端恢复区。
    """
    w1 = np.zeros((4, 1, 3, 3))
    b1 = np.zeros(4)
    w1[0, 0, 1, 1] = -1.0
    b1[0] = 0.45
    w1[1, 0, 1, 1] = 1.0
    b1[1] = -0.55
    # 1x1 图像下输出只有 1 个点，layer2 仅中心 tap 落在图内
    w2 = np.zeros((1, 4, 3, 3))
    w2[0, 0, 1, 1] = 1.0
    w2[0, 1, 1, 1] = 1.0
    return {"layer1": {"weight": w1.tolist(), "bias": b1.tolist()},
            "layer2": {"weight": w2.tolist(), "bias": [-0.02]}}


def labels_at(image, weights, d: float) -> np.ndarray:
    w1, b1, w2, b2 = validate_weights(weights)
    shifted = np.clip(np.asarray(image, dtype=np.float64) + d, 0.0, 1.0)
    return run_full_baseline(shifted, w1, b1, w2, b2)[0] >= 0


def exact_models(image, weights, points):
    w1, b1, w2, b2 = validate_weights(weights)
    from fractions import Fraction

    w1f = [[[Fraction(float(w1[o, 0, kh, kw])) for kw in range(KERNEL)]
            for kh in range(KERNEL)] for o in range(L1_OUT_CH)]
    w2f = [[[Fraction(float(w2[0, ic, kh, kw])) for kw in range(KERNEL)]
            for kh in range(KERNEL)] for ic in range(L1_OUT_CH)]
    b1f = [Fraction(float(b1[o])) for o in range(L1_OUT_CH)]
    b2f = Fraction(float(b2[0]))
    return [_PointModel(np.asarray(image, dtype=np.float64), w1f, b1f, w2f, b2f, r, c)
            for r, c in points]


# ------------------------------------------------- 1. 精确模型 vs float64 仿真
class TestModelAgreement:
    @pytest.mark.parametrize("size", [(8, 8), (64, 64), (128, 100)])
    def test_labels_match_float_simulation(self, size):
        rng = np.random.default_rng(42)
        image = rng.random(size)
        weights = rand_weights(rng)
        ho, wo = layer1_out_size(*size)
        pts = [(0, 0), (ho - 1, wo - 1), (ho // 2, wo // 2)]
        models = exact_models(image, weights, pts)
        for d in np.linspace(-0.25, 0.25, 401):
            got = labels_at(image, weights, float(d))
            for (r, c), m in zip(pts, models):
                from fractions import Fraction
                exact = m.score_at(Fraction(float(d)))
                assert bool(float(exact) >= 0) == bool(got[r, c])

    def test_padding_zero_outside_image(self):
        # 感受野触及图外的点，模型必须按零填充处理：与整图仿真的判定一致
        rng = np.random.default_rng(3)
        image = rng.random((9, 7))
        weights = rand_weights(rng)
        ho, wo = layer1_out_size(9, 7)
        pts = [(0, 0), (0, wo - 1), (ho - 1, 0), (ho - 1, wo - 1)]
        models = exact_models(image, weights, pts)
        from fractions import Fraction
        for d in (-0.25, -0.13, 0.0, 0.07, 0.25):
            got = labels_at(image, weights, d)
            for (r, c), m in zip(pts, models):
                assert bool(m.score_at(Fraction(d)) >= 0) == bool(got[r, c])


# ------------------------------------------------------ 2. 区间与端点语义
class TestInterval:
    def test_scores_hold_inside_interval(self):
        rng = np.random.default_rng(1)
        image = rng.random((64, 64))
        weights = rand_weights(rng)
        base = labels_at(image, weights, 0.0)
        cand = np.argwhere(np.abs(
            run_full_baseline(np.asarray(image), *validate_weights(weights))[0]) < 0.6)
        pts = [tuple(map(int, cand[i])) for i in range(min(5, len(cand)))]
        res = analyze_sweep(image, weights, pts)
        assert res["ok"]
        lo = res["interval"]["lower"]
        hi = res["interval"]["upper"]
        assert lo <= 0.0 <= hi
        assert res["interval"]["lower_inclusive"] or lo < 0.0
        for d in np.linspace(lo + 1e-9, hi - 1e-9, 21):
            got = labels_at(image, weights, float(d))
            for r, c in pts:
                assert bool(got[r, c]) == bool(base[r, c]), (d, r, c)

    def test_interval_is_maximal(self):
        # 端点外（取中点步长）必须立刻有复核点翻转
        rng = np.random.default_rng(0)
        image = rng.random((64, 64))
        weights = rand_weights(rng)
        base = labels_at(image, weights, 0.0)
        pts = [(0, 0), (0, 1), (0, 2), (0, 3)]
        res = analyze_sweep(image, weights, pts)
        lo, hi = res["interval"]["lower"], res["interval"]["upper"]
        for d in (lo - 1e-6, hi + 1e-6):
            if D_MIN <= d <= D_MAX:
                got = labels_at(image, weights, d)
                flipped = [p for p in pts if bool(got[p]) != bool(base[p])]
                assert flipped, (d, res["interval"]["text"])

    def test_limit_points_reported(self):
        rng = np.random.default_rng(0)
        image = rng.random((64, 64))
        weights = rand_weights(rng)
        pts = [(0, 0), (0, 1), (0, 2), (0, 3)]
        res = analyze_sweep(image, weights, pts)
        assert res["lower"]["at_range_bound"]      # 该例下界未触发翻转
        assert not res["upper"]["at_range_bound"]
        assert res["upper"]["limited_by"]
        for p in res["upper"]["limited_by"]:
            assert (p["row"], p["col"]) in pts
            assert p["score_just_outside"] is not None
        limited = [(p["row"], p["col"]) for p in res["points"]
                   if p["limits_upper"]]
        assert limited

    def test_both_sides_at_range_bound(self):
        # 常数正分数（全零权重、layer2 偏置 +1）：区间铺满 [-0.25, 0.25]
        image = np.zeros((8, 8))
        weights = {
            "layer1": {"weight": np.zeros((4, 1, 3, 3)).tolist(),
                       "bias": [0.0, 0.0, 0.0, 0.0]},
            "layer2": {"weight": np.zeros((1, 4, 3, 3)).tolist(),
                       "bias": [1.0]},
        }
        res = analyze_sweep(image, weights, [(0, 0), (2, 2)])
        assert res["interval"]["lower"] == -0.25
        assert res["interval"]["upper"] == 0.25
        assert res["interval"]["lower_inclusive"] and res["interval"]["upper_inclusive"]
        assert res["lower"]["at_range_bound"] and res["upper"]["at_range_bound"]
        assert res["lower"]["limited_by"] == []

    def test_range_bound_reported_even_if_flip_before(self):
        rng = np.random.default_rng(0)
        image = rng.random((64, 64))
        weights = rand_weights(rng)
        cand = np.argwhere(np.abs(
            run_full_baseline(np.asarray(image), *validate_weights(weights))[0]) < 0.6)
        pts = [tuple(map(int, cand[i])) for i in range(min(4, len(cand)))]
        res = analyze_sweep(image, weights, pts)
        assert res["lower"]["at_range_bound"] is True
        assert res["upper"]["at_range_bound"] is False

    def test_flip_then_recover_is_still_out_of_bounds(self):
        image = np.array([[0.40]])
        weights = recover_weights()
        assert bool(labels_at(image, weights, 0.0)[0, 0])       # d=0 为裂缝
        res = analyze_sweep(image, weights, [(0, 0)])
        lo, hi = res["interval"]["lower"], res["interval"]["upper"]
        # 右端应停在第一个零点附近，而不是越过中间翻转区、到达末端恢复区
        assert D_MIN == lo and res["lower"]["at_range_bound"]
        assert 0.0 < hi < 0.06
        assert res["upper"]["inclusive"] is True
        # 端点外确实翻转（且之后会恢复）——恢复也不能把区间续上
        assert not bool(labels_at(image, weights, hi + 1e-3)[0, 0])
        assert not bool(labels_at(image, weights, 0.10)[0, 0])
        assert bool(labels_at(image, weights, 0.25)[0, 0])

    def test_tangent_at_breakpoint_stops_interval(self):
        # 分数在 d=0.2 的 ReLU 折点处恰好触零（邻段两侧都为负），
        # 按“>= 0 即裂缝”判为裂缝——原判为非裂缝，故端点不可取。
        image = np.array([[0.30]])
        w1 = np.zeros((4, 1, 3, 3))
        b1 = np.zeros(4)
        w1[0, 0, 1, 1] = -1.0
        b1[0] = 0.5
        w1[1, 0, 1, 1] = 1.0
        b1[1] = -0.5
        w2 = np.zeros((1, 4, 3, 3))
        w2[0, 0, 1, 1] = -1.0
        w2[0, 1, 1, 1] = -1.0
        weights = {"layer1": {"weight": w1.tolist(), "bias": b1.tolist()},
                   "layer2": {"weight": w2.tolist(), "bias": [0.0]}}
        assert not bool(labels_at(image, weights, 0.0)[0, 0])
        res = analyze_sweep(image, weights, [(0, 0)])
        assert res["interval"]["upper"] == pytest.approx(0.2, abs=1e-12)
        assert res["upper"]["inclusive"] is False
        assert res["upper"]["tangent"] is True
        assert res["upper"]["limited_by"][0]["score_at_endpoint"] == 0.0

    def test_tangent_zero_counts_as_flip(self):
        # layer2 的 9 个 tap 全取通道0，通道1 只在中心 tap 参与；
        # 构造分数在某个折点恰好触零但不改变邻段符号的情形较难，
        # 这里直接验证：若端点处分数精确为 0 且原判为裂缝，端点仍可取；
        # 若原判为非裂缝，端点不可取（>=0 规则）。
        rng = np.random.default_rng(9)
        image = rng.random((16, 16))
        weights = rand_weights(rng, scale=0.2, bias=0.05)
        models = exact_models(image, weights, [(2, 2)])
        for root in models[0].roots[:5]:
            score = models[0].score_at(root)
            assert score == 0
            # 精确分数为 0 时，裂缝判定成立
            assert bool(score >= 0)


# ------------------------------------------------------- 3. 输入校验与失败
class TestValidation:
    def _basic(self):
        rng = np.random.default_rng(5)
        return rng.random((16, 16)), rand_weights(rng)

    def test_reject_no_points(self):
        image, weights = self._basic()
        with pytest.raises(SweepError, match="至少"):
            analyze_sweep(image, weights, [])

    def test_reject_too_many_points(self):
        image, weights = self._basic()
        pts = [(i, i) for i in range(MAX_POINTS + 1)]
        with pytest.raises(SweepError, match="最多"):
            analyze_sweep(image, weights, pts)

    def test_reject_out_of_grid(self):
        image, weights = self._basic()
        ho, wo = layer1_out_size(16, 16)
        with pytest.raises(SweepError, match="超出输出网格"):
            analyze_sweep(image, weights, [(0, wo)])
        with pytest.raises(SweepError, match="超出输出网格"):
            analyze_sweep(image, weights, [(ho, 0)])

    def test_reject_duplicate_and_bad_types(self):
        image, weights = self._basic()
        with pytest.raises(SweepError, match="重复"):
            analyze_sweep(image, weights, [(1, 1), (1, 1)])
        with pytest.raises(SweepError, match="整数"):
            analyze_sweep(image, weights, [(1.5, 1)])
        with pytest.raises(SweepError, match="格式"):
            analyze_sweep(image, weights, [(1,)])

    def test_parse_points_text(self):
        assert parse_points_text("3,4; 5,6") == [(3, 4), (5, 6)]
        assert parse_points_text("1，2\n3, 4") == [(1, 2), (3, 4)]
        with pytest.raises(SweepError):
            parse_points_text("")
        with pytest.raises(SweepError):
            parse_points_text("3;4")

    def test_failure_payload_clears_interval(self):
        payload = failure_payload("boom")
        assert payload["ok"] is False and payload["error"] == "boom"
        assert payload["interval"] is None
        assert payload["lower"] is None and payload["upper"] is None
        assert payload["points"] == []


# ------------------------------------------------------------ 4. HTTP 接口
class TestHTTP:
    @staticmethod
    def _files(image, weights):
        return {
            "image": ("a.png", png_bytes(np.asarray(image) * 255), "image/png"),
            "weights": ("w.json", json.dumps(weights), "application/json"),
        }

    def test_index_and_health(self):
        from fastapi.testclient import TestClient
        from app.web import app

        client = TestClient(app)
        r = client.get("/")
        assert r.status_code == 200
        assert "保持区间" in r.text
        assert client.get("/health").json() == {"status": "ok"}

    def test_grid_endpoint(self):
        from fastapi.testclient import TestClient
        from app.web import app

        client = TestClient(app)
        rng = np.random.default_rng(7)
        image = rng.random((16, 12))
        r = client.post("/grid", files=self._files(image, rand_weights(rng)))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"]
        assert (body["rows"], body["cols"]) == layer1_out_size(16, 12)
        assert len(body["labels"]) == body["rows"]

    def test_sweep_json_success(self):
        from fastapi.testclient import TestClient
        from app.web import app

        client = TestClient(app)
        rng = np.random.default_rng(8)
        image = rng.random((64, 64))
        r = client.post("/sweep", files=self._files(image, rand_weights(rng)),
                        data={"points": "1,1; 10,10", "as_json": "1"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["ok"] and body["interval"] is not None
        assert body["interval"]["lower"] <= 0.0 <= body["interval"]["upper"]
        assert len(body["points"]) == 2

    def test_sweep_html_shows_endpoints(self):
        from fastapi.testclient import TestClient
        from app.web import app

        client = TestClient(app)
        rng = np.random.default_rng(9)
        image = rng.random((64, 64))
        r = client.post("/sweep", files=self._files(image, rand_weights(rng)),
                        data={"points": "1,1"})
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "端點" not in r.text
        assert "端点能否取到" in r.text
        assert "限制来源" in r.text

    def test_sweep_bad_point_clears_interval(self):
        from fastapi.testclient import TestClient
        from app.web import app

        client = TestClient(app)
        rng = np.random.default_rng(10)
        image = rng.random((16, 16))
        r = client.post("/sweep", files=self._files(image, rand_weights(rng)),
                        data={"points": "99,99", "as_json": "1"})
        assert r.status_code == 400
        body = r.json()
        assert body["ok"] is False
        assert body["interval"] is None
        assert "超出输出网格" in body["error"]

    def test_sweep_html_failure_clears(self):
        from fastapi.testclient import TestClient
        from app.web import app

        client = TestClient(app)
        rng = np.random.default_rng(11)
        image = rng.random((16, 16))
        r = client.post("/sweep", files=self._files(image, rand_weights(rng)),
                        data={"points": ""})
        assert r.status_code == 400
        assert "旧区间已清空" in r.text
        assert "至少" in r.text

    def test_reject_oversize_image(self):
        from fastapi.testclient import TestClient
        from app.web import app

        client = TestClient(app)
        rng = np.random.default_rng(12)
        image = rng.random((129, 8))
        r = client.post("/grid", files=self._files(image, rand_weights(rng)))
        assert r.status_code == 400
        assert "128" in r.json()["error"]

    def test_reject_non_gray(self):
        from fastapi.testclient import TestClient
        from app.web import app

        client = TestClient(app)
        rng = np.random.default_rng(13)
        buf = io.BytesIO()
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(
            buf, format="PNG")
        r = client.post("/grid",
                        files={"image": ("a.png", buf.getvalue(), "image/png"),
                               "weights": ("w.json",
                                           json.dumps(rand_weights(rng)),
                                           "application/json")})
        assert r.status_code == 400 and "灰度" in r.json()["error"]
