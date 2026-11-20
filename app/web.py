"""FastAPI 网页：灰度整体偏移 d 的复核点判定保持区间演示。"""

from __future__ import annotations

import html
import json
import traceback

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from .io_utils import load_grayscale_png, load_weights_json
from .conv import run_full_baseline, validate_weights
from .sweep import (
    MAX_POINTS,
    SweepError,
    analyze_sweep,
    failure_payload,
    parse_points_text,
)

app = FastAPI(title="灰度偏移 d 判定保持区间", version="1.0.0")

MAX_IMAGE_SIDE = 128

PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>灰度偏移 d · 判定保持区间</title>
<style>
 :root{--line:#cbd2d9;--ink:#1f2933;--muted:#52606d;--bg:#f8fafc}
 body{font-family:system-ui,"PingFang SC","Microsoft YaHei",sans-serif;
  margin:2rem auto;max-width:1180px;padding:0 1rem;color:var(--ink)}
 h1{font-size:1.35rem} h2{font-size:1.08rem;margin-top:1.6rem}
 h3{font-size:.98rem;margin:1.1rem 0 .35rem}
 form{border:1px solid var(--line);border-radius:8px;padding:1rem 1.2rem;background:var(--bg)}
 fieldset{border:1px solid var(--line);border-radius:6px;padding:.55rem .9rem;margin:.9rem 0}
 legend{font-weight:600;padding:0 .3rem}
 label{display:block;margin:.55rem 0 .2rem;font-weight:600;font-size:.94rem}
 input[type=text],input[type=file]{padding:.35rem;border:1px solid var(--line);
  border-radius:5px;background:#fff}
 button{margin-top:.7rem;padding:.5rem 1.4rem;background:#1d4ed8;color:#fff;
  border:0;border-radius:6px;cursor:pointer;font-size:.95rem}
 button.ghost{background:#e4e9f0;color:var(--ink)}
 button:disabled{opacity:.5;cursor:not-allowed}
 table{border-collapse:collapse;margin:.6rem 0;font-size:.9rem}
 th,td{border:1px solid var(--line);padding:.28rem .6rem;vertical-align:top}
 th{background:#eef2f7;text-align:left}
 .ok{color:#15803d;font-weight:700}.bad{color:#b91c1c;font-weight:700}
 .crack{color:#991b1b;font-weight:600}.plain{color:#14532d;font-weight:600}
 .note{color:var(--muted);font-size:.86rem;line-height:1.55}
 .err{background:#fef2f2;border:1px solid #fca5a5;color:#991b1b;
  padding:.7rem 1rem;border-radius:6px;white-space:pre-wrap;margin-top:1rem}
 .warn{background:#fffbeb;border:1px solid #fcd34d;color:#92400e;
  padding:.6rem .9rem;border-radius:6px;margin:.6rem 0}
 .gridwrap{overflow:auto;margin-top:.6rem;border:1px solid var(--line);
  border-radius:6px;background:#fff;max-height:520px;max-width:100%}
 canvas{display:block;image-rendering:pixelated;cursor:crosshair}
 code{background:#eef2f7;padding:.1rem .35rem;border-radius:4px}
 .legend{display:flex;gap:1.1rem;flex-wrap:wrap;font-size:.83rem;margin-top:.45rem}
 .swatch{display:inline-block;width:.8rem;height:.8rem;border-radius:3px;
  vertical-align:-1px;margin-right:.25rem;border:1px solid #0002}
 .interval{font-size:1.25rem;font-weight:700;font-family:ui-monospace,monospace}
 details{margin-top:.6rem} summary{cursor:pointer;font-weight:600;font-size:.93rem}
</style></head><body>
<h1>灰度整体偏移 d · 复核点判定保持区间</h1>
<p class="note">上传每边至多 128 像素的灰度 PNG（mode=L）与原两层权重 JSON。
在模型输出网格上点选 1～8 个复核点。对图内归一化灰度统一加偏移
<code>d ∈ [−0.25, 0.25]</code>，截到 <code>[0,1]</code> 后按现有两层卷积计算
（图外零填充仍为零），分数 <code>&ge; 0</code> 记为裂缝。求包含
<code>d=0</code> 的最大连续区间，使区间内所有复核点都保持 <code>d=0</code>
时的原判。</p>

<form id="f" action="/sweep" method="post" enctype="multipart/form-data">
 <fieldset><legend>1. 输入</legend>
  <label>灰度 PNG（每边 ≤128）</label>
  <input type="file" name="image" id="image" accept="image/png" required>
  <label>两层权重/偏置 JSON</label>
  <input type="file" name="weights" id="weights" accept=".json,application/json" required>
  <button type="button" class="ghost" id="loadBtn">加载并显示输出网格</button>
  <span class="note" id="loadStatus"></span>
 </fieldset>

 <fieldset><legend>2. 复核点（在输出网格上点选）</legend>
  <div class="legend">
   <span><span class="swatch" style="background:#f3c6c6"></span>裂缝（原判）</span>
   <span><span class="swatch" style="background:#c9e4d2"></span>非裂缝（原判）</span>
   <span><span class="swatch" style="background:#fff;outline:2px solid #1d4ed8"></span>复核点</span>
   <span class="note">点击选点 / 再点取消，最多 __MAX_POINTS__ 个。</span>
  </div>
  <div class="gridwrap"><canvas id="grid" width="0" height="0"></canvas></div>
  <p class="note" id="gridHint">先上传两个文件并点“加载并显示输出网格”。</p>
  <label>复核点文本（与点选同步，可手填 <code>行,列; 行,列</code>）</label>
  <input type="text" id="pointsText" name="points" value="__POINTS__" size="60"
    placeholder="例如 3,4; 10,12">
 </fieldset>

 <fieldset><legend>3. 计算</legend>
  <label class="note"><input type="checkbox" name="as_json" id="as_json" value="1">
    以 JSON 返回（便于脚本调用）</label>
  <button type="submit" id="submitBtn">计算保持区间</button>
 </fieldset>
</form>

<div id="result">__RESULT__</div>

<script>
(function(){
  var META=__GRID_META__;
  var INITIAL=__POINTS_JSON__;
  var MAXPTS=__MAX_POINTS__;
  var canvas=document.getElementById('grid'), ctx=canvas.getContext('2d');
  var pointsText=document.getElementById('pointsText');
  var statusEl=document.getElementById('loadStatus');
  var hintEl=document.getElementById('gridHint');
  var CELL=18, meta=META, selected=[];

  function parseText(t){
    var out=[];
    (t||'').replace(/\\n/g,';').split(';').forEach(function(ch){
      ch=ch.trim(); if(!ch) return;
      var a=ch.replace(/，/g,',').split(',');
      if(a.length!==2) return;
      var r=parseInt(a[0],10), c=parseInt(a[1],10);
      if(!isNaN(r)&&!isNaN(c)) out.push([r,c]);
    });
    return out;
  }
  function sync(){
    pointsText.value=selected.map(function(p){return p[0]+','+p[1];}).join('; ');
    draw();
  }
  function draw(){
    if(!meta){ canvas.width=0; canvas.height=0; return; }
    canvas.width=meta.cols*CELL; canvas.height=meta.rows*CELL;
    for(var r=0;r<meta.rows;r++)
      for(var c=0;c<meta.cols;c++){
        ctx.fillStyle=meta.labels[r][c] ? '#f3c6c6' : '#c9e4d2';
        ctx.fillRect(c*CELL,r*CELL,CELL,CELL);
      }
    for(var r=1;r<meta.rows;r++){ ctx.strokeStyle='#ffffff'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(0,r*CELL); ctx.lineTo(meta.cols*CELL,r*CELL); ctx.stroke(); }
    for(var c=1;c<meta.cols;c++){ ctx.strokeStyle='#ffffff'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(c*CELL,0); ctx.lineTo(c*CELL,meta.rows*CELL); ctx.stroke(); }
    selected.forEach(function(p,idx){
      ctx.lineWidth=2; ctx.strokeStyle='#1d4ed8';
      ctx.strokeRect(p[1]*CELL+1,p[0]*CELL+1,CELL-2,CELL-2);
      ctx.fillStyle='#1d4ed8'; ctx.font='bold 9px sans-serif';
      ctx.fillText(String(idx+1), p[1]*CELL+2, p[0]*CELL+9);
    });
    hintEl.textContent='输出网格 '+meta.rows+' × '+meta.cols+
      '；已选 '+selected.length+' 个点（左上角数字为选择顺序）。';
  }
  canvas.addEventListener('click',function(e){
    if(!meta) return;
    var rect=canvas.getBoundingClientRect();
    var c=Math.floor((e.clientX-rect.left)/CELL);
    var r=Math.floor((e.clientY-rect.top)/CELL);
    if(r<0||c<0||r>=meta.rows||c>=meta.cols) return;
    var i=-1;
    for(var k=0;k<selected.length;k++)
      if(selected[k][0]===r&&selected[k][1]===c) i=k;
    if(i>=0) selected.splice(i,1);
    else if(selected.length>=MAXPTS){ alert('最多选择 '+MAXPTS+' 个复核点'); return; }
    else selected.push([r,c]);
    sync();
  });
  pointsText.addEventListener('input',function(){
    selected=parseText(pointsText.value);
    if(meta) selected=selected.filter(function(p){
      return p[0]>=0&&p[1]>=0&&p[0]<meta.rows&&p[1]<meta.cols;});
    draw();
  });

  var loadBtn=document.getElementById('loadBtn');
  loadBtn.addEventListener('click',function(){
    var fi=document.getElementById('image'), fw=document.getElementById('weights');
    if(!fi.files.length||!fw.files.length){
      statusEl.textContent='请先选择图像与权重文件。'; return; }
    var fd=new FormData(); fd.append('image',fi.files[0]); fd.append('weights',fw.files[0]);
    statusEl.textContent='加载中…'; loadBtn.disabled=true;
    fetch('/grid',{method:'POST',body:fd}).then(function(resp){
      return resp.json().then(function(j){ return {ok:resp.ok, body:j}; });
    }).then(function(res){
      loadBtn.disabled=false;
      if(!res.ok||!res.body.ok){
        statusEl.textContent='加载失败：'+(res.body.error||'未知错误'); return; }
      meta=res.body; selected=[]; sync();
      statusEl.textContent='已加载：输出网格 '+meta.rows+' × '+meta.cols;
    }).catch(function(err){
      loadBtn.disabled=false; statusEl.textContent='加载失败：'+err; });
  });

  // 渐进增强：已加载输入时，用 fetch 提交并就地替换结果，免去重选文件。
  // 未加载时保持普通表单提交（也保留“以 JSON 返回”的原生行为）。
  document.getElementById('f').addEventListener('submit',function(ev){
    var fi=document.getElementById('image'), fw=document.getElementById('weights');
    if(!meta || document.getElementById('as_json').checked
       || !fi.files.length || !fw.files.length) return;
    ev.preventDefault();
    var fd=new FormData();
    fd.append('image',fi.files[0]); fd.append('weights',fw.files[0]);
    fd.append('points',selected.map(function(p){return p[0]+','+p[1];}).join('; '));
    var submitBtn=document.getElementById('submitBtn');
    submitBtn.disabled=true;
    fetch('/sweep',{method:'POST',body:fd}).then(function(resp){
      return resp.text();
    }).then(function(txt){
      var doc=new DOMParser().parseFromString(txt,'text/html');
      var fresh=doc.getElementById('result');
      document.getElementById('result').innerHTML=fresh?fresh.innerHTML:txt;
      submitBtn.disabled=false;
    }).catch(function(err){
      submitBtn.disabled=false;
      document.getElementById('result').innerHTML=
        '<div class="err"><b>计算失败，旧区间已清空。</b><br>'+err+'</div>';
    });
  });

  selected=INITIAL.slice();
  if(meta) selected=selected.filter(function(p){
    return p[0]>=0&&p[1]>=0&&p[0]<meta.rows&&p[1]<meta.cols;});
  draw();
})();
</script>
</body></html>"""


# --------------------------------------------------------------------------
# 渲染辅助
# --------------------------------------------------------------------------
def _esc(text) -> str:
    return html.escape(str(text))


def _fmt_score(x: float) -> str:
    return f"{x:+.6f}"


def _render_limit_table(rows: list[dict]) -> str:
    body = "".join(
        f"<tr><td>({p['row']}, {p['col']})</td>"
        f"<td>{_esc(p['label_at_zero'])}</td>"
        f"<td>{_fmt_score(p['score_at_zero'])}</td>"
        f"<td>{_fmt_score(p['score_at_endpoint'])}</td>"
        f"<td>{'—' if p.get('score_just_outside') is None else _fmt_score(p['score_just_outside'])}</td></tr>"
        for p in rows)
    return ("<table><tr><th>限制点</th><th>d=0 判定</th><th>d=0 分数</th>"
            "<th>端点精确分数</th><th>端点外分数</th></tr>"
            f"{body}</table>")


def _render_result(payload: dict) -> str:
    if not payload.get("ok"):
        grid = payload.get("grid") or {}
        grid_txt = (f"<br>输出网格：{grid['rows']} × {grid['cols']}，可点选 "
                    f"0..{grid['rows'] - 1} 行 / 0..{grid['cols'] - 1} 列。"
                    if grid.get("rows") and grid.get("cols") else "")
        return (
            "<div class='err'><b>计算失败，旧区间已清空。</b><br>"
            f"{_esc(payload.get('error', '未知错误'))}{grid_txt}</div>")

    iv = payload["interval"]
    parts = ["<h2>结果</h2>",
             "<table>"
             f"<tr><th>保持区间</th><td class='interval'>{_esc(iv['text'])}</td></tr>"
             f"<tr><th>规则</th><td>分数 ≥ {payload['threshold']:.1f} 记为裂缝"
             "（端点均按精确计算判定）</td></tr>"
             f"<tr><th>输出网格</th><td>{payload['grid']['rows']} × "
             f"{payload['grid']['cols']}</td></tr>"
             f"<tr><th>复核点数</th><td>{len(payload['points'])}</td></tr>"
             "</table>"]

    for side, title in (("lower", "左端点（下界）"), ("upper", "右端点（上界）")):
        info = payload[side]
        cls = "ok" if info["inclusive"] else "bad"
        incl = "可取到（闭端点）" if info["inclusive"] else "不可取到（开端点）"
        source = "到达偏移范围边界" if info["at_range_bound"] else "由复核点判定翻转限制"
        parts.append(
            f"<h3>{title}：d = {info['d']:+.9f}</h3><table>"
            f"<tr><th>端点能否取到</th><td class='{cls}'>{incl}</td></tr>"
            f"<tr><th>限制来源</th><td>{source}</td></tr>"
            f"<tr><th>说明</th><td>{_esc(info['reason'])}</td></tr>"
            f"<tr><th>精确值</th><td><code>{_esc(info['exact_fraction'])}</code>"
            f"<br><span class='note'>≈ {_esc(info['exact_decimal'])}</span></td></tr>"
            "</table>")
        rk = info.get("float64_recheck")
        if rk is not None:
            verdict = ("仍全部保持原判" if rk["all_hold"] else "出现翻转")
            parts.append(
                "<p class='note'>端点值经 float64 现有流水线复核："
                f"{verdict}。</p>")
        if info["limited_by"]:
            parts.append(_render_limit_table(info["limited_by"]))
        if info.get("tangent"):
            parts.append(
                "<p class='note'>该侧越界发生在折点（分数恰好触零）处，"
                "不存在“端点外”的采样值；按“分数 ≥ 0 即裂缝”的规则，"
                "折点本身即已改变判定。</p>")

    rows = "".join(
        f"<tr><td>({p['row']}, {p['col']})</td>"
        f"<td class='{'crack' if p['label_at_zero'] == '裂缝' else 'plain'}'>"
        f"{_esc(p['label_at_zero'])}</td>"
        f"<td>{_fmt_score(p['score_at_zero'])}</td>"
        f"<td>{_fmt_score(p['score_at_lower'])}</td>"
        f"<td>{_fmt_score(p['score_at_upper'])}</td>"
        f"<td>{'—' if not (p['limits_lower'] or p['limits_upper']) else ''}"
        f"{'左' if p['limits_lower'] else ''}"
        f"{'、' if p['limits_lower'] and p['limits_upper'] else ''}"
        f"{'右' if p['limits_upper'] else ''}</td></tr>"
        for p in payload["points"])
    parts.append(
        "<h2>复核点明细</h2><table><tr><th>点 (行, 列)</th><th>d=0 判定</th>"
        "<th>d=0 分数</th><th>左端点分数</th><th>右端点分数</th>"
        f"<th>限制端点</th></tr>{rows}</table>")

    if payload["notes"]:
        parts.append("<ul class='note'>" + "".join(
            f"<li>{_esc(n)}</li>" for n in payload["notes"]) + "</ul>")
    return "".join(parts)


def _page(result: str = "", grid_meta: dict | None = None,
          points: list[tuple[int, int]] | None = None) -> str:
    points_text = "; ".join(f"{r},{c}" for r, c in (points or []))
    return (PAGE
            .replace("__MAX_POINTS__", str(MAX_POINTS))
            .replace("__GRID_META__", json.dumps(grid_meta)
                     if grid_meta is not None else "null")
            .replace("__POINTS_JSON__", json.dumps(
                [[r, c] for r, c in (points or [])]))
            .replace("__POINTS__", _esc(points_text))
            .replace("__RESULT__", result))


# --------------------------------------------------------------------------
# 输入处理
# --------------------------------------------------------------------------
def _load_inputs(image_bytes: bytes, weights_bytes: bytes):
    image = load_grayscale_png(image_bytes)
    h, w = image.shape
    if h > MAX_IMAGE_SIDE or w > MAX_IMAGE_SIDE:
        raise SweepError(
            f"图像每边至多 {MAX_IMAGE_SIDE} 像素，收到 {w}x{h}")
    weights = load_weights_json(weights_bytes)
    validate_weights(weights)
    return image, weights


def _grid_meta(image, weights) -> dict:
    w1, b1, w2, b2 = validate_weights(weights)
    out = run_full_baseline(image, w1, b1, w2, b2)[0]
    if not np.isfinite(out).all():
        raise SweepError("整图推理结果含 NaN/Inf")
    return {
        "rows": int(out.shape[0]),
        "cols": int(out.shape[1]),
        "labels": (out >= 0).astype(int).tolist(),
        "scores": [[round(float(v), 6) for v in row] for row in out],
    }


def _parse_points(raw: str) -> list[tuple[int, int]]:
    return parse_points_text(raw) if raw and raw.strip() else []


# --------------------------------------------------------------------------
# 路由
# --------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _page()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/grid", response_model=None)
async def grid(image: UploadFile = File(...),
               weights: UploadFile = File(...)) -> JSONResponse:
    try:
        img, wts = _load_inputs(await image.read(), await weights.read())
        meta = _grid_meta(img, wts)
    except (SweepError, ValueError) as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)
    return JSONResponse({"ok": True, **meta})


@app.post("/sweep", response_model=None)
async def sweep(image: UploadFile = File(...),
                weights: UploadFile = File(...),
                points: str = Form(""),
                as_json: str = Form("")) -> HTMLResponse | JSONResponse:
    want_json = as_json == "1"
    image_bytes = await image.read()
    weights_bytes = await weights.read()
    try:
        img, wts = _load_inputs(image_bytes, weights_bytes)
    except (SweepError, ValueError) as exc:
        payload = failure_payload(str(exc))
        if want_json:
            return JSONResponse(payload, status_code=400)
        return HTMLResponse(_page(_render_result(payload)),
                            status_code=400)

    meta = None
    try:
        meta = _grid_meta(img, wts)
    except Exception:  # 页面网格展示失败不应吞掉区间计算
        traceback.print_exc()
        meta = None

    try:
        pts = _parse_points(points)
        payload = analyze_sweep(img, wts, pts)
        status = 200
    except SweepError as exc:
        payload = failure_payload(str(exc), grid=meta and {
            "rows": meta["rows"], "cols": meta["cols"]})
        status = 400
    except Exception as exc:
        traceback.print_exc()
        payload = failure_payload(
            f"计算异常：{type(exc).__name__}: {exc}",
            grid=meta and {"rows": meta["rows"], "cols": meta["cols"]})
        status = 500

    if want_json:
        return JSONResponse(payload, status_code=status)
    try:
        pts = _parse_points(points)
    except SweepError:
        pts = []
    return HTMLResponse(
        _page(_render_result(payload), meta, pts), status_code=status)
