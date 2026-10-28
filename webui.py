#!/usr/bin/env python3
"""
webui.py - live camera web UI: person tracking, hand gestures, held-gesture
alerts, metric distance, 3D point cloud and facial expression.

Toggleable features: distance (metric depth), 3d (point cloud), emotion.

  /              dashboard
  /stream        MJPEG
  /snapshot      annotated JPEG
  /depthmap      metric depth heatmap JPEG
  /pointcloud    binary float32 cloud (x,y,z,r,g,b)
  /measure?x=&y= distance at normalised image coords
  /api/toggle    POST {f: "depth"|"cloud"|"emotion", on: bool}
  /events        JSON state
"""
import argparse
import base64
import json
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from gesture import HandTracker, draw_hands
    HAVE_GESTURES = True
    _GESTURE_ERR = ""
except Exception as _e:
    HAVE_GESTURES = False
    _GESTURE_ERR = str(_e)

try:
    from depth import DepthEngine
    HAVE_DEPTH = True
    _DEPTH_ERR = ""
except Exception as _e:
    HAVE_DEPTH = False
    _DEPTH_ERR = str(_e)

try:
    from emotion import EmotionEngine
    HAVE_EMOTION = True
    _EMOTION_ERR = ""
except Exception as _e:
    HAVE_EMOTION = False
    _EMOTION_ERR = str(_e)

LOCK = threading.Lock()
STATE = {
    "frame": None,
    "raw": None,
    "persons": [],
    "objects": [],
    "hands": [],
    "gestures": deque(maxlen=30),
    "events": deque(maxlen=40),
    "fps": 0.0,
    "last_vlm": 0.0,
    "vlm_latency": 0.0,
    "vlm_model": "",
    "yolo_model": "",
    "gesture_name": "",
    "gesture_progress": 0.0,
    "features": {"depth": False, "cloud": False, "emotion": False, "objects": False},
    "running": True,
    "depth_heat": None,
    "depth_map": None,
    "depth_ms": 0.0,
    "depth_stats": {},
    "emotion": {"face": False, "top": "", "confidence": 0.0, "scores": {}},
    "started": time.time(),
}

HOLD = 1.2


def _make_stop_frame():
    """Return a JPEG used in place of the video while the camera is stopped."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (12, 16, 22)
    cv2.putText(img, "CAMERA STOPPED", (392, 348), cv2.FONT_HERSHEY_SIMPLEX,
                1.5, (110, 120, 132), 3, cv2.LINE_AA)
    cv2.putText(img, "press Start", (560, 404), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (86, 96, 108), 2, cv2.LINE_AA)
    ok, jpg = cv2.imencode(".jpg", img)
    return jpg.tobytes() if ok else b""


STOP_FRAME = _make_stop_frame()

HTML = """
<!doctype html>
<html><head><meta charset="utf-8"><title>cv-lab live</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0a0d12; color:#e6edf3; min-height:100vh;
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { display:flex; align-items:center; gap:12px; padding:12px 20px;
           border-bottom:1px solid #1d2632; background:#0d1218; flex-wrap:wrap; }
  header h1 { font-size:15px; margin:0 8px 0 0; font-weight:600; letter-spacing:.02em; }
  .dot { width:8px; height:8px; border-radius:50%; background:#3fb950;
         box-shadow:0 0 0 3px rgba(63,185,80,.15); animation:pulse 2s infinite; }
  @keyframes pulse { 50% { opacity:.35; } }
  .toggles { display:flex; gap:6px; }
  button.tgl { background:#131a24; color:#9aa7b4; border:1px solid #24303e;
               border-radius:7px; padding:6px 12px; font-size:12.5px; font-weight:500;
               cursor:pointer; transition:all .15s ease; }
  button.tgl:hover { border-color:#33445a; color:#e6edf3; }
  button.tgl.on { background:#123222; border-color:#2e7d52; color:#63e6a0; }
  .meta { margin-left:auto; color:#7d8896; font-size:12px; font-variant-numeric:tabular-nums; }
  .runbar { display:flex; gap:6px; padding-left:10px; border-left:1px solid #24303e; }
  button.tgl.run.on { background:#123222; border-color:#2e7d52; color:#63e6a0; }
  #btn-stop.on { background:#3a1717; border-color:#7d2e2e; color:#ff8f8f; }
  .overlay { position:absolute; inset:0; display:none; align-items:center; justify-content:center;
             background:rgba(6,9,13,.55); color:#8b98a6; font-size:13px;
             letter-spacing:.14em; text-transform:uppercase; }

  /* stats strip above the camera */
  .statbar { display:flex; gap:10px; align-items:stretch; flex-wrap:wrap;
             padding:12px 20px 0; }
  .stat { background:#131a24; border:1px solid #1d2632; border-radius:8px;
          padding:8px 12px; min-width:88px; }
  .stat b { display:block; font-size:20px; font-variant-numeric:tabular-nums; line-height:1.15; }
  .stat span { font-size:10px; color:#7d8896; text-transform:uppercase; letter-spacing:.08em; }
  .gcard { background:#131a24; border:1px solid #1d2632; border-radius:8px;
           padding:8px 12px; flex:1 1 230px; min-width:210px; }
  .grow { display:flex; align-items:baseline; gap:8px; }
  .gname { font-size:16px; font-weight:600; color:#ffd600; text-transform:capitalize; }
  .gsub { margin-left:auto; font-size:11px; color:#7d8896; }
  .gbar { margin-top:7px; height:5px; background:#1c2531; border-radius:3px; overflow:hidden; }
  .gbar i { display:block; height:100%; width:0%; background:#ffd600; transition:width .15s linear; }
  .ecard .etop { font-size:16px; font-weight:600; color:#58a6ff; text-transform:capitalize; }
  .ecard .rows { margin-top:5px; }
  .ecard .row { display:flex; align-items:center; gap:7px; font-size:11px; color:#9aa7b4; margin-top:3px; }
  .ecard .row b { width:66px; color:#e6edf3; font-weight:500; text-transform:capitalize; }
  .ecard .mini { flex:1; height:4px; background:#1c2531; border-radius:2px; overflow:hidden; }
  .ecard .mini i { display:block; height:100%; background:#58a6ff; }
  .oname { font-size:16px; font-weight:600; color:#58a6ff; }
  .olist { font-size:11px; color:#9aa7b4; margin-top:3px; line-height:1.45; max-height:32px; overflow:hidden; }
  .dread { font-size:22px; font-weight:700; color:#ffd600; font-variant-numeric:tabular-nums; line-height:1.2; }
  .hidden { display:none !important; }

  main { padding:12px 20px 20px; display:flex; flex-direction:column; gap:14px; }
  .card { background:#0f151d; border:1px solid #1d2632; border-radius:10px; overflow:hidden; }

  /* centred camera, narrower than the page */
  .camera { max-width:900px; width:100%; margin:0 auto; }
  .stage { position:relative; background:#000; line-height:0; cursor:crosshair; }
  .stage img, .stage canvas { width:100%; height:auto; display:block; }
  #three { aspect-ratio:16/9; height:auto; display:none; cursor:grab; }
  .badge { position:absolute; top:10px; left:10px; background:rgba(0,0,0,.62);
           border:1px solid #2a3543; border-radius:6px; padding:4px 9px;
           font-size:11px; color:#c9d1d9; line-height:1.4; letter-spacing:.03em; }
  #heat { position:absolute; bottom:10px; right:10px; width:20%; border:1px solid #2a3543;
          border-radius:6px; display:none; opacity:.92; }
  #marker { position:absolute; border:2px solid #ffd600; border-radius:50%; width:26px; height:26px;
            margin:-13px 0 0 -13px; display:none; pointer-events:none;
            box-shadow:0 0 0 1px rgba(0,0,0,.6); }
  #mlabel { position:absolute; background:#ffd600; color:#101418; font-weight:700; font-size:12px;
            padding:2px 7px; border-radius:5px; margin:16px 0 0 16px; display:none;
            pointer-events:none; font-variant-numeric:tabular-nums; }

  /* two feeds under the camera */
  .lists { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:820px) { .lists { grid-template-columns:1fr; } }
  .list { display:flex; flex-direction:column; }
  .lhead { display:flex; justify-content:space-between; align-items:baseline;
           padding:10px 14px; border-bottom:1px solid #1d2632;
           font-size:11px; text-transform:uppercase; letter-spacing:.09em; color:#7d8896; }
  .lhead em { font-style:normal; font-size:10px; color:#4d5866; }
  .list ul { list-style:none; margin:0; padding:0; max-height:330px; min-height:248px; overflow-y:auto; }
  .list li { padding:10px 14px; border-bottom:1px solid #18202b; position:relative; }
  .list li:last-child { border-bottom:0; }
  .list li .lt { font-size:10.5px; color:#5c6773; letter-spacing:.04em;
                 font-variant-numeric:tabular-nums; margin-bottom:2px; }
  .list li .lx { color:#c9d1d9; }
  .list li.latest { background:linear-gradient(90deg, rgba(63,185,80,.13), rgba(63,185,80,0) 70%);
                    border-left:3px solid #3fb950; padding-left:11px; }
  .list li.latest .lx { color:#ffffff; font-weight:500; }
  .list li.latest .lt { color:#3fb950; }
  .list.g li.latest { background:linear-gradient(90deg, rgba(255,214,0,.13), rgba(255,214,0,0) 70%);
                      border-left-color:#ffd600; }
  .list.g li.latest .lt { color:#ffd600; }
  .empty { color:#5c6773; font-style:italic; padding:12px 14px; }
  footer { color:#4d5866; font-size:11px; padding:0 20px 18px; }
</style></head>
<body>
<header>
  <div class="dot"></div>
  <h1>cv-lab</h1>
  <div class="toggles">
    <button class="tgl" id="btn-depth" data-f="depth" title="Metric distance to any point">Distance</button>
    <button class="tgl" id="btn-cloud" data-f="cloud" title="Live 3D point cloud">3D view</button>
    <button class="tgl" id="btn-emotion" data-f="emotion" title="Facial expression estimate">Emotions</button>
    <button class="tgl" id="btn-objects" data-f="objects" title="Detect all 80 COCO classes">Objects</button>
  </div>
  <div class="runbar">
    <button class="tgl run" id="btn-start" title="Open the camera and resume">Start</button>
    <button class="tgl run" id="btn-stop" title="Stop and release the camera">Stop</button>
  </div>
  <div class="meta" id="meta">connecting...</div>
</header>

<div class="statbar">
  <div class="stat"><b id="fps">0</b><span>fps</span></div>
  <div class="stat"><b id="people">0</b><span>people</span></div>
  <div class="stat"><b id="lat">-</b><span>vlm s</span></div>
  <div class="gcard">
    <div class="grow">
      <span class="gname" id="gname">no hand</span>
      <span class="gsub" id="gsub">hold to trigger</span>
    </div>
    <div class="gbar"><i id="gbar"></i></div>
  </div>
  <div class="gcard ecard hidden" id="ecard-wrap">
    <div class="grow"><span class="etop" id="etop">no face</span><span class="gsub">expression</span></div>
    <div class="rows" id="escores"></div>
  </div>
  <div class="gcard hidden" id="ocard-wrap">
    <div class="grow"><span class="oname" id="oname">0 objects</span><span class="gsub" id="osub">yolo11n / 80 classes</span></div>
    <div class="olist" id="olist"></div>
  </div>
  <div class="gcard hidden" id="dcard-wrap">
    <div class="grow"><span class="dread" id="dread">-- m</span><span class="gsub" id="dsub">click the video</span></div>
  </div>
</div>

<main>
  <section class="card camera">
    <div class="stage" id="stage">
      <div class="badge" id="badge">YOLO11 + ByteTrack</div>
      <img id="vid" src="/stream" alt="live camera">
      <canvas id="three"></canvas>
      <img id="heat" alt="depth">
      <div id="marker"></div>
      <div id="mlabel"></div>
      <div id="stopped" class="overlay"><span>camera stopped</span></div>
    </div>
  </section>

  <section class="lists">
    <div class="card list g">
      <div class="lhead"><span>Triggered gestures</span><em id="gcount">0</em></div>
      <ul id="gestures"><li class="empty">none yet</li></ul>
    </div>
    <div class="card list">
      <div class="lhead"><span>What is happening</span><em id="ecount">0</em></div>
      <ul id="events"><li class="empty">waiting for the first description...</li></ul>
    </div>
  </section>
</main>
<footer>local only &middot; nothing leaves this machine</footer>
<script>
var feat = { depth: false, cloud: false, emotion: false, objects: false };
var video = document.getElementById("vid");
var heat = document.getElementById("heat");
var three = document.getElementById("three");
var cardE = document.getElementById("ecard-wrap");
var cardD = document.getElementById("dcard-wrap");
var marker = document.getElementById("marker");
var mlabel = document.getElementById("mlabel");

function esc(s) {
  return String(s).replace(/[&<>]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c];
  });
}

function setFeature(f, on) {
  feat[f] = on;
  var btn = document.getElementById("btn-" + f);
  if (btn) { btn.classList.toggle("on", on); }
  if (f === "emotion") { cardE.classList.toggle("hidden", !on); }
  if (f === "objects") { document.getElementById("ocard-wrap").classList.toggle("hidden", !on); }
  if (f === "depth") {
    cardD.classList.toggle("hidden", !on);
    heat.style.display = on ? "block" : "none";
    if (on) { heat.src = "/depthmap?t=" + Date.now(); }
  }
  if (f === "cloud") {
    three.style.display = on ? "block" : "none";
    video.style.display = on ? "none" : "block";
    if (on) { sizeCanvas(); }
  }
  fetch("/api/toggle", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ f: f, on: on })
  });
}

Array.prototype.forEach.call(document.querySelectorAll("button.tgl"), function (b) {
  b.addEventListener("click", function () { setFeature(b.dataset.f, !feat[b.dataset.f]); });
});

var q = new URLSearchParams(location.search).get("on");
if (q) { q.split(",").forEach(function (f) { if (feat.hasOwnProperty(f)) { setFeature(f, true); } }); }

document.getElementById("stage").addEventListener("click", function (ev) {
  if (!feat.depth || feat.cloud) { return; }
  var r = video.getBoundingClientRect();
  if (ev.clientX < r.left || ev.clientX > r.right || ev.clientY < r.top || ev.clientY > r.bottom) { return; }
  var nx = (ev.clientX - r.left) / r.width;
  var ny = (ev.clientY - r.top) / r.height;
  fetch("/measure?x=" + nx.toFixed(4) + "&y=" + ny.toFixed(4))
    .then(function (res) { return res.json(); })
    .then(function (d) {
      if (!d.ok) { return; }
      document.getElementById("dread").textContent = d.metres.toFixed(2) + " m";
      var st = document.getElementById("stage").getBoundingClientRect();
      marker.style.display = "block";
      mlabel.style.display = "block";
      marker.style.left = (r.left - st.left + nx * r.width) + "px";
      marker.style.top = (r.top - st.top + ny * r.height) + "px";
      mlabel.textContent = d.metres.toFixed(2) + " m";
      mlabel.style.left = marker.style.left;
      mlabel.style.top = marker.style.top;
      document.getElementById("dsub").textContent = "at " + Math.round(nx * 100) + "%, " + Math.round(ny * 100) + "%";
    });
});

function setRunning(on) {
  fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ on: on })
  });
  document.getElementById("btn-start").classList.toggle("on", on);
  document.getElementById("btn-stop").classList.toggle("on", !on);
  document.getElementById("stopped").style.display = on ? "none" : "flex";
}

document.getElementById("btn-start").addEventListener("click", function () { setRunning(true); });
document.getElementById("btn-stop").addEventListener("click", function () { setRunning(false); });

var sigG = "", sigE = "";

function updateList(id, items, html, sigRef) {
  var ul = document.getElementById(id);
  var sig = items.length + "|" + (items.length ? (items[0].ts + items[0].txt) : "");
  if (sig === sigRef.value) { return; }
  sigRef.value = sig;
  var top = ul.scrollTop;
  ul.innerHTML = html;
  ul.scrollTop = top;
}

async function tick() {
  try {
    var r = await fetch("/events", { cache: "no-store" });
    var d = await r.json();
    document.getElementById("fps").textContent = Math.round(d.stats.fps);
    document.getElementById("people").textContent = d.stats.people;
    document.getElementById("lat").textContent = d.stats.vlm_latency ? d.stats.vlm_latency.toFixed(1) : "-";
    document.getElementById("meta").textContent = d.stats.yolo_model +
      (d.stats.running ? "" : " | stopped") + " | " + d.stats.uptime;
    document.getElementById("btn-start").classList.toggle("on", d.stats.running);
    document.getElementById("btn-stop").classList.toggle("on", !d.stats.running);
    document.getElementById("stopped").style.display = d.stats.running ? "none" : "flex";
    var ids = d.stats.ids.length ? "tracking id " + d.stats.ids.join(", ") : "no person locked";
    document.getElementById("badge").textContent = "YOLO11 + ByteTrack | " + ids;

    var g = d.stats.gesture;
    document.getElementById("gname").textContent = g.name ? g.name.replace(/_/g, " ") : "no hand";
    document.getElementById("gbar").style.width = Math.round(g.progress * 100) + "%";
    document.getElementById("gsub").textContent = g.name
      ? (g.progress >= 1 ? "triggered" : Math.round(g.progress * 100) + "% of " + g.hold + "s")
      : "hold to trigger";

    if (feat.emotion) {
      var e = d.stats.emotion;
      document.getElementById("etop").textContent = e.face ? e.top : "no face";
      var keys = Object.keys(e.scores || {}).sort(function (a, b) { return e.scores[b] - e.scores[a]; }).slice(0, 2);
      document.getElementById("escores").innerHTML = keys.map(function (k) {
        return '<div class="row"><b>' + esc(k) + '</b><span class="mini"><i style="width:' +
               Math.round(e.scores[k] * 100) + '%"></i></span></div>';
      }).join("");
    }
    if (feat.objects) {
      var objs = d.stats.objects || [];
      document.getElementById("oname").textContent = objs.length + (objs.length === 1 ? " object" : " objects");
      var counts = {};
      objs.forEach(function (o) { counts[o.name] = (counts[o.name] || 0) + 1; });
      document.getElementById("olist").textContent = Object.keys(counts).map(function (k) {
        return k + " x" + counts[k];
      }).join("  ");
    }
    if (feat.depth) {
      var ds = d.stats.depth || {};
      if (ds.median) {
        document.getElementById("dsub").textContent = "median " + ds.median.toFixed(1) + " m | " + Math.round(ds.ms) + " ms";
      }
    }

    var gItems = d.gestures.map(function (e) { return { ts: e.ts, txt: e.gesture }; });
    var gHtml = gItems.length === 0 ? '<li class="empty">none yet</li>' :
      gItems.map(function (e, i) {
        return '<li class="' + (i === 0 ? "latest" : "") + '"><div class="lt">' + esc(e.ts) +
               '</div><div class="lx">' + esc(e.txt.replace(/_/g, " ")) + "</div></li>";
      }).join("");
    document.getElementById("gcount").textContent = gItems.length;
    updateList("gestures", gItems, gHtml, { value: sigG });

    var eItems = d.events.map(function (e) { return { ts: e.ts, txt: e.text, lat: e.latency }; });
    var eHtml = eItems.length === 0 ? '<li class="empty">waiting for the first description...</li>' :
      eItems.map(function (e, i) {
        return '<li class="' + (i === 0 ? "latest" : "") + '"><div class="lt">' + esc(e.ts) +
               " &middot; " + e.lat + 's</div><div class="lx">' + esc(e.txt) + "</div></li>";
      }).join("");
    document.getElementById("ecount").textContent = eItems.length;
    updateList("events", eItems, eHtml, { value: sigE });
  } catch (err) {
    document.getElementById("meta").textContent = "disconnected";
  }
}
setInterval(tick, 1200);
tick();

/* 3D point cloud viewer: plain canvas, no dependencies */
var pts = new Float32Array(0);
var yaw = 0.35, pitch = 0.12, camDist = 3.2;
var dragging = false, lastX = 0, lastY = 0;
var ctx = three.getContext("2d");

function sizeCanvas() {
  three.width = three.clientWidth || 900;
  three.height = Math.round(three.width * 9 / 16);
}

window.addEventListener("resize", function () { if (feat.cloud) { sizeCanvas(); } });

three.addEventListener("mousedown", function (e) { dragging = true; lastX = e.clientX; lastY = e.clientY; });
window.addEventListener("mouseup", function () { dragging = false; });
window.addEventListener("mousemove", function (e) {
  if (!dragging || !feat.cloud) { return; }
  yaw += (e.clientX - lastX) * 0.008;
  pitch += (e.clientY - lastY) * 0.006;
  pitch = Math.max(-1.2, Math.min(1.2, pitch));
  lastX = e.clientX; lastY = e.clientY;
});
three.addEventListener("wheel", function (e) {
  if (!feat.cloud) { return; }
  e.preventDefault();
  camDist = Math.max(0.8, Math.min(12, camDist + e.deltaY * 0.002));
}, { passive: false });

async function fetchCloud() {
  if (!feat.cloud) { return; }
  try {
    var r = await fetch("/pointcloud?cols=88&rows=66", { cache: "no-store" });
    var buf = await r.arrayBuffer();
    pts = new Float32Array(buf);
    drawFrame();
  } catch (err) {}
}

function render() {
  requestAnimationFrame(render);
  if (feat.cloud) { drawFrame(); }
}

function drawFrame() {
  if (!three.width) { sizeCanvas(); }
  var W = three.width, H = three.height;
  ctx.fillStyle = "#05080d";
  ctx.fillRect(0, 0, W, H);
  var cx = W / 2, cy = H / 2, f = H * 0.95;
  var cY = Math.cos(yaw), sY = Math.sin(yaw), cP = Math.cos(pitch), sP = Math.sin(pitch);
  for (var i = 0; i < pts.length; i += 6) {
    var x = pts[i], y = pts[i + 1], z = pts[i + 2];
    var x1 = x * cY - z * sY;
    var z1 = x * sY + z * cY;
    var y1 = y * cP - z1 * sP;
    var z2 = y * sP + z1 * cP;
    var zz = z2 + camDist;
    if (zz < 0.15) { continue; }
    var sx = cx + f * x1 / zz;
    var sy = cy + f * y1 / zz;
    if (sx < -20 || sx > W + 20 || sy < -20 || sy > H + 20) { continue; }
    var rr = (pts[i + 3] * 255) | 0, gg = (pts[i + 4] * 255) | 0, bb = (pts[i + 5] * 255) | 0;
    ctx.fillStyle = "rgb(" + rr + "," + gg + "," + bb + ")";
    var s = Math.max(1, 4.2 / zz);
    ctx.fillRect(sx, sy, s, s);
  }
  ctx.fillStyle = "rgba(230,237,243,.55)";
  ctx.font = "11px ui-sans-serif,system-ui";
  ctx.fillText("drag to orbit | scroll to zoom | " + (pts.length / 6 | 0) + " points", 12, H - 12);
}
render();
setInterval(fetchCloud, 450);
</script>
</body></html>
"""


def describe(frame, model, prompt, host, timeout=120):
    """Send one frame to the local vision model; return (text, latency_seconds)."""
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise RuntimeError("encode failed")
    img = base64.b64encode(buf.tobytes()).decode()
    body = json.dumps({"model": model, "prompt": prompt, "images": [img], "stream": False,
                       "options": {"temperature": 0.2, "num_predict": 90}}).encode()
    req = urllib.request.Request(host.rstrip("/") + "/api/generate",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data.get("response", "").strip(), time.time() - t0


def add_event(text, latency):
    """Prepend a narration line to the shared feed and stamp the VLM time."""
    with LOCK:
        STATE["events"].appendleft({"ts": time.strftime("%H:%M:%S"),
                                    "text": text, "latency": round(latency, 1)})
        STATE["last_vlm"] = time.time()
        STATE["vlm_latency"] = latency


def add_gesture(name, handedness):
    """Record one completed held-gesture event."""
    with LOCK:
        STATE["gestures"].appendleft({"ts": time.strftime("%H:%M:%S"),
                                      "gesture": name, "hand": handedness})
    print("[%s] gesture: %s (%s)" % (time.strftime("%H:%M:%S"), name, handedness), flush=True)


def vision_loop(args, emotion_engine):
    """Capture frames; run tracking, hands and expression; publish the annotated JPEG."""
    model = YOLO(args.yolo)
    tracker = None
    if args.gestures and HAVE_GESTURES:
        tracker = HandTracker(args.hand_model, max_hands=2, hold=args.hold)
        print("gestures: on (hold %.1fs)" % args.hold, flush=True)
    elif args.gestures:
        print("gestures: unavailable -", _GESTURE_ERR, flush=True)
    with LOCK:
        STATE["yolo_model"] = args.yolo

    cap = None
    fps = 0.0
    n = 0
    while True:
        with LOCK:
            running = STATE["running"]
        if not running:
            if cap is not None:
                cap.release()
                cap = None
                if tracker is not None:
                    tracker.reset()
                with LOCK:
                    STATE["frame"] = STOP_FRAME
                    STATE["persons"] = []
                    STATE["objects"] = []
                    STATE["hands"] = []
                    STATE["gesture_name"] = ""
                    STATE["gesture_progress"] = 0.0
                    STATE["depth_map"] = None
                    STATE["depth_heat"] = None
                    STATE["depth_stats"] = {}
                    STATE["emotion"] = {"face": False, "top": "", "confidence": 0.0, "scores": {}}
                print("[%s] camera stopped and released" % time.strftime("%H:%M:%S"), flush=True)
            time.sleep(0.2)
            continue
        if cap is None:
            cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
            if not cap.isOpened():
                print("camera busy, retrying...", flush=True)
                cap = None
                time.sleep(2)
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print("[%s] camera started" % time.strftime("%H:%M:%S"), flush=True)
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.05)
            continue
        n += 1
        # keep an overlay-free copy for the models; every third frame is enough
        clean = frame.copy() if (n % 3 == 0) else None
        t0 = time.time()
        with LOCK:
            show_all = STATE["features"]["objects"]
        # persist=True keeps track ids stable between calls (ByteTrack)
        result = model.track(frame, persist=True,
                             classes=(None if show_all else [0]), conf=0.35,
                             imgsz=640, verbose=False)[0]
        dt = time.time() - t0
        fps = (1.0 / dt) if fps == 0 else fps * 0.9 + (1.0 / dt) * 0.1

        persons = []
        objects = []
        if result.boxes is not None and len(result.boxes):
            for b in result.boxes:
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                conf = float(b.conf)
                cls = int(b.cls)
                name = result.names[cls]
                tid = int(b.id) if b.id is not None else -1
                objects.append({"name": name, "conf": round(conf, 2), "id": tid})
                if cls == 0:
                    persons.append({"id": tid, "conf": round(conf, 2)})
                    colour = (0, 214, 120)
                    label = ("id %d  %.0f%%" % (tid, conf * 100)) if tid >= 0 else ("%.0f%%" % (conf * 100))
                else:
                    colour = (255, 166, 88)
                    label = ("%s %.0f%%" % (name, conf * 100))
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), colour, -1)
                cv2.putText(frame, label, (x1 + 5, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (8, 12, 14), 1, cv2.LINE_AA)

        hands = []
        if tracker is not None and (n % max(1, args.gesture_every) == 0):
            sw = args.hand_scale
            small = cv2.resize(frame, None, fx=sw, fy=sw) if sw != 1.0 else frame
            hands = tracker.detect(small, time.time())
            for hd in hands:
                hd["landmarks"] = [(x / sw, y / sw) for x, y in hd["landmarks"]]
                if hd["fired"]:
                    add_gesture(hd["gesture"], hd["handedness"])
            draw_hands(frame, hands)

        if emotion_engine is not None and STATE["features"]["emotion"] and clean is not None:
            try:
                emo = emotion_engine.detect(clean)
                with LOCK:
                    STATE["emotion"] = emo
                if emo["face"] and emo["box"]:
                    x1, y1, x2, y2 = emo["box"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (88, 166, 255), 1)
                    cv2.putText(frame, emo["top"], (x1 + 2, max(14, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (88, 166, 255), 1, cv2.LINE_AA)
            except Exception as e:
                print("emotion error:", e, flush=True)

        with LOCK:
            hands_out = list(hands)
            STATE["hands"] = hands_out
            best = max(hands_out, key=lambda h: h["progress"]) if hands_out else None
            STATE["gesture_name"] = best["gesture"] if best else ""
            STATE["gesture_progress"] = best["progress"] if best else 0.0

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if ok:
            with LOCK:
                STATE["frame"] = jpg.tobytes()
                STATE["persons"] = persons
                STATE["objects"] = objects
                STATE["fps"] = fps
                if clean is not None:
                    STATE["raw"] = clean


def depth_loop(args):
    """Compute metric depth while a depth-consuming feature is on, then go idle."""
    engine = None
    last = 0.0
    failed = 0
    while True:
        time.sleep(0.08)
        with LOCK:
            want = (STATE["features"]["depth"] or STATE["features"]["cloud"]) and STATE["running"]
            raw = STATE.get("raw")
        if not want or raw is None:
            continue
        now = time.time()
        # depth costs ~300 ms; the UI only needs a few updates per second
        if (now - last) < args.depth_interval:
            continue
        if engine is None:
            if not HAVE_DEPTH:
                print("depth unavailable:", _DEPTH_ERR, flush=True)
                time.sleep(5)
                continue
            try:
                t0 = time.time()
                engine = DepthEngine()
                print("depth engine ready in %.1fs" % (time.time() - t0), flush=True)
            except Exception as e:
                print("depth load error:", e, flush=True)
                time.sleep(5)
                continue
        try:
            t0 = time.time()
            d = engine.estimate(raw)
            ms = (time.time() - t0) * 1000.0
            heat = DepthEngine.colourise(d)
            ok, jpg = cv2.imencode(".jpg", heat, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            with LOCK:
                STATE["depth_map"] = d
                STATE["depth_ms"] = ms
                STATE["depth_stats"] = {"min": float(d.min()), "median": float(np.median(d)),
                                        "max": float(d.max()), "ms": round(ms, 1)}
                if ok:
                    STATE["depth_heat"] = jpg.tobytes()
            last = now
            failed = 0
        except Exception as e:
            failed += 1
            print("depth error:", e, flush=True)
            if failed > 3:
                time.sleep(3)
            last = now


def vlm_loop(args):
    """Narrate the scene on change; stays quiet while depth owns the GPU."""
    with LOCK:
        STATE["vlm_model"] = args.vlm if not args.no_vlm else ""
    if args.no_vlm:
        return
    prev_small = None
    while True:
        time.sleep(0.5)
        with LOCK:
            raw = STATE.get("raw")
            persons = list(STATE.get("persons", []))
            last = STATE.get("last_vlm", 0.0)
            # one GPU lane: depth work and a stopped camera both silence the VLM
            busy = STATE["features"]["depth"] or STATE["features"]["cloud"] or not STATE["running"]
        if raw is None or not persons or busy:
            continue
        if (time.time() - last) < args.interval:
            continue
        # motion gate: skip the model when the downscaled frame barely changed
        small = cv2.cvtColor(cv2.resize(raw, (64, 48)), cv2.COLOR_BGR2GRAY)
        diff = 1e9 if prev_small is None else float(
            abs(small.astype("int16") - prev_small.astype("int16")).mean())
        if diff < args.threshold:
            continue
        prev_small = small
        try:
            text, lat = describe(raw, args.vlm, args.prompt, args.host)
            if text:
                add_event(text, lat)
                print("[%s] %.1fs %s" % (time.strftime("%H:%M:%S"), lat, text), flush=True)
        except urllib.error.URLError as e:
            print("ollama error:", e, flush=True)
            time.sleep(5)
        except Exception as e:
            print("vlm error:", e, flush=True)


class Handler(BaseHTTPRequestHandler):
    """Serve the page, the MJPEG stream, depth data and JSON state."""
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        """Suppress the default per-request stderr logging."""
        pass

    def _send(self, code, ctype, body):
        """Write a complete response with the headers this UI relies on."""
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """Handle feature toggles and start/stop requests."""
        path = self.path.split("?")[0]
        if path not in ("/api/toggle", "/api/run"):
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            body = {}
        if path == "/api/run":
            on = bool(body.get("on"))
            with LOCK:
                STATE["running"] = on
            print("[%s] run=%s" % (time.strftime("%H:%M:%S"), on), flush=True)
            self._send(200, "application/json", json.dumps({"running": on}).encode())
            return
        f = body.get("f", "")
        on = bool(body.get("on"))
        with LOCK:
            if f in STATE["features"]:
                STATE["features"][f] = on
                snap = dict(STATE["features"])
            else:
                snap = dict(STATE["features"])
        print("[%s] toggle %s=%s" % (time.strftime("%H:%M:%S"), f, on), flush=True)
        self._send(200, "application/json", json.dumps({"features": snap}).encode())

    def do_GET(self):
        """Route every read-only endpoint by path."""
        path = self.path.split("?")[0]
        query = {}
        if "?" in self.path:
            for part in self.path.split("?", 1)[1].split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    query[k] = v

        if path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", HTML.encode())
        elif path == "/events":
            with LOCK:
                persons = list(STATE["persons"])
                hands = list(STATE["hands"])
                payload = {
                    "events": list(STATE["events"]),
                    "gestures": list(STATE["gestures"]),
                    "objects": list(STATE["objects"]),
                    "stats": {
                        "fps": round(STATE["fps"], 1),
                        "people": len(persons),
                        "ids": sorted({p["id"] for p in persons if p["id"] >= 0}),
                        "hands": len(hands),
                        "gesture": {"name": STATE["gesture_name"],
                                    "progress": STATE["gesture_progress"], "hold": HOLD},
                        "emotion": STATE["emotion"],
                        "depth": STATE["depth_stats"],
                        "features": dict(STATE["features"]),
                        "running": STATE["running"],
                        "vlm_latency": round(STATE["vlm_latency"], 2),
                        "vlm_model": STATE["vlm_model"],
                        "yolo_model": STATE["yolo_model"],
                        "uptime": str(int(time.time() - STATE["started"])) + "s",
                    },
                }
            self._send(200, "application/json", json.dumps(payload).encode())
        elif path == "/measure":
            with LOCK:
                d = STATE.get("depth_map")
            if d is None:
                self._send(200, "application/json", json.dumps({"ok": False, "reason": "depth not ready"}).encode())
                return
            try:
                nx = float(query.get("x", 0.5))
                ny = float(query.get("y", 0.5))
            except ValueError:
                nx, ny = 0.5, 0.5
            h, w = d.shape[:2]
            metres = DepthEngine.distance_at(d, nx * w, ny * h)
            self._send(200, "application/json",
                       json.dumps({"ok": True, "metres": round(metres, 2),
                                   "x": round(nx, 3), "y": round(ny, 3)}).encode())
        elif path == "/depthmap":
            with LOCK:
                data = STATE.get("depth_heat")
            if not data:
                self._send(503, "text/plain", b"depth not ready")
            else:
                self._send(200, "image/jpeg", data)
        elif path == "/pointcloud":
            with LOCK:
                d = STATE.get("depth_map")
                raw = STATE.get("raw")
            if d is None or raw is None:
                self._send(200, "application/octet-stream", b"")
                return
            try:
                cols = max(16, min(160, int(query.get("cols", 88))))
                rows = max(12, min(120, int(query.get("rows", 66))))
            except ValueError:
                cols, rows = 88, 66
            arr = DepthEngine.pointcloud(d, raw, rows, cols)
            self._send(200, "application/octet-stream", arr.astype(np.float32).tobytes())
        elif path == "/snapshot":
            with LOCK:
                data = STATE.get("frame")
            if not data:
                self._send(503, "text/plain", b"no frame yet")
            else:
                self._send(200, "image/jpeg", data)
        elif path == "/health":
            self._send(200, "text/plain", b"ok")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = None
            try:
                while True:
                    with LOCK:
                        data = STATE.get("frame")
                    if data and data is not last:
                        last = data
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(("Content-Length: %d\r\n\r\n" % len(data)).encode())
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                        self.wfile.flush()
                    time.sleep(0.02)
            except (BrokenPipeError, ConnectionResetError, OSError):
                return
        else:
            self._send(404, "text/plain", b"not found")


def main():
    """Parse options, start the workers and serve until interrupted."""
    global HOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--yolo", default="yolo11n.pt")
    ap.add_argument("--vlm", default="qwen2.5vl:3b")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--no-gestures", dest="gestures", action="store_false")
    ap.add_argument("--hand-model", default="models/hand_landmarker.task")
    ap.add_argument("--face-model", default="models/face_landmarker.task")
    ap.add_argument("--hold", type=float, default=1.2)
    ap.add_argument("--gesture-every", type=int, default=2)
    ap.add_argument("--hand-scale", type=float, default=0.6)
    ap.add_argument("--depth-interval", type=float, default=0.6)
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--threshold", type=float, default=1.2)
    ap.add_argument("--prompt", default="Describe what is happening in this image in one short sentence. Focus on the person and their actions.")
    ap.add_argument("--host-url", default="http://127.0.0.1:11434", dest="host")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--bind", default="127.0.0.1")
    args = ap.parse_args()
    HOLD = args.hold

    emotion_engine = None
    if HAVE_EMOTION:
        try:
            emotion_engine = EmotionEngine(args.face_model)
            print("emotion engine ready", flush=True)
        except Exception as e:
            print("emotion unavailable:", e, flush=True)

    threading.Thread(target=vision_loop, args=(args, emotion_engine), daemon=True).start()
    threading.Thread(target=depth_loop, args=(args,), daemon=True).start()
    threading.Thread(target=vlm_loop, args=(args,), daemon=True).start()
    srv = ThreadingHTTPServer((args.bind, args.port), Handler)
    srv.daemon_threads = True
    print("serving on http://%s:%d" % (args.bind, args.port), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("bye", flush=True)


if __name__ == "__main__":
    main()
