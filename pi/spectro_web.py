"""
SpikerBox 即時頻譜圖 Web 檢視器（不含 AI / 不含水泵）
====================================================
目的：純粹確認「SpikerBox → 錄音 → 頻譜圖 → 瀏覽器」這條路通不通。
不需要訓練好的模型，也不碰 GPIO。

啟動：
    python -m pi.spectro_web

然後在同一個 Wi-Fi 下用手機或電腦打開：
    http://<這台電腦或樹莓派的IP>:5000

畫面上會每約 1.5 秒更新一次：上面是波形，下面是頻譜圖(dB)。
頁面頂端會顯示目前模式：
    audio  = 有抓到 USB 音訊裝置(正常)
    serial = 走序列埠的新版 SpikerBox
    mock   = ⚠️ 沒抓到硬體，正在放模擬音檔(代表偵測失敗，去看 test_spikerbox)

（儀表板版的即時頻譜在 pi/web_controller.py，兩者共用 pi/spectro_render.py 的繪圖函式）
"""

import os
import sys
import json
import time
import threading
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from flask import Flask, jsonify

from pi.audio_capture import AudioCapture
from pi.spectro_render import (
    wav_to_mono_float,
    stats_text,
    render_waveform_spectrogram_b64,
)


# ─── 設定 ───────────────────────────────────────────
DURATION_SECONDS = 2.0     # 每段擷取長度（頻譜圖看 2 秒比較清楚）
REFRESH_SECONDS = 1.5      # 每隔多久擷取並更新一次
SPECTRO_MAX_FREQ = 500     # 頻譜圖 y 軸上限(Hz)；植物電位訊號都在低頻，設 None 看全頻寬
PORT = 5000


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode("ascii"))


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


CFG = load_config()
SB = CFG.get("spikerbox", {})
SAMPLE_RATE = SB.get("sample_rate", CFG["audio"]["sample_rate"])
DEVICE_KEYWORDS = SB.get("device_keywords", ["SpikerBox", "Backyard", "USB Audio"])
FALLBACK_DEVICE = SB.get("fallback_device")

app = Flask(__name__)

_latest = {
    "png": None,
    "ts": None,
    "mode": "starting",
    "device": "",
    "stats_text": "",
}
_lock = threading.Lock()


# ─── 背景擷取迴圈 ──────────────────────────────────
def capture_loop():
    audio = AudioCapture(
        sample_rate=SAMPLE_RATE,
        duration=DURATION_SECONDS,
        device_keywords=DEVICE_KEYWORDS,
        fallback_device=FALLBACK_DEVICE,
    )
    with _lock:
        _latest["mode"] = audio.mode
        _latest["device"] = audio.device_name

    tmp_wav = os.path.join(os.path.dirname(__file__), "live_spectro.wav")

    while True:
        try:
            ok = audio.record_audio(tmp_wav)
            if ok and os.path.isfile(tmp_wav):
                x, sr = wav_to_mono_float(tmp_wav)
                png = render_waveform_spectrogram_b64(x, sr, max_freq=SPECTRO_MAX_FREQ)
                stats = stats_text(x, sr)
                with _lock:
                    _latest["png"] = png
                    _latest["ts"] = datetime.now().strftime("%H:%M:%S")
                    _latest["mode"] = audio.mode
                    _latest["device"] = audio.device_name
                    _latest["stats_text"] = stats
        except Exception as e:
            _safe_print(f"[SPECTRO] 擷取失敗: {e}")
        time.sleep(REFRESH_SECONDS)


# ─── 路由 ─────────────────────────────────────────
PAGE = """<!doctype html><html lang="zh-Hant"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SpikerBox 即時頻譜</title>
<style>
 body{font-family:system-ui,sans-serif;background:#111;color:#eee;text-align:center;margin:0;padding:16px}
 h2{margin:.2rem 0 .6rem}
 #mode{display:inline-block;padding:.25rem .7rem;border-radius:6px;background:#333;font-size:14px}
 #mode.mock{background:#7a2;color:#111;font-weight:bold}
 #mode.ok{background:#264}
 img{max-width:100%;border-radius:10px;margin-top:10px;background:#000}
 #stats{font-family:ui-monospace,monospace;font-size:13px;color:#8f8;margin-top:8px;word-break:break-all}
 .hint{color:#888;font-size:12px;margin-top:6px}
</style></head><body>
<h2>🍅 SpikerBox 即時頻譜圖</h2>
<div id="mode">連線中…</div>
<div><img id="spec" alt="等待第一張頻譜圖…"></div>
<div id="stats"></div>
<div class="hint">每約 1.5 秒更新一次。mock = 沒抓到硬體，請跑 python -m pi.test_spikerbox 檢查。</div>
<script>
async function tick(){
  try{
    const r = await fetch('/api/spectro',{cache:'no-store'});
    const d = await r.json();
    const m = document.getElementById('mode');
    m.textContent = '模式: ' + d.mode + '　裝置: ' + (d.device||'-') + '　更新: ' + (d.ts||'-');
    m.className = (d.mode === 'mock') ? 'mock' : 'ok';
    if(d.png) document.getElementById('spec').src = 'data:image/png;base64,' + d.png;
    document.getElementById('stats').textContent = d.stats_text || '';
  }catch(e){
    document.getElementById('mode').textContent = '連線失敗: ' + e;
  }
}
setInterval(tick, 1000); tick();
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


@app.route("/api/spectro")
def api_spectro():
    with _lock:
        return jsonify(dict(_latest))


def main():
    _safe_print("=" * 55)
    _safe_print("[SPECTRO-WEB] SpikerBox 即時頻譜圖")
    _safe_print("=" * 55)
    _safe_print(f"  取樣率      : {SAMPLE_RATE} Hz")
    _safe_print(f"  擷取長度    : {DURATION_SECONDS}s")
    _safe_print(f"  更新間隔    : {REFRESH_SECONDS}s")
    _safe_print(f"  頻譜上限    : {SPECTRO_MAX_FREQ} Hz")
    _safe_print(f"  網址        : http://0.0.0.0:{PORT}")
    _safe_print("=" * 55)
    _safe_print("  手機/電腦打開 http://<這台的IP>:5000　Ctrl+C 停止\n")

    threading.Thread(target=capture_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)


if __name__ == "__main__":
    main()
