"""
番茄灌溉系統 — Flask Web 遠端控制面板（整合版）
=============================================
整合了：水泵控制、自動監測、即時頻譜圖、AI 推論、資料集管理。

啟動方式：
    python -m pi.web_controller

然後在同一個 Wi-Fi 下的任何裝置打開瀏覽器，輸入：
    http://<樹莓派IP>:5000
"""

import os
import sys
import json
import time
import base64
import threading
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, jsonify, request
from pi.gpio_controller import GPIOController
from pi.audio_capture import AudioCapture
from pi.dataset_manager import DatasetManager
from pi.spectro_render import (
    wav_to_mono_float,
    stats_text,
    render_waveform_spectrogram_b64,
)

# 即時波形/頻譜圖設定（儀表板「即時訊號」卡片用，與 AI 模型無關）
LIVE_SPECTRO_DURATION = 2.0     # 每段擷取秒數
LIVE_SPECTRO_REFRESH = 2.0      # 每隔多久更新一次
LIVE_SPECTRO_MAX_FREQ = 500     # 頻譜圖 y 軸上限(Hz)；植物電位都在低頻

# ──────────────────────────────────────────────
#  設定載入
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

config = load_config()

# ──────────────────────────────────────────────
#  Flask App 初始化
# ──────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), "templates"),
    static_folder=os.path.join(os.path.dirname(__file__), "static"),
)

# 核心模組
gpio = GPIOController(relay_pin=config["gpio"]["relay_pin"])
audio = AudioCapture(
    sample_rate=config.get("spikerbox", {}).get("sample_rate", config["audio"]["sample_rate"]),
    duration=config["audio"]["duration_seconds"],
    device_keywords=config.get("spikerbox", {}).get("device_keywords", []),
    fallback_device=config.get("spikerbox", {}).get("fallback_device"),
)
dataset_mgr = DatasetManager(
    raw_dir=os.path.join(PROJECT_ROOT, config["dataset"]["save_dir"]),
    spectrogram_dir=os.path.join(PROJECT_ROOT, config["dataset"]["spectrogram_dir"]),
)

# 延遲載入推論引擎（模型檔可能不存在）
inference_engine = None
preprocessor = None

def _init_inference():
    global inference_engine, preprocessor
    if inference_engine is not None:
        return True
    try:
        model_path = os.path.normpath(os.path.join(PROJECT_ROOT, config["model_path"]))
        if not os.path.isfile(model_path):
            _safe_print(f"[MODEL] 模型檔不存在: {model_path}，推論功能暫不啟用")
            return False
        from pi.inference import TFLiteInferenceEngine
        from pi.preprocessor import AudioPreprocessor
        inference_engine = TFLiteInferenceEngine(model_path=model_path)
        preprocessor = AudioPreprocessor(
            spectrogram_save_dir=os.path.join(PROJECT_ROOT, config["dataset"]["spectrogram_dir"])
        )
        _safe_print("[MODEL] 推論引擎載入成功")
        return True
    except Exception as e:
        _safe_print(f"[MODEL] 推論引擎載入失敗: {e}")
        return False

# ──────────────────────────────────────────────
#  系統狀態
# ──────────────────────────────────────────────
system_state = {
    "pump_on": False,
    "auto_mode": False,
    "monitoring": False,
    "last_irrigation": None,
    "total_cycles": 0,
    "start_time": datetime.now().isoformat(),
    "activity_log": [],
    "latest_inference": None,
    "latest_spectrogram_b64": None,
    "inference_history": [],
    # 即時波形/頻譜（永遠在跑，不需要模型）
    "live_spectro_b64": None,
    "live_spectro_stats": "",
    "live_spectro_mode": "starting",
    "live_spectro_ts": None,
}
state_lock = threading.Lock()
monitor_thread = None
monitor_stop_event = threading.Event()

# 序列埠/音訊裝置一次只能給一個迴圈用，AI 監測和即時頻譜共用這把鎖
capture_lock = threading.Lock()

MAX_LOG_ENTRIES = 100
MAX_HISTORY = 50


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def add_log(action: str, detail: str = ""):
    with state_lock:
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "action": action,
            "detail": detail,
        }
        system_state["activity_log"].insert(0, entry)
        if len(system_state["activity_log"]) > MAX_LOG_ENTRIES:
            system_state["activity_log"] = system_state["activity_log"][:MAX_LOG_ENTRIES]
    _safe_print(f"[WEB] {entry['time']} | {action} | {detail}")


add_log("系統啟動", "Web 控制面板已上線")

# ──────────────────────────────────────────────
#  自動監測迴圈
# ──────────────────────────────────────────────
def _monitor_loop():
    """背景自動監測：錄音 → 頻譜圖 → 推論 → 灌溉決策"""
    from pi.irrigation import IrrigationDecisionMaker

    decision_maker = IrrigationDecisionMaker(
        thirsty_threshold=config["logic"]["thirsty_threshold"],
        consecutive_required=config["logic"]["consecutive_thirsty_required"],
    )
    interval = config["audio"]["listen_interval_seconds"]
    irrigation_dur = config["gpio"]["irrigation_duration_seconds"]
    temp_wav = os.path.join(os.path.dirname(__file__), "live_audio.wav")

    has_model = _init_inference()

    add_log("自動監測啟動", f"間隔 {interval} 秒")

    while not monitor_stop_event.is_set():
        try:
            # 1. 錄音（跟即時頻譜迴圈共用裝置，用鎖避免同時讀序列埠）
            with capture_lock:
                success = audio.record_audio(temp_wav)
            if not success:
                add_log("錄音失敗", "等待下一次")
                monitor_stop_event.wait(interval)
                continue

            # 2. 存到資料集
            if config["dataset"]["enabled"]:
                label = "unlabeled"
                audio.save_to_dataset(
                    temp_wav,
                    save_dir=os.path.join(PROJECT_ROOT, config["dataset"]["save_dir"]),
                    label=label,
                )

            # 3. 推論（若模型可用）
            if has_model and preprocessor and inference_engine:
                input_tensor = preprocessor.process(temp_wav)
                result = inference_engine.predict(input_tensor)

                # 更新頻譜圖 base64
                spec_path = preprocessor.get_latest_spectrogram_path()
                if spec_path and os.path.isfile(spec_path):
                    with open(spec_path, "rb") as f:
                        b64 = base64.b64encode(f.read()).decode("ascii")
                    with state_lock:
                        system_state["latest_spectrogram_b64"] = b64

                # 更新推論結果
                with state_lock:
                    system_state["latest_inference"] = {
                        "prediction": result["prediction"],
                        "confidence": round(result["confidence"], 4),
                        "probabilities": {k: round(v, 4) for k, v in result["probabilities"].items()},
                        "time": datetime.now().strftime("%H:%M:%S"),
                    }
                    system_state["inference_history"].insert(0, system_state["latest_inference"])
                    if len(system_state["inference_history"]) > MAX_HISTORY:
                        system_state["inference_history"] = system_state["inference_history"][:MAX_HISTORY]

                pred = result["prediction"]
                conf = result["confidence"]
                add_log(f"推論: {pred}", f"信心度 {conf:.1%}")

                # 4. 灌溉決策
                needs_water = decision_maker.process_prediction(result)
                if needs_water:
                    add_log("自動灌溉啟動", f"運轉 {irrigation_dur} 秒")
                    with state_lock:
                        system_state["pump_on"] = True
                    gpio.turn_on_relay()
                    monitor_stop_event.wait(irrigation_dur)
                    gpio.turn_off_relay()
                    with state_lock:
                        system_state["pump_on"] = False
                        system_state["total_cycles"] += 1
                        system_state["last_irrigation"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    add_log("自動灌溉完成", f"已運轉 {irrigation_dur} 秒")
                    # 灌完休息 60 秒
                    monitor_stop_event.wait(60)
            else:
                add_log("錄音完成", "模型未載入，僅存檔")

        except Exception as e:
            add_log("監測錯誤", str(e))

        # 等待下一次
        monitor_stop_event.wait(interval)

    add_log("自動監測已停止", "")
    # 清理
    temp_wav_path = os.path.join(os.path.dirname(__file__), "live_audio.wav")
    if os.path.exists(temp_wav_path):
        try:
            os.remove(temp_wav_path)
        except Exception:
            pass


# ──────────────────────────────────────────────
#  即時波形/頻譜迴圈（背景常駐，與 AI 模型無關）
# ──────────────────────────────────────────────
def _spectro_loop():
    """持續擷取短音訊 → 畫波形+頻譜圖 → 存進 system_state，供儀表板輪詢。"""
    tmp_wav = os.path.join(os.path.dirname(__file__), "live_spectro.wav")
    audio.duration = LIVE_SPECTRO_DURATION

    while True:
        try:
            with capture_lock:
                ok = audio.record_audio(tmp_wav)
            if ok and os.path.isfile(tmp_wav):
                x, sr = wav_to_mono_float(tmp_wav)
                png = render_waveform_spectrogram_b64(x, sr, max_freq=LIVE_SPECTRO_MAX_FREQ)
                stats = stats_text(x, sr)
                with state_lock:
                    system_state["live_spectro_b64"] = png
                    system_state["live_spectro_stats"] = stats
                    system_state["live_spectro_mode"] = audio.mode
                    system_state["live_spectro_ts"] = datetime.now().strftime("%H:%M:%S")
        except Exception as e:
            _safe_print(f"[SPECTRO] 擷取失敗: {e}")
        time.sleep(LIVE_SPECTRO_REFRESH)


threading.Thread(target=_spectro_loop, daemon=True).start()


# ──────────────────────────────────────────────
#  路由：頁面
# ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("dashboard.html")


# ──────────────────────────────────────────────
#  路由：API — 系統狀態
# ──────────────────────────────────────────────
@app.route("/api/status", methods=["GET"])
def get_status():
    with state_lock:
        uptime = (datetime.now() - datetime.fromisoformat(system_state["start_time"])).total_seconds()
        return jsonify({
            "pump_on": system_state["pump_on"],
            "monitoring": system_state["monitoring"],
            "last_irrigation": system_state["last_irrigation"],
            "total_cycles": system_state["total_cycles"],
            "uptime_seconds": int(uptime),
            "relay_pin": config["gpio"]["relay_pin"],
            "has_model": inference_engine is not None,
        })


# ──────────────────────────────────────────────
#  路由：API — 水泵控制
# ──────────────────────────────────────────────
@app.route("/api/pump/on", methods=["POST"])
def pump_on():
    with state_lock:
        if system_state["pump_on"]:
            return jsonify({"success": False, "message": "水泵已在運轉中"}), 400
        system_state["pump_on"] = True
    gpio.turn_on_relay()
    add_log("水泵開啟", "手動操作")
    return jsonify({"success": True, "message": "水泵已開啟"})


@app.route("/api/pump/off", methods=["POST"])
def pump_off():
    with state_lock:
        system_state["pump_on"] = False
    gpio.turn_off_relay()
    add_log("水泵關閉", "手動操作")
    return jsonify({"success": True, "message": "水泵已關閉"})


@app.route("/api/pump/cycle", methods=["POST"])
def pump_cycle():
    data = request.get_json(silent=True) or {}
    duration = data.get("duration", config["gpio"]["irrigation_duration_seconds"])
    duration = max(1, min(duration, 300))

    with state_lock:
        if system_state["pump_on"]:
            return jsonify({"success": False, "message": "水泵已在運轉中"}), 400
        system_state["pump_on"] = True

    def _run():
        try:
            gpio.turn_on_relay()
            add_log("定時灌溉開始", f"預計 {duration} 秒")
            time.sleep(duration)
        finally:
            gpio.turn_off_relay()
            with state_lock:
                system_state["pump_on"] = False
                system_state["total_cycles"] += 1
                system_state["last_irrigation"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            add_log("定時灌溉完成", f"已運轉 {duration} 秒")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"success": True, "message": f"灌溉已啟動（{duration}秒）", "duration": duration})


# ──────────────────────────────────────────────
#  路由：API — 自動監測控制
# ──────────────────────────────────────────────
@app.route("/api/monitor/start", methods=["POST"])
def monitor_start():
    global monitor_thread
    with state_lock:
        if system_state["monitoring"]:
            return jsonify({"success": False, "message": "監測已在執行中"}), 400
        system_state["monitoring"] = True

    monitor_stop_event.clear()
    monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
    monitor_thread.start()
    return jsonify({"success": True, "message": "自動監測已啟動"})


@app.route("/api/monitor/stop", methods=["POST"])
def monitor_stop():
    with state_lock:
        if not system_state["monitoring"]:
            return jsonify({"success": False, "message": "監測未在執行"}), 400
        system_state["monitoring"] = False

    monitor_stop_event.set()
    return jsonify({"success": True, "message": "自動監測已停止"})


# ──────────────────────────────────────────────
#  路由：API — 推論與頻譜圖
# ──────────────────────────────────────────────
@app.route("/api/inference/latest", methods=["GET"])
def inference_latest():
    with state_lock:
        return jsonify({
            "inference": system_state["latest_inference"],
            "spectrogram_b64": system_state["latest_spectrogram_b64"],
        })


@app.route("/api/spectro", methods=["GET"])
def live_spectro():
    """即時波形/頻譜圖（與 AI 模型無關，儀表板「即時訊號」卡片用）。"""
    with state_lock:
        return jsonify({
            "png": system_state["live_spectro_b64"],
            "stats_text": system_state["live_spectro_stats"],
            "mode": system_state["live_spectro_mode"],
            "ts": system_state["live_spectro_ts"],
        })


@app.route("/api/history", methods=["GET"])
def inference_history():
    limit = request.args.get("limit", 20, type=int)
    with state_lock:
        return jsonify({"history": system_state["inference_history"][:limit]})


# ──────────────────────────────────────────────
#  路由：API — 資料集統計
# ──────────────────────────────────────────────
@app.route("/api/dataset/stats", methods=["GET"])
def dataset_stats():
    stats = dataset_mgr.get_stats()
    return jsonify(stats)


# ──────────────────────────────────────────────
#  路由：API — 日誌
# ──────────────────────────────────────────────
@app.route("/api/log", methods=["GET"])
def get_log():
    with state_lock:
        return jsonify({"log": system_state["activity_log"]})


@app.route("/api/log/clear", methods=["POST"])
def clear_log():
    with state_lock:
        system_state["activity_log"] = []
    add_log("日誌已清除", "手動操作")
    return jsonify({"success": True})


# ──────────────────────────────────────────────
#  主程式入口
# ──────────────────────────────────────────────
if __name__ == "__main__":
    _safe_print("=" * 55)
    _safe_print("[TOMATO] Irrigation System - Web Control Panel")
    _safe_print("=" * 55)
    _safe_print(f"  Relay Pin: GPIO {config['gpio']['relay_pin']}")
    _safe_print(f"  Sample Rate: {config['audio']['sample_rate']} Hz")
    _safe_print(f"  Listen Interval: {config['audio']['listen_interval_seconds']}s")
    _safe_print(f"  Dataset Auto-Save: {'ON' if config['dataset']['enabled'] else 'OFF'}")
    _safe_print(f"  URL: http://0.0.0.0:5000")
    _safe_print("=" * 55)

    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    finally:
        monitor_stop_event.set()
        gpio.cleanup()
        # 主動關閉序列埠，否則下次啟動可能撞到 (5, 'Input/output error')
        try:
            audio.close()
        except Exception:
            pass
