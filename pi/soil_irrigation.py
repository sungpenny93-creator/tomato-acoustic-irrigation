"""
土壤濕度自動灌溉（不使用 AI）— A 路 / 基準系統
================================================
邏輯很單純：
    每隔一段時間讀土壤濕度 + 溫度
    → 連續 N 次偵測到「乾」就啟動水泵幾秒
    → 澆完進入冷卻期，避免連環狂澆

這支程式跟聲學/模型完全無關，只靠感測器門檻，
目的是先讓「會自動澆水的系統」實際運作起來。

啟動：
    python -m pi.soil_irrigation

設定都在 pi/config.json 的 "soil" 區塊，見下方 DEFAULTS 說明。
在一般電腦上會用模擬感測器跑（每 4 次讀到一次「乾」），可先看流程。
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pi.gpio_controller import GPIOController
from pi.sensors import TemperatureSensor, SoilMoistureSensor


DEFAULTS = {
    "do_pin": 23,                      # 土壤模組 DO 接的 GPIO(BCM)
    "power_pin": None,                 # 選用：土壤模組 VCC 由這隻 GPIO 供電(減少腐蝕)；不用就填 null
    "dry_is_high": True,              # DO 在「乾」時是高電平嗎(不同板子不同，實測後改)
    "read_interval_seconds": 300,      # 每幾秒讀一次感測器
    "consecutive_dry_required": 3,     # 連續幾次讀到「乾」才澆水(防誤判)
    "irrigation_duration_seconds": 5,  # 每次澆水幾秒
    "cooldown_seconds": 1800,          # 澆完後多久內不再澆(讓水滲透)
    "min_temp_c": 5.0,                # 低於這個溫度不澆水(避免寒害/結冰)；不需要就設 -100
}


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode("ascii"))


def load_soil_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    soil = dict(DEFAULTS)
    soil.update(cfg.get("soil", {}))
    relay_pin = cfg.get("gpio", {}).get("relay_pin", 17)
    return soil, relay_pin


def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    soil_cfg, relay_pin = load_soil_config()

    _safe_print("=" * 55)
    _safe_print("[SOIL-IRRIGATION] 土壤濕度自動灌溉（無 AI）")
    _safe_print("=" * 55)
    _safe_print(f"  繼電器腳位      : GPIO {relay_pin}")
    _safe_print(f"  土壤 DO 腳位    : GPIO {soil_cfg['do_pin']}")
    _safe_print(f"  讀取間隔        : {soil_cfg['read_interval_seconds']}s")
    _safe_print(f"  連續乾判定      : {soil_cfg['consecutive_dry_required']} 次")
    _safe_print(f"  每次澆水        : {soil_cfg['irrigation_duration_seconds']}s")
    _safe_print(f"  冷卻時間        : {soil_cfg['cooldown_seconds']}s")
    _safe_print(f"  最低澆水溫度    : {soil_cfg['min_temp_c']}°C")
    _safe_print("=" * 55)
    _safe_print("  Ctrl+C 停止\n")

    temp_sensor = TemperatureSensor()
    soil_sensor = SoilMoistureSensor(
        do_pin=soil_cfg["do_pin"],
        dry_is_high=soil_cfg["dry_is_high"],
        power_pin=soil_cfg["power_pin"],
    )
    gpio = GPIOController(relay_pin=relay_pin)

    consecutive_dry = 0
    last_watered_at = 0.0

    try:
        while True:
            temp_c = temp_sensor.read_celsius()
            is_dry = soil_sensor.read_is_dry()
            temp_str = f"{temp_c:.1f}°C" if temp_c is not None else "N/A"

            if is_dry:
                consecutive_dry += 1
            else:
                consecutive_dry = 0

            _safe_print(f"[{ts()}] 溫度={temp_str}  土壤={'乾' if is_dry else '濕'}  "
                        f"連續乾={consecutive_dry}/{soil_cfg['consecutive_dry_required']}")

            if consecutive_dry >= soil_cfg["consecutive_dry_required"]:
                in_cooldown = (time.time() - last_watered_at) < soil_cfg["cooldown_seconds"]
                too_cold = (temp_c is not None) and (temp_c < soil_cfg["min_temp_c"])

                if in_cooldown:
                    remain = int(soil_cfg["cooldown_seconds"] - (time.time() - last_watered_at))
                    _safe_print(f"[{ts()}] 達標但仍在冷卻期，還要 {remain}s")
                elif too_cold:
                    _safe_print(f"[{ts()}] 達標但溫度過低({temp_str})，暫不澆水")
                else:
                    _safe_print(f"[{ts()}] 💧 啟動水泵 {soil_cfg['irrigation_duration_seconds']}s")
                    gpio.run_irrigation_cycle(soil_cfg["irrigation_duration_seconds"])
                    last_watered_at = time.time()
                    consecutive_dry = 0
                    _safe_print(f"[{ts()}] 澆水完成，進入冷卻期 {soil_cfg['cooldown_seconds']}s")

            time.sleep(soil_cfg["read_interval_seconds"])

    except KeyboardInterrupt:
        _safe_print("\n[SOIL-IRRIGATION] 收到終止訊號，關閉中...")
    finally:
        soil_sensor.cleanup()
        gpio.cleanup()
        _safe_print("[SOIL-IRRIGATION] 已安全關閉。")


if __name__ == "__main__":
    main()
