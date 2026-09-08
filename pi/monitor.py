"""
番茄灌溉系統 — 整合版主程式入口
================================
單一指令啟動整個系統：Web 控制面板 + 自動監測。

啟動方式：
    python -m pi.monitor

這等同於 python -m pi.web_controller，
但會在啟動時自動開始監測迴圈。
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.web_controller import app, shutdown, _safe_print, config


def main():
    _safe_print("=" * 55)
    _safe_print("[TOMATO] Irrigation System - Full Monitor Mode")
    _safe_print("=" * 55)
    _safe_print(f"  Relay Pin:    GPIO {config['gpio']['relay_pin']}")
    _safe_print(f"  Sample Rate:  {config['audio']['sample_rate']} Hz")
    _safe_print(f"  Interval:     {config['audio']['listen_interval_seconds']}s")
    _safe_print(f"  Dataset:      {'ON' if config['dataset']['enabled'] else 'OFF'}")
    _safe_print(f"  Web Panel:    http://0.0.0.0:5000")
    _safe_print("=" * 55)
    _safe_print("")
    _safe_print("  打開瀏覽器 → http://<IP>:5000")
    _safe_print("  點「啟動監測」開始自動監測迴圈")
    _safe_print("")

    try:
        app.run(host="0.0.0.0", port=5000, debug=False)
    except KeyboardInterrupt:
        _safe_print("\n系統關閉中...")
    finally:
        shutdown()
        _safe_print("系統已安全關閉。")


if __name__ == "__main__":
    main()
