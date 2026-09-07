"""
感測器模組 — DS18B20 溫度 + 電阻式土壤濕度（數位 DO 模式）
==========================================================
設計目標：
  * 樹莓派沒有 ADC，所以電阻式土壤感測器用「數位輸出 DO」讀取。
    模組上的藍色可調電阻(trimpot)拿小螺絲起子轉，設定「多乾算缺水」的門檻。
  * 在一般電腦(Windows/Mac)上會自動進入「模擬模式」，方便先跑流程。
  * 電阻式感測器會電解腐蝕，這裡支援「用 GPIO 供電」：平常斷電，
    只有要讀值前才通電幾秒 → 壽命可以延長好幾倍。

接線（BCM 編號，對照 README 的實體腳位圖）：
  DS18B20：
      VCC  → 3.3V (Pin 1)
      GND  → GND  (Pin 6)
      DATA → GPIO 4 (Pin 7)
      DATA 與 VCC 之間接一顆 4.7kΩ 上拉電阻（必要！）
      先啟用 1-Wire：sudo raspi-config → Interface Options → 1-Wire → Enable → 重開機

  電阻式土壤濕度模組：
      VCC → 3.3V（不要接 5V！DO 會變成 5V 打壞 GPIO）
      GND → GND
      DO  → GPIO 23 (Pin 16)   ← 可在 config.json 改
      AO  → 不接（沒有 ADC）
      (選用) 改成 VCC → GPIO 24，讓程式控制供電以減少腐蝕
"""

import os
import sys
import glob
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    import RPi.GPIO as GPIO
    IS_RPI = True
except (ImportError, RuntimeError):
    IS_RPI = False


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode("ascii"))


# ======================================================================
# DS18B20 溫度感測器（1-Wire）
# ======================================================================
class TemperatureSensor:
    """讀取 DS18B20。找不到硬體時回傳模擬溫度。"""

    def __init__(self, mock_value: float = 25.0):
        self.mock_value = mock_value
        self.device_file = None

        if IS_RPI:
            # 1-Wire 裝置會出現在 /sys/bus/w1/devices/28-xxxxxxxx/
            try:
                base = "/sys/bus/w1/devices/"
                folders = glob.glob(base + "28*")
                if folders:
                    self.device_file = os.path.join(folders[0], "w1_slave")
                    _safe_print(f"[TEMP] 找到 DS18B20: {folders[0]}")
                else:
                    _safe_print("[TEMP] 找不到 DS18B20（是否已啟用 1-Wire 並重開機？）→ 用模擬值")
            except Exception as e:
                _safe_print(f"[TEMP] 初始化失敗: {e} → 用模擬值")
        else:
            _safe_print("[TEMP] 非樹莓派環境 → 用模擬值")

    def read_celsius(self):
        """回傳攝氏溫度(float)，讀取失敗回傳 None。"""
        if self.device_file is None:
            return self.mock_value

        try:
            with open(self.device_file, "r") as f:
                lines = f.readlines()
            # 第一行結尾要有 "YES" 代表 CRC 正確
            if not lines or lines[0].strip()[-3:] != "YES":
                time.sleep(0.2)
                return self.read_celsius()
            eq = lines[1].find("t=")
            if eq == -1:
                return None
            return float(lines[1][eq + 2:]) / 1000.0
        except Exception as e:
            _safe_print(f"[TEMP] 讀取失敗: {e}")
            return None


# ======================================================================
# 電阻式土壤濕度感測器（數位 DO 模式）
# ======================================================================
class SoilMoistureSensor:
    """
    只讀數位輸出：乾 / 濕 兩種狀態。
    do_pin        : DO 接的 GPIO（BCM）
    dry_is_high   : DO 在「乾」的時候是高電平還低電平（不同板子不一樣，先實測）
    power_pin     : 選用。若接了，讀值前才通電，減少電解腐蝕
    settle_seconds: 通電後等多久再讀（讓讀數穩定）
    """

    def __init__(self, do_pin: int = 23, dry_is_high: bool = True,
                 power_pin: int = None, settle_seconds: float = 2.0):
        self.do_pin = do_pin
        self.dry_is_high = dry_is_high
        self.power_pin = power_pin
        self.settle_seconds = settle_seconds
        self._mock_counter = 0

        if IS_RPI:
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.do_pin, GPIO.IN)
            if self.power_pin is not None:
                GPIO.setup(self.power_pin, GPIO.OUT)
                GPIO.output(self.power_pin, GPIO.LOW)  # 預設斷電
            _safe_print(f"[SOIL] DO=GPIO{self.do_pin} "
                        f"{'(GPIO供電 GPIO'+str(self.power_pin)+')' if self.power_pin else '(常時供電)'}")
        else:
            _safe_print("[SOIL] 非樹莓派環境 → 用模擬值（乾濕交替）")

    def read_is_dry(self) -> bool:
        """回傳 True = 缺水(乾)，False = 濕潤。"""
        if not IS_RPI:
            # 模擬：每讀 4 次給一次「乾」
            self._mock_counter += 1
            return self._mock_counter % 4 == 0

        try:
            if self.power_pin is not None:
                GPIO.output(self.power_pin, GPIO.HIGH)
                time.sleep(self.settle_seconds)

            level = GPIO.input(self.do_pin)

            if self.power_pin is not None:
                GPIO.output(self.power_pin, GPIO.LOW)

            is_high = (level == GPIO.HIGH)
            return is_high if self.dry_is_high else (not is_high)
        except Exception as e:
            _safe_print(f"[SOIL] 讀取失敗: {e}")
            return False  # 出錯時當作不缺水，避免誤澆

    def cleanup(self):
        if IS_RPI and self.power_pin is not None:
            try:
                GPIO.output(self.power_pin, GPIO.LOW)
            except Exception:
                pass


# ======================================================================
if __name__ == "__main__":
    _safe_print("=== 感測器測試（讀 5 次，每次間隔 2 秒）===")
    temp = TemperatureSensor()
    soil = SoilMoistureSensor(do_pin=23)

    for i in range(5):
        t = temp.read_celsius()
        dry = soil.read_is_dry()
        t_str = f"{t:.2f}°C" if t is not None else "讀取失敗"
        _safe_print(f"  [{i+1}] 溫度={t_str}   土壤={'缺水(乾)' if dry else '濕潤'}")
        time.sleep(2)

    soil.cleanup()
    _safe_print("測試結束")
