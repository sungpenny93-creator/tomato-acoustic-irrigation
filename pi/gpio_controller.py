import time
import sys


def _safe_print(msg: str):
    """跨平台安全 print，避免 Windows CP950 遇到 emoji 噴錯"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))

# 嘗試載入樹莓派專用的 GPIO 套件
try:
    import RPi.GPIO as GPIO
    IS_RPI = True
except (ImportError, RuntimeError):
    # 如果是在一般電腦 (Windows/Mac) 上執行，則進入模擬模式
    IS_RPI = False
    print("[WARNING] 找不到 RPi.GPIO 套件。已自動切換至「模擬模式 (Mock Mode)」進行測試。")


class GPIOController:
    def __init__(self, relay_pin: int):
        self.relay_pin = relay_pin
        self.is_rpi = IS_RPI
        
        if self.is_rpi:
            # 樹莓派硬體初始化
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(self.relay_pin, GPIO.OUT)
            # 預設關閉繼電器 (假設為高電平觸發)
            GPIO.output(self.relay_pin, GPIO.LOW)
            print(f"[GPIO] 實體 GPIO {self.relay_pin} 初始化完成。")
        else:
            print(f"[MOCK] 模擬 GPIO {self.relay_pin} 初始化完成。")

    def turn_on_relay(self):
        """開啟繼電器（啟動馬達）"""
        if self.is_rpi:
            GPIO.output(self.relay_pin, GPIO.HIGH)
            _safe_print(f"[GPIO] 繼電器 ON (Pin {self.relay_pin}) - 馬達運轉中 💦")
        else:
            _safe_print(f"[MOCK] 繼電器 ON (Pin {self.relay_pin}) - 模擬馬達運轉中 💦")

    def turn_off_relay(self):
        """關閉繼電器（停止馬達）"""
        if self.is_rpi:
            GPIO.output(self.relay_pin, GPIO.LOW)
            _safe_print(f"[GPIO] 繼電器 OFF (Pin {self.relay_pin}) - 馬達已停止 🛑")
        else:
            _safe_print(f"[MOCK] 繼電器 OFF (Pin {self.relay_pin}) - 模擬馬達已停止 🛑")

    def run_irrigation_cycle(self, duration_seconds: int):
        """執行一次完整的灌溉週期"""
        self.turn_on_relay()
        time.sleep(duration_seconds)
        self.turn_off_relay()

    def cleanup(self):
        """釋放 GPIO 資源"""
        if self.is_rpi:
            GPIO.cleanup()
            print("[GPIO] 資源已釋放。")
        else:
            print("[MOCK] 資源已釋放。")

if __name__ == "__main__":
    # 簡單的測試程式
    controller = GPIOController(relay_pin=17)
    print("開始測試灌溉 3 秒鐘...")
    controller.run_irrigation_cycle(3)
    controller.cleanup()
