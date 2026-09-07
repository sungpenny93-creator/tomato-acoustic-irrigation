"""
音訊擷取模組 — 支援 Plant SpikerBox 序列埠 + USB 音訊自動偵測
================================================================
1. 優先搜尋序列埠 /dev/ttyACM* (新版 Plant SpikerBox 使用 CDC ACM 協定)
2. 其次搜尋 USB 音訊裝置 (舊版 SpikerBox 或一般 USB 麥克風)
3. 若都找不到，進入模擬模式（使用現有音檔測試）
"""

import numpy as np
from scipy.io.wavfile import write
import os
import shutil
import random
import glob
import time
import struct
from datetime import datetime

# === 序列埠相關（新版 Plant SpikerBox） ===
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# === USB 音訊相關（舊版 SpikerBox / 一般麥克風） ===
try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False


class AudioCapture:
    # 通訊模式常數
    MODE_SERIAL = "serial"       # 新版 SpikerBox (序列埠 /dev/ttyACM*)
    MODE_SOUNDDEVICE = "audio"   # 舊版 SpikerBox (USB 音訊)
    MODE_MOCK = "mock"           # 模擬模式 (無硬體)

    def __init__(self, sample_rate: int = 10000, duration: float = 1.0,
                 device_keywords: list = None, fallback_device=None,
                 serial_port: str = None, serial_baud: int = 230400):
        self.sample_rate = sample_rate
        self.duration = duration
        self.device_id = None
        self.has_mic = False
        self.device_name = "Unknown"
        self.mode = self.MODE_MOCK

        # 序列埠參數
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        self.ser = None

        # 偵測順序：序列埠 → USB 音訊 → 模擬模式
        keywords = device_keywords or ["SpikerBox", "Backyard", "USB Audio"]
        self._detect_all(keywords, fallback_device)

    # ------------------------------------------------------------------
    # 裝置偵測
    # ------------------------------------------------------------------
    def _detect_all(self, keywords: list, fallback_device):
        """依序嘗試：序列埠 → USB 音訊 → 模擬模式"""

        # 1. 嘗試序列埠 (新版 Plant SpikerBox)
        if self._detect_serial():
            return

        # 2. 嘗試 USB 音訊 (舊版 SpikerBox / 一般麥克風)
        if self._detect_audio(keywords, fallback_device):
            return

        # 3. 全部失敗，進入模擬模式
        self.mode = self.MODE_MOCK
        self.has_mic = False
        _safe_print("[WARNING] 未偵測到任何音訊裝置")
        _safe_print("[WARNING] 切換為模擬錄音模式（使用現有音檔測試）")

    def _detect_serial(self) -> bool:
        """搜尋序列埠上的 Plant SpikerBox"""
        if not HAS_SERIAL:
            _safe_print("[SERIAL] pyserial 未安裝，跳過序列埠偵測")
            return False

        # 如果使用者有指定序列埠，直接嘗試
        ports_to_try = []
        if self.serial_port:
            ports_to_try.append(self.serial_port)
        else:
            # 自動搜尋 /dev/ttyACM* (Linux) 或 COM* (Windows)
            ports_to_try = sorted(glob.glob("/dev/ttyACM*"))
            if not ports_to_try:
                ports_to_try = sorted(glob.glob("/dev/ttyUSB*"))

        if not ports_to_try:
            _safe_print("[SERIAL] 未找到序列埠裝置，跳過")
            return False

        for port in ports_to_try:
            try:
                _safe_print(f"[SERIAL] 嘗試連接: {port} (baud={self.serial_baud})...")
                test_ser = serial.Serial(
                    port=port,
                    baudrate=self.serial_baud,
                    timeout=2,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE
                )
                # 清空緩衝區
                time.sleep(0.5)
                test_ser.reset_input_buffer()

                # 嘗試讀取少量資料，驗證裝置有在傳資料
                test_data = test_ser.read(256)
                if len(test_data) > 0:
                    self.ser = test_ser
                    self.serial_port = port
                    self.mode = self.MODE_SERIAL
                    self.has_mic = True
                    self.device_name = f"Plant SpikerBox ({port})"
                    _safe_print(f"  ★ 成功連接 Plant SpikerBox: {port}")
                    _safe_print(f"  ★ 收到 {len(test_data)} bytes 測試資料")
                    _safe_print(f"[AUDIO] 裝置就緒: {self.device_name} (Serial, baud={self.serial_baud})")
                    return True
                else:
                    test_ser.close()
                    _safe_print(f"  → {port} 無資料回應，跳過")

            except Exception as e:
                _safe_print(f"  → {port} 連接失敗: {e}")

        return False

    def _detect_audio(self, keywords: list, fallback_device) -> bool:
        """搜尋 USB 音訊裝置 (舊版 SpikerBox / 一般麥克風)"""
        if not HAS_SD:
            _safe_print("[AUDIO] sounddevice 未安裝，跳過音訊偵測")
            return False

        try:
            devices = sd.query_devices()
            _safe_print(f"[AUDIO] 偵測到 {len(devices)} 個音訊裝置：")

            # 優先搜尋 SpikerBox 關鍵字
            for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                    name = dev['name']
                    for kw in keywords:
                        if kw.lower() in name.lower():
                            self.device_id = i
                            self.device_name = name
                            _safe_print(f"  ★ 找到 SpikerBox: [{i}] {name}")
                            break
                    if self.device_id is not None:
                        break

            # 如果沒找到，使用 fallback 或預設裝置
            if self.device_id is None:
                if fallback_device is not None:
                    self.device_id = fallback_device
                    self.device_name = f"Fallback Device #{fallback_device}"
                    _safe_print(f"  → 使用指定裝置: [{fallback_device}]")
                else:
                    default_input = sd.default.device[0]
                    if default_input is not None and default_input >= 0:
                        self.device_id = default_input
                        dev_info = sd.query_devices(default_input)
                        self.device_name = dev_info['name']
                        _safe_print(f"  → 使用系統預設輸入: [{default_input}] {self.device_name}")

            # 驗證裝置
            if self.device_id is not None:
                sd.check_input_settings(
                    device=self.device_id,
                    samplerate=self.sample_rate,
                    channels=1
                )
                self.has_mic = True
                self.mode = self.MODE_SOUNDDEVICE
                _safe_print(f"[AUDIO] 裝置就緒: {self.device_name} (SR={self.sample_rate}Hz)")
                return True

        except Exception as e:
            _safe_print(f"[AUDIO] 音訊裝置偵測失敗: {e}")

        return False

    # ------------------------------------------------------------------
    # 錄音功能
    # ------------------------------------------------------------------
    def record_audio(self, output_path: str) -> bool:
        """錄製音訊並儲存為 WAV，回傳是否成功"""
        if self.mode == self.MODE_SERIAL:
            return self._record_serial(output_path)
        elif self.mode == self.MODE_SOUNDDEVICE:
            return self._record_sounddevice(output_path)
        else:
            return self._mock_record(output_path)

    def _record_serial(self, output_path: str) -> bool:
        """從序列埠讀取 Plant SpikerBox 的 ADC 資料並存成 WAV"""
        try:
            num_samples = int(self.duration * self.sample_rate)
            _safe_print(f"[REC] 開始序列埠錄音 ({self.duration}s, {num_samples} samples)...")

            # 清空舊資料
            self.ser.reset_input_buffer()

            # Backyard Brains SpikerBox 序列協定：
            # 每個 sample frame 以高位元組 (MSB bit7=1) 開頭
            # 資料格式: [高位元組 (1XXXXXXX)] [低位元組 (0XXXXXXX)]
            # 10-bit ADC 值 = ((高位元組 & 0x7F) << 7) | (低位元組 & 0x7F)
            samples = []
            timeout_start = time.time()
            timeout_limit = self.duration + 5  # 多給 5 秒容錯

            # 讀取足夠的原始位元組
            bytes_needed = num_samples * 2 + 512  # 多讀一些以確保有足夠的 frames
            raw_data = bytearray()

            while len(raw_data) < bytes_needed:
                if time.time() - timeout_start > timeout_limit:
                    _safe_print(f"[WARNING] 序列埠讀取超時，已收到 {len(raw_data)} bytes")
                    break
                chunk = self.ser.read(min(1024, bytes_needed - len(raw_data)))
                if chunk:
                    raw_data.extend(chunk)

            # 解析 Backyard Brains 的雙位元組協定
            i = 0
            while i < len(raw_data) - 1 and len(samples) < num_samples:
                # 尋找高位元組 (bit7 = 1)
                if raw_data[i] & 0x80:
                    high_byte = raw_data[i] & 0x7F
                    if i + 1 < len(raw_data) and not (raw_data[i + 1] & 0x80):
                        low_byte = raw_data[i + 1] & 0x7F
                        # 組合成 10-bit ADC 值 (0~1023)
                        adc_value = (high_byte << 7) | low_byte
                        samples.append(adc_value)
                        i += 2
                        continue
                i += 1

            if len(samples) == 0:
                _safe_print("[ERROR] 無法從序列埠解析出任何 sample")
                return self._mock_record(output_path)

            _safe_print(f"[REC] 成功解析 {len(samples)} 個 samples")

            # 轉換成 numpy array，正規化為 int16 (-32768 ~ 32767)
            arr = np.array(samples, dtype=np.float32)
            # 原始 ADC 值是 0~1023 (10-bit)，先正規化到 -1.0 ~ 1.0
            arr = (arr - 512.0) / 512.0
            # 轉成 int16
            audio_int16 = np.clip(arr * 32767, -32768, 32767).astype(np.int16)

            # 如果 samples 不夠，用 0 補齊
            if len(audio_int16) < num_samples:
                padding = np.zeros(num_samples - len(audio_int16), dtype=np.int16)
                audio_int16 = np.concatenate([audio_int16, padding])

            # 存成 WAV
            write(output_path, self.sample_rate, audio_int16[:num_samples])
            _safe_print(f"[REC] 錄音完成: {output_path}")
            return True

        except Exception as e:
            _safe_print(f"[ERROR] 序列埠錄音失敗: {e}")
            # 嘗試重新連接
            self._reconnect_serial()
            return self._mock_record(output_path)

    def _reconnect_serial(self):
        """嘗試重新連接序列埠"""
        try:
            if self.ser:
                self.ser.close()
            time.sleep(1)
            self.ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.serial_baud,
                timeout=2
            )
            _safe_print(f"[SERIAL] 重新連接成功: {self.serial_port}")
        except Exception as e:
            _safe_print(f"[SERIAL] 重新連接失敗: {e}")
            self.mode = self.MODE_MOCK
            self.has_mic = False

    def _record_sounddevice(self, output_path: str) -> bool:
        """使用 sounddevice 從 USB 音訊裝置錄音"""
        try:
            _safe_print(f"[REC] 開始錄音 ({self.duration}s, SR={self.sample_rate}Hz)...")
            recording = sd.rec(
                int(self.duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                device=self.device_id
            )
            sd.wait()
            write(output_path, self.sample_rate, recording)
            _safe_print(f"[REC] 錄音完成: {output_path}")
            return True
        except Exception as e:
            _safe_print(f"[ERROR] 錄音失敗: {e}")
            return False

    def _mock_record(self, output_path: str) -> bool:
        """模擬模式：從現有資料集隨機複製一個音檔"""
        mock_dirs = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "Tomato Dry"),
            r"F:\PlantSounds\Tomato Dry",
        ]
        for mock_dir in mock_dirs:
            try:
                if os.path.isdir(mock_dir):
                    files = [f for f in os.listdir(mock_dir) if f.endswith(".wav")]
                    if files:
                        chosen = random.choice(files)
                        shutil.copy(os.path.join(mock_dir, chosen), output_path)
                        _safe_print(f"[MOCK] 模擬錄音 (使用: {chosen})")
                        return True
            except Exception:
                continue

        # 最後手段：產生一段靜音
        _safe_print("[MOCK] 無可用音檔，產生靜音測試")
        silence = np.zeros(int(self.duration * self.sample_rate), dtype=np.int16)
        write(output_path, self.sample_rate, silence)
        return True

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def list_devices(self) -> str:
        """列出所有音訊裝置（供除錯用）"""
        info = f"模式: {self.mode}\n裝置: {self.device_name}\n"
        if HAS_SD:
            info += str(sd.query_devices())
        if self.serial_port:
            info += f"\n序列埠: {self.serial_port} (baud={self.serial_baud})"
        return info

    def save_to_dataset(self, wav_path: str, save_dir: str, label: str = "unlabeled") -> str:
        """
        將錄好的 WAV 存一份到資料集目錄。
        路徑格式：save_dir/<日期>/<label>/timestamp.wav
        回傳儲存路徑。
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H%M%S_%f")[:-3]
        target_dir = os.path.join(save_dir, date_str, label)
        os.makedirs(target_dir, exist_ok=True)

        target_path = os.path.join(target_dir, f"{time_str}.wav")
        shutil.copy2(wav_path, target_path)
        _safe_print(f"[DATASET] 已存檔: {target_path}")
        return target_path

    def close(self):
        """關閉序列埠連線"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            _safe_print("[SERIAL] 序列埠已關閉")


def _safe_print(msg: str):
    """跨平台安全 print"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    print("=== 音訊裝置列表 ===")
    capture = AudioCapture()
    print(capture.list_devices())
    print("\n=== 測試錄音 ===")
    capture.record_audio("test_recording.wav")
    capture.close()
