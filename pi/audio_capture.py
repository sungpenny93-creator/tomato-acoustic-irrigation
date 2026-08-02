import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import os
import shutil
import random

class AudioCapture:
    def __init__(self, sample_rate: int = 500000, duration: float = 1.0):
        self.sample_rate = sample_rate
        self.duration = duration
        self.has_mic = True
        
        try:
            # 測試麥克風是否可用
            sd.check_input_settings(samplerate=self.sample_rate, channels=1)
            print(f"[AUDIO] 麥克風測試成功。取樣率: {self.sample_rate}Hz")
        except Exception as e:
            self.has_mic = False
            print(f"[WARNING] 麥克風測試失敗: {e}")
            print("[WARNING] 切換為「模擬錄音模式」：將隨機抽取現有音檔來測試系統。")
            
    def record_audio(self, output_path: str) -> bool:
        """錄製音訊並儲存為 wav 檔，回傳是否成功"""
        if self.has_mic:
            try:
                print(f"🎤 開始錄音 ({self.duration} 秒)...")
                # 錄製單聲道
                recording = sd.rec(
                    int(self.duration * self.sample_rate), 
                    samplerate=self.sample_rate, 
                    channels=1, 
                    dtype='int16'
                )
                sd.wait()  # 等待錄音完成
                write(output_path, self.sample_rate, recording)
                print(f"✅ 錄音完成: {output_path}")
                return True
            except Exception as e:
                print(f"[ERROR] 錄音時發生錯誤: {e}")
                return False
        else:
            # 模擬模式：從資料集隨便複製一個音檔當作錄到的聲音
            mock_source_dir = r"F:\PlantSounds\Tomato Dry"
            try:
                files = [f for f in os.listdir(mock_source_dir) if f.endswith(".wav")]
                if not files:
                    raise FileNotFoundError
                chosen_file = random.choice(files)
                shutil.copy(os.path.join(mock_source_dir, chosen_file), output_path)
                print(f"🎤 [MOCK] 模擬錄音完成 (使用: {chosen_file})")
                return True
            except Exception as e:
                print(f"[ERROR] 模擬錄音失敗 (找不到測試音檔): {e}")
                return False

if __name__ == "__main__":
    capture = AudioCapture()
    capture.record_audio("test_recording.wav")
