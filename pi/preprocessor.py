"""
音訊前處理模組 — 將 WAV 轉為 Mel-Spectrogram
==============================================
支援 SpikerBox 低取樣率（10kHz）和 Khait 高取樣率（250kHz）。
每次產生的頻譜圖可自動儲存到資料集目錄。
"""

import os
import librosa
import numpy as np
from PIL import Image
from datetime import datetime

# ============================================================================
# 為什麼這裡常常炸掉：「No module named 'src'」除錯說明
# ============================================================================
# pi/preprocessor.py 需要用到「電腦端訓練用」的 src/noise_reduction.py 與
# src/spectrogram.py，這樣樹莓派上的前處理邏輯才會跟訓練時完全一致。
#
# 這兩個模組是用「絕對匯入」的方式寫的：
#       from src.noise_reduction import spectral_subtraction
# 這代表 Python 必須能在 sys.path 裡找到一個叫做 src 的「頂層套件」
# （也就是專案根目錄下要有一個 src/ 資料夾，裡面有 __init__.py）。
#
# 你在樹莓派上執行的指令是：
#       cd ~/tomato-acoustic-irrigation
#       python -m pi.monitor
# 只要是「在專案根目錄下」用 -m 執行，Python 就會自動把目前工作目錄
# （也就是 ~/tomato-acoustic-irrigation）加進 sys.path，理論上 src/
# 應該找得到。
#
# 但如果還是出現「No module named 'src'」，最常見、也幾乎可以說是
# 唯一合理的原因是：
#   ★★★ 樹莓派上的專案資料夾裡，根本沒有 src/ 這個資料夾！ ★★★
#
# 這通常發生在「只用 scp 手動複製部分檔案到樹莓派」的部署方式，例如
# README 裡示範的：
#       scp models/best_model.tflite pi@<IP>:~/tomato-acoustic-irrigation/models/
# 這種做法只會複製你指定的檔案，不會自動把整個專案（尤其是 src/）
# 一起帶過去。如果你不是用「git clone / git pull」把整包專案同步到
# 樹莓派，src/ 資料夾很容易就這樣被漏掉。
#
# 排查步驟（用 SSH 連進樹莓派後執行）：
#   1. 確認 src 資料夾是否存在：
#         ls -la ~/tomato-acoustic-irrigation/src
#      如果顯示 "No such file or directory"，就證實是上面說的原因。
#   2. 修正方式（擇一）：
#      a) 直接把整個專案用 git 同步過去（建議）：
#             樹莓派上第一次： git clone <你的repo網址> tomato-acoustic-irrigation
#             之後更新用：      cd ~/tomato-acoustic-irrigation && git pull
#      b) 或用 rsync 把整個資料夾（含 src/）同步過去：
#             rsync -av --exclude venv --exclude data ./ pi@<IP>:~/tomato-acoustic-irrigation/
#      c) 或至少手動把 src/ 整個資料夾 scp 過去：
#             scp -r src pi@<IP>:~/tomato-acoustic-irrigation/
#
# 下面的 try/except 加強了錯誤訊息：會把「目前工作目錄」「sys.path」
# 以及「src 資料夾到底存不存在」都印出來，這樣你下次遇到匯入失敗時，
# 一眼就能看出是「根本没有 src 資料夾」還是「src 裡面某個依賴套件
# 沒裝好」（例如 matplotlib，見 src/spectrogram.py 的說明與
# requirements-pi.txt 的修正）。
# ============================================================================
import sys

try:
    from src.noise_reduction import spectral_subtraction
    from src.spectrogram import audio_to_mel_spectrogram, save_spectrogram_image
except ImportError as e:
    # 計算「理論上」的專案根目錄（pi/ 的上一層），方便印出來讓你比對
    _project_root_guess = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _src_dir_guess = os.path.join(_project_root_guess, "src")

    print("=" * 60)
    print("[ERROR] 匯入 src 模組失敗，前處理模組無法使用。")
    print(f"        原始錯誤訊息: {e}")
    print(f"        目前工作目錄 (cwd): {os.getcwd()}")
    print(f"        推測的專案根目錄:    {_project_root_guess}")
    print(f"        推測的 src 資料夾:   {_src_dir_guess}")

    if str(e) == "No module named 'src'":
        # 這種訊息代表「src 這個頂層套件整個都找不到」，
        # 幾乎可以肯定是 src/ 資料夾根本沒有部署到這台機器上。
        if not os.path.isdir(_src_dir_guess):
            print("        → 診斷結果：src 資料夾『不存在』！")
            print("          請確認你是用 git clone/pull 或 rsync 把整個專案")
            print("          （包含 src/）同步到這台樹莓派，而不是只複製")
            print("          pi/ 和 models/ 資料夾。詳見本檔案上方的中文說明。")
        else:
            print("        → src 資料夾存在，但仍匯入失敗，")
            print("          請確認你是在『專案根目錄』下執行 python -m pi.monitor，")
            print("          而不是在 pi/ 資料夾內執行。")
    else:
        # 訊息不是 "No module named 'src'"，代表 src 套件本身找得到，
        # 但它內部 import 的某個依賴套件（例如 matplotlib）沒裝。
        print("        → 診斷結果：src 資料夾找得到，但它內部還需要的")
        print("          某個套件沒裝好（請看上面的原始錯誤訊息是哪個模組）。")
        print("          常見情況：src/spectrogram.py 需要 matplotlib，")
        print("          但 requirements-pi.txt 曾經漏掉這個套件，")
        print("          請執行： pip install matplotlib")
    print("=" * 60)
    raise


class AudioPreprocessor:
    def __init__(self, target_size=(224, 224), spectrogram_save_dir=None):
        self.target_size = target_size
        self.temp_img_path = "temp_spectrogram.png"
        self.spectrogram_save_dir = spectrogram_save_dir
        self.last_spectrogram_path = None

        if self.spectrogram_save_dir:
            os.makedirs(self.spectrogram_save_dir, exist_ok=True)

    def process(self, wav_path: str) -> np.ndarray:
        """
        將 WAV 音檔轉換為模型可以直接吃的 Numpy Array (1, 224, 224, 3)。
        同時儲存頻譜圖供 Web 面板顯示。
        """
        # 1. 讀取音檔（librosa 自動處理各種取樣率）
        y, sr = librosa.load(wav_path, sr=None)

        # 2. 頻譜減法抗噪
        y_clean = spectral_subtraction(y, sr)

        # 3. 轉成 Mel-Spectrogram (dB)
        mel_spec = audio_to_mel_spectrogram(y_clean, sr)

        # 4. 存成暫存圖片
        if os.path.exists(self.temp_img_path):
            os.remove(self.temp_img_path)
        save_spectrogram_image(mel_spec, sr, self.temp_img_path)

        # 5. 自動存一份到資料集目錄
        self.last_spectrogram_path = self.temp_img_path
        if self.spectrogram_save_dir:
            self._save_spectrogram_copy()

        # 6. 用 PIL 讀取並 Resize 到 224x224
        img = Image.open(self.temp_img_path).convert("RGB")
        img = img.resize(self.target_size)

        # 7. 轉為 float32 Numpy Array (1, 224, 224, 3)
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def _save_spectrogram_copy(self):
        """將暫存的頻譜圖複製一份到資料集目錄"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        save_path = os.path.join(self.spectrogram_save_dir, f"spec_{timestamp}.png")
        try:
            from shutil import copy2
            copy2(self.temp_img_path, save_path)
            self.last_spectrogram_path = save_path
        except Exception as e:
            _safe_print(f"[WARNING] 頻譜圖存檔失敗: {e}")

    def get_latest_spectrogram_path(self) -> str:
        """取得最新一張頻譜圖的路徑"""
        return self.last_spectrogram_path


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    preprocessor = AudioPreprocessor()
    test_wav = "test_recording.wav"
    if os.path.exists(test_wav):
        print(f"正在處理: {test_wav}")
        input_data = preprocessor.process(test_wav)
        print(f"處理完成！形狀: {input_data.shape}")
        print(f"頻譜圖路徑: {preprocessor.get_latest_spectrogram_path()}")
    else:
        print(f"找不到 {test_wav}，請先執行 audio_capture.py")
