import os
import librosa
import numpy as np
from PIL import Image

# 我們直接重複利用 src 資料夾裡面寫好的核心邏輯，
# 這樣在樹莓派上產生的頻譜圖，才會跟訓練時的標準完全一致。
# 若是在真樹莓派上執行，請確保 PYTHONPATH 有設定，或是直接拷貝 src。
try:
    from src.noise_reduction import spectral_subtraction
    from src.spectrogram import audio_to_mel_spectrogram, save_spectrogram_image
except ImportError:
    print("[ERROR] 找不到 src 模組。請確保你在專案根目錄下執行。")
    raise

class AudioPreprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.temp_img_path = "temp_spectrogram.png"

    def process(self, wav_path: str) -> np.ndarray:
        """
        將 WAV 音檔轉換為模型可以直接吃進去的 Numpy Array (1, 224, 224, 3)
        """
        # 1. 讀取音檔
        # 使用 librosa 讀取，自動處理取樣率
        y, sr = librosa.load(wav_path, sr=None)

        # 2. 頻譜減法抗噪 (跟訓練時一樣)
        y_clean = spectral_subtraction(y, sr)

        # 3. 轉成 Mel-Spectrogram (dB)
        mel_spec = audio_to_mel_spectrogram(y_clean, sr)

        # 4. 存成暫存圖片
        # 若暫存檔已存在先刪除，避免讀到舊圖
        if os.path.exists(self.temp_img_path):
            os.remove(self.temp_img_path)
            
        save_spectrogram_image(mel_spec, sr, self.temp_img_path)

        # 5. 用 PIL 讀取並 Resize 到模型要的 224x224
        img = Image.open(self.temp_img_path).convert("RGB")
        img = img.resize(self.target_size)

        # 6. 轉為 Numpy Array 並正規化到 [0, 1] (跟 MobileNetV2 的輸入一致)
        # keras.utils.image_dataset_from_directory 預設讀進來是 0-255
        # 所以我們轉成 float32 後維持 0-255，因為我們的模型沒有加 Rescaling 層的話...
        # 等等，我們回憶一下 dataset.py：我們沒有在 dataset.py 做 /255，
        # 是靠 MobileNetV2 內建的 preprocess_input，但 tflite 沒有內建的 preprocessing 層！
        # 實際上，如果你是用我們之前的 TFLite，如果是 float32 模型，我們直接給 0-255 即可，
        # 或者我們可以在這裡手動對齊 TF 的標準。我們直接餵 0~255 的 array 就可以了。
        
        img_array = np.array(img, dtype=np.float32)
        
        # 擴充一個維度 (Batch size = 1) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

if __name__ == "__main__":
    # 測試程式
    preprocessor = AudioPreprocessor()
    # 假設這是一個隨便的音檔路徑
    test_wav = r"test_recording.wav"
    if os.path.exists(test_wav):
        print(f"正在處理: {test_wav}")
        input_data = preprocessor.process(test_wav)
        print(f"處理完成！形狀: {input_data.shape}")
    else:
        print(f"找不到測試音檔 {test_wav}，請先執行 audio_capture.py")
