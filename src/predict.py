import os
import sys
import numpy as np
import tensorflow as tf
from src.config import Config
from src.audio_utils import load_audio, normalize_audio
from src.noise_reduction import spectral_subtraction
from src.spectrogram import audio_to_mel_spectrogram

def predict_single_audio(audio_path: str):
    """讀取單一 WAV 音檔，並預測其類別"""
    
    if not os.path.exists(audio_path):
        print(f"[!] 找不到檔案: {audio_path}")
        return

    print(f"🎤 正在分析音訊: {audio_path}")
    
    # 1. 載入音訊
    y, sr = load_audio(audio_path)
    
    # 2. 抗噪處理
    y = spectral_subtraction(y, sr)
    
    # 3. 正規化
    y = normalize_audio(y)
    
    # 4. 轉換為頻譜圖 (得到 Numpy 陣列)
    mel_db = audio_to_mel_spectrogram(y, sr)
    
    # 為了讓 AI 能看，我們需要把它轉成 3 通道 (RGB) 圖片格式，並且調整大小為 128x128
    # 這裡我們模擬圖片的形狀
    img = tf.expand_dims(mel_db, axis=-1)  # 加上通道維度
    img = tf.image.resize(img, Config.IMG_SIZE) # 縮放到 128x128
    img = tf.image.grayscale_to_rgb(img) # 轉成 3 通道
    img = tf.expand_dims(img, axis=0) # 加上 batch 維度 (變成 [1, 128, 128, 3])
    
    # 5. 載入模型並預測
    print(f"🧠 載入模型...")
    model = tf.keras.models.load_model(Config.BEST_MODEL_PATH)
    predictions = model.predict(img, verbose=0)
    
    # 6. 解析結果
    class_names = Config.CLASSES
    predicted_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_idx]
    confidence = predictions[0][predicted_idx] * 100
    
    print("-" * 40)
    print(f"✅ 預測結果: 【{predicted_class}】")
    print(f"📊 信心水準: {confidence:.2f}%")
    
    # 印出每個類別的機率
    print("\n詳細機率分佈:")
    for i, name in enumerate(class_names):
        print(f"  - {name}: {predictions[0][i]*100:.2f}%")
    print("-" * 40)

if __name__ == "__main__":
    # 如果使用者有傳入參數，就測試該檔案；否則測試一個預設檔案
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        # 從測試集隨便挑一個正常植物的聲音來測試
        test_file = r"F:\PlantSounds\Empty Pot\id_0_sound_21.wav"
        
    predict_single_audio(test_file)
