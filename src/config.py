"""
全域設定檔 — 番茄生理聲學監測系統

所有可調整的超參數都集中在這裡管理。
修改設定時只需改這個檔案，不需要到處找程式碼修改。

使用方式：
    from src.config import Config
    print(Config.SAMPLE_RATE)   # 250000
"""

import os


class Config:
    """集中管理所有設定參數的類別"""

    # ===================================================================
    # 路徑設定
    # ===================================================================
    # 專案根目錄（自動偵測，不需手動修改）
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # 資料路徑
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    RAW_AUDIO_DIR = os.path.join(DATA_DIR, "raw")
    SPECTROGRAM_DIR = os.path.join(DATA_DIR, "spectrograms")
    DATASET_DIR = os.path.join(DATA_DIR, "dataset")
    TRAIN_DIR = os.path.join(DATASET_DIR, "train")
    VAL_DIR = os.path.join(DATASET_DIR, "val")
    TEST_DIR = os.path.join(DATASET_DIR, "test")

    # 模型路徑
    MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.keras")
    TFLITE_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.tflite")

    # ===================================================================
    # 音訊參數
    # ===================================================================
    # Khait et al. (2023) 資料集使用 250kHz 取樣率
    SAMPLE_RATE = 250000

    # ===================================================================
    # 頻譜圖參數
    # ===================================================================
    N_FFT = 2048            # FFT 視窗大小
    HOP_LENGTH = 128        # 跳躍長度（越小解析度越高，但計算量越大）
    N_MELS = 256            # Mel 濾波器組數量（影響頻率解析度）

    # 氣穴現象頻率範圍 (Khait et al., 2023)
    # 設定這個範圍可以讓模型專注學習氣穴訊號，忽略低頻噪音
    #
    # ⚠️ 注意：這個範圍是根據 Khait 資料集 250kHz 高取樣率設計的。
    # 若音訊來源的取樣率比較低（例如樹莓派上 Plant SpikerBox 目前
    # 設定的 10000Hz，奈奎斯特頻率只有 5000Hz），這個頻段會完全
    # 錄不到！src/spectrogram.py 的 audio_to_mel_spectrogram() 已經
    # 加了自動防呆（超過奈奎斯特頻率時會自動退回全頻寬 0~sr/2），
    # 但長期而言，訓練資料的取樣率與實際部署裝置的取樣率應該要
    # 盡量一致，模型才學得到真正有意義的特徵。
    FMIN = 20000            # 最低頻率 20kHz
    FMAX = 100000           # 最高頻率 100kHz（受取樣率限制）

    # 頻譜圖圖片尺寸
    SPECTROGRAM_FIGSIZE = (10, 5)

    # ===================================================================
    # 抗噪參數（頻譜減法）
    # ===================================================================
    # 噪音頻譜估計的過度減法係數
    # 越大 = 去噪越激進（可能傷害訊號）
    # 越小 = 保留更多原始特徵（可能殘留噪音）
    SPECTRAL_SUB_ALPHA = 1.5

    # 頻譜下限保護，避免減法後產生負值
    SPECTRAL_SUB_FLOOR = 0.02

    # ===================================================================
    # 資料集參數
    # ===================================================================
    # MobileNetV2 的標準輸入尺寸
    IMG_SIZE = (224, 224)

    # 分類標籤（依照字母排序，與 TensorFlow 讀取資料夾的順序一致）
    CLASSES = ["noise", "normal", "thirsty"]
    NUM_CLASSES = len(CLASSES)

    # 資料分割比例
    TRAIN_SPLIT = 0.8       # 80% 用於訓練
    VAL_SPLIT = 0.2         # 20% 用於驗證

    # ===================================================================
    # 資料增強參數
    # ===================================================================
    # 高斯噪音增強的強度
    NOISE_FACTOR = 0.005

    # 時間平移增強的比例
    TIME_SHIFT_RATIO = 0.2

    # ===================================================================
    # 訓練參數
    # ===================================================================
    BATCH_SIZE = 8           # 每次餵給模型的圖片數量
    EPOCHS = 50              # 最大訓練輪數（有 EarlyStopping 會提前結束）
    LEARNING_RATE = 1e-5     # 學習率（遷移學習建議用較小的值）

    # MobileNetV2 微調：凍結前 N 層
    FINE_TUNE_AT = 100

    # Dropout 比率（防止過擬合）
    DROPOUT_RATE = 0.5

    # EarlyStopping 耐心值
    EARLY_STOP_PATIENCE = 6

    # ReduceLROnPlateau 參數
    REDUCE_LR_FACTOR = 0.5
    REDUCE_LR_PATIENCE = 2
    REDUCE_LR_MIN = 1e-6

    # ===================================================================
    # 樹莓派 GPIO 腳位（BCM 編號）
    # ===================================================================
    RELAY_PIN = 17           # 繼電器控制（水泵）
    LED_GREEN_PIN = 27       # 綠色 LED（正常狀態）
    LED_RED_PIN = 22         # 紅色 LED（缺水警示）

    # ===================================================================
    # 灌溉控制參數
    # ===================================================================
    # 連續偵測到 N 次缺水訊號才觸發灌溉（避免誤判）
    STRESS_THRESHOLD = 3

    # 每次灌溉持續秒數
    IRRIGATION_DURATION = 30

    # 灌溉後冷卻時間（秒），避免重複觸發
    COOLDOWN_PERIOD = 300


# 確保所有必要的資料夾都存在
def ensure_dirs():
    """自動建立所有需要的資料夾"""
    dirs = [
        Config.RAW_AUDIO_DIR,
        Config.SPECTROGRAM_DIR,
        Config.TRAIN_DIR,
        Config.VAL_DIR,
        Config.TEST_DIR,
        Config.MODEL_DIR,
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
