"""
頻譜圖轉換模組 — 將一維音訊波形轉換為二維 Mel-Spectrogram 圖片

為什麼要把聲音變成圖片？
    CNN（卷積神經網路）原本是設計來「看圖片」的。
    如果我們能把聲音變成一張圖，就能借用 CNN 強大的
    圖像辨識能力來辨識聲音特徵。

    Mel-Spectrogram 就是這樣一種「聲音的照片」：
    - X 軸 = 時間
    - Y 軸 = 頻率（用 Mel 刻度，更符合人類聽覺）
    - 顏色深淺 = 該頻率在該時間點的能量大小

主要功能：
    1. audio_to_mel_spectrogram() : 音訊 → Mel 頻譜矩陣
    2. save_spectrogram_image()   : 頻譜矩陣 → PNG 圖片檔
"""

import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")  # 不開啟視窗，適合伺服器環境
import matplotlib.pyplot as plt
import numpy as np
import os
from src.config import Config


def audio_to_mel_spectrogram(y: np.ndarray, sr: int) -> np.ndarray:
    """
    將音訊波形轉換為 Mel-Spectrogram（以 dB 為單位）。

    參數：
        y (np.ndarray): 音訊波形（建議已經過抗噪和正規化）
        sr (int): 取樣率

    回傳：
        np.ndarray: Mel-Spectrogram 矩陣（dB 刻度）
            形狀為 (n_mels, time_frames)
    """
    # 先做最小長度防呆
    if len(y) < Config.N_FFT:
        y = np.pad(y, (0, Config.N_FFT - len(y)), mode="constant")

    # 計算 Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        fmin=Config.FMIN,     # 只關注 20kHz 以上（氣穴頻段）
        fmax=Config.FMAX,     # 上限 100kHz
    )

    # 轉換為 dB 刻度（人類聽覺是對數感知的）
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db


def save_spectrogram_image(mel_spec_db: np.ndarray, sr: int,
                           save_path: str) -> None:
    """
    將 Mel-Spectrogram 矩陣儲存為 PNG 圖片。

    注意：這個函式會：
        - 關閉座標軸（CNN 不需要看到刻度數字）
        - 清除所有邊距（讓圖片乾乾淨淨只有頻譜）
        - 自動釋放記憶體（避免處理幾千張圖後記憶體爆炸）

    參數：
        mel_spec_db (np.ndarray): dB 刻度的 Mel-Spectrogram
        sr (int): 取樣率（用於正確顯示頻率軸）
        save_path (str): 輸出圖片的完整路徑（含 .png 副檔名）
    """
    # 如果檔案已存在，跳過（避免重複處理浪費時間）
    if os.path.exists(save_path):
        return

    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 建立畫布
    fig = plt.figure(figsize=Config.SPECTROGRAM_FIGSIZE)

    # 繪製頻譜圖
    librosa.display.specshow(
        mel_spec_db,
        sr=sr,
        hop_length=Config.HOP_LENGTH,
        fmin=Config.FMIN,
    )

    # 關閉座標軸（CNN 只需要看頻譜的「紋路」，不需要數字）
    plt.axis("off")
    plt.tight_layout(pad=0)

    # 儲存圖片
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=100)

    # 釋放記憶體（非常重要！不做這步幾千張圖後記憶體會爆掉）
    fig.clf()
    plt.close("all")
