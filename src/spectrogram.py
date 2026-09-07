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

    # ------------------------------------------------------------------
    # 重要防呆：奈奎斯特頻率 (Nyquist frequency) 檢查
    # ------------------------------------------------------------------
    # Config.FMIN=20000Hz、FMAX=100000Hz 是根據 Khait et al. (2023)
    # 論文的錄音設備所設定的（原始資料取樣率高達 250kHz，
    # 奈奎斯特頻率 = 250000/2 = 125kHz，所以看得到 20kHz~100kHz 的
    # 氣穴超音波訊號）。
    #
    # 但是！Plant SpikerBox 實際錄音的取樣率通常只有 10000Hz 左右
    # （見 pi/config.json 的 audio.sample_rate），根據取樣定理，
    # 這種訊號最高只能還原到「奈奎斯特頻率 = sr / 2」的頻率內容，
    # 也就是最多只有 5000Hz！
    #
    # 如果 fmin(20000) 已經超過奈奎斯特頻率，等於是在要求
    # librosa 畫出一段「這個取樣率下根本不可能存在」的頻段，
    # 算出來的 Mel 濾波器組會全部落在有效頻率之外，導致：
    #   - 產生的 mel_spec 幾乎全部是極小值/全黑一片（沒有任何資訊）
    #   - 模型看到的頻譜圖等於是「空白圖片」，永遠學不會分辨
    #     thirsty / normal / noise，準確率會長期異常低落
    #   - librosa 通常還會跳出 UserWarning: "Empty filters detected"
    #
    # 這裡加上自動防呆：如果設定的 fmin/fmax 超出目前這筆音訊
    # 實際的奈奎斯特頻率，就自動退回「這個取樣率下可用的全部頻寬」
    # (0 ~ sr/2)，並印出警告，讓你能立刻發現「這筆音訊的取樣率
    # 涵蓋不到原本設計的氣穴頻段」，需要回頭檢查硬體/設定是否正確
    # （例如：SpikerBox 的取樣率設定太低、或訓練資料與實際部署
    # 用的頻段假設不一致，這兩者必須對齊才能讓模型真正有效）。
    nyquist = sr / 2.0
    fmin = Config.FMIN
    fmax = Config.FMAX

    if fmin >= nyquist:
        print(
            f"[WARNING] 取樣率 {sr}Hz 的奈奎斯特頻率只有 {nyquist:.0f}Hz，"
            f"低於設定的 FMIN={fmin}Hz，原本設計的氣穴頻段完全錄不到！"
            f"已自動改用 0~{nyquist:.0f}Hz 的全頻寬，但這通常代表訓練資料"
            f"（高取樣率）與實際部署裝置（低取樣率）的頻段假設不一致，"
            f"建議檢查 SpikerBox 取樣率設定或改用符合此取樣率重新訓練的模型。"
        )
        fmin = 0
        fmax = nyquist
    elif fmax > nyquist:
        # fmin 還在可用範圍內，但 fmax 超過了，只需把上限夾回奈奎斯特頻率
        print(
            f"[WARNING] FMAX={fmax}Hz 超過取樣率 {sr}Hz 的奈奎斯特頻率 "
            f"({nyquist:.0f}Hz)，已自動夾到 {nyquist:.0f}Hz。"
        )
        fmax = nyquist

    # 計算 Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=Config.N_FFT,
        hop_length=Config.HOP_LENGTH,
        n_mels=Config.N_MELS,
        fmin=fmin,             # 已依實際取樣率做過奈奎斯特頻率防呆
        fmax=fmax,              # 已依實際取樣率做過奈奎斯特頻率防呆
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

    # 確保輸出目錄存在（save_path 可能是「純檔名」，此時 dirname 會是空字串，跳過即可）
    parent_dir = os.path.dirname(save_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

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
