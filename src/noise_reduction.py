"""
頻譜減法抗噪模組 — 基於 Boll (1979) 的經典演算法

原理簡介（用白話文說）：
    想像你在農場旁邊錄音，背景一直有風扇聲「嗡嗡嗡」。
    頻譜減法的做法是：
    1. 先估計「嗡嗡嗡」的聲音長什麼樣子（噪音頻譜）
    2. 把整段錄音中的「嗡嗡嗡」成分減掉
    3. 留下來的就是番茄的氣穴聲

    本模組使用「中值估計法」來估計噪音：
    取整段聲音頻率的中位數 = 持續存在的背景噪音
    因為氣穴聲是「瞬間」的，不會影響中位數

參考文獻：
    Boll, S. (1979). "Suppression of acoustic noise in speech
    using spectral subtraction." IEEE TASSP.
"""

import librosa
import numpy as np
from src.config import Config


def spectral_subtraction(y: np.ndarray, sr: int,
                         alpha: float = None,
                         floor: float = None) -> np.ndarray:
    """
    頻譜減法：從含噪訊號中減去估計的背景噪音。

    演算法步驟：
        1. 對音訊做 STFT（短時傅立葉轉換），得到頻譜
        2. 將頻譜分成「振幅」和「相位」兩個部分
        3. 用中位數估計每個頻率的背景噪音強度
        4. 從振幅中減去噪音估計值
        5. 保留原始相位，重新合成乾淨的音訊

    參數：
        y (np.ndarray): 含噪音的音訊波形
        sr (int): 取樣率
        alpha (float): 過度減法係數。
            預設使用 Config.SPECTRAL_SUB_ALPHA (1.5)
            越大去噪越猛，但可能傷害有用訊號
        floor (float): 頻譜下限保護比例。
            預設使用 Config.SPECTRAL_SUB_FLOOR (0.02)
            避免減法後出現負值造成的「音樂噪音」偽影

    回傳：
        np.ndarray: 降噪後的音訊波形
    """
    if alpha is None:
        alpha = Config.SPECTRAL_SUB_ALPHA
    if floor is None:
        floor = Config.SPECTRAL_SUB_FLOOR

    # 步驟 1：STFT 將時域訊號轉成頻域
    # 結果是一個複數矩陣，橫軸=時間幀，縱軸=頻率
    stft = librosa.stft(y, n_fft=Config.N_FFT, hop_length=Config.HOP_LENGTH)

    # 步驟 2：分離振幅（聲音大小）和相位（聲音波形的時間點）
    magnitude, phase = librosa.magphase(stft)

    # 步驟 3：估計噪音頻譜
    # 對每個頻率帶取「中位數」→ 代表持續存在的背景噪音
    # 為什麼用中位數而不是平均值？
    #   因為氣穴聲是瞬間的「脈衝」，只佔少數時間幀，
    #   中位數不會被這些極端值影響，完美代表穩態噪音
    noise_estimate = np.median(magnitude, axis=1, keepdims=True)

    # 步驟 4：減法運算
    # 從原始振幅中減去 alpha 倍的噪音估計值
    # np.maximum 確保結果不會低於 floor * 原始振幅（避免負值）
    magnitude_clean = np.maximum(
        magnitude - alpha * noise_estimate,
        floor * magnitude
    )

    # 步驟 5：用乾淨的振幅 + 原始相位，重建音訊波形
    stft_clean = magnitude_clean * phase
    y_clean = librosa.istft(stft_clean, hop_length=Config.HOP_LENGTH)

    return y_clean
