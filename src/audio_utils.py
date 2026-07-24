"""
音訊工具函式 — 負責載入、驗證、正規化 WAV 檔案

這個模組是整個管線的「入口」：
    原始 WAV 檔 → audio_utils.py → 乾淨的 numpy 陣列

主要功能：
    1. load_audio()     : 載入單個 WAV 檔案
    2. normalize_audio(): 將音訊振幅正規化到 [-1, 1]
    3. pad_or_trim()    : 統一音訊長度（太短補零、太長截斷）
"""

import librosa
import numpy as np
import os
from src.config import Config


def load_audio(file_path: str, sr: int = None) -> tuple:
    """
    載入一個 WAV 音訊檔案，回傳音訊陣列和取樣率。

    參數：
        file_path (str): WAV 檔案的完整路徑
        sr (int, optional): 目標取樣率。
            - None = 保持原始取樣率（推薦，因為 Khait 資料集是 250kHz）
            - 指定數值 = 重新取樣

    回傳：
        tuple: (y, sr)
            y  - numpy 陣列，音訊波形資料
            sr - int，取樣率（每秒採樣數）

    範例：
        >>> y, sr = load_audio("data/raw/thirsty/id_4_sound_1.wav")
        >>> print(f"取樣率: {sr}, 長度: {len(y)} 個採樣點")
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到音訊檔案: {file_path}")

    # librosa.load() 會自動將音訊轉為單聲道（mono）
    # sr=None 代表保持原始取樣率，不做重新取樣
    y, sr = librosa.load(file_path, sr=sr)

    return y, sr


def normalize_audio(y: np.ndarray) -> np.ndarray:
    """
    將音訊振幅正規化到 [-1, 1] 範圍。

    為什麼要正規化？
        不同錄音設備、距離會導致音量差異很大。
        正規化確保所有音訊在相同的音量基準上被分析，
        讓微弱的氣穴聲也能被模型「看見」。

    參數：
        y (np.ndarray): 原始音訊波形

    回傳：
        np.ndarray: 正規化後的音訊波形
    """
    # 如果音訊全部都是 0（靜音或損壞），直接回傳避免除以零
    if np.max(np.abs(y)) == 0:
        return y

    return librosa.util.normalize(y)


def pad_or_trim(y: np.ndarray, target_length: int) -> np.ndarray:
    """
    統一音訊長度：太短的補零、太長的截斷。

    為什麼需要統一長度？
        CNN 模型需要固定大小的輸入。
        如果每個音訊片段長度不同，產生的頻譜圖大小也會不同，
        模型就沒辦法批次處理。

    參數：
        y (np.ndarray): 音訊波形
        target_length (int): 目標長度（採樣點數）

    回傳：
        np.ndarray: 長度統一的音訊波形
    """
    current_length = len(y)

    if current_length < target_length:
        # 太短 → 在尾巴補零（zero-padding）
        y = np.pad(y, (0, target_length - current_length), mode="constant")
    elif current_length > target_length:
        # 太長 → 只取前面一段
        y = y[:target_length]

    return y


def is_valid_audio(y: np.ndarray, min_length: int = 100) -> bool:
    """
    檢查音訊是否有效（非空、非全零、長度足夠）。

    參數：
        y (np.ndarray): 音訊波形
        min_length (int): 最小有效長度（採樣點數）

    回傳：
        bool: True = 有效, False = 無效（應跳過）
    """
    if y is None or len(y) == 0:
        return False
    if len(y) < min_length:
        return False
    if np.max(np.abs(y)) == 0:
        return False
    return True
