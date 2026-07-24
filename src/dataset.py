"""
資料集建構模組 — 完整的資料處理管線

這個模組負責把原始 WAV 檔案變成可以訓練的圖片資料集。

完整流程：
    F:\PlantSounds\        (原始資料)
        ↓  prepare_raw_data()     將原始資料分類到 data/raw/
        ↓  generate_spectrograms() WAV → 抗噪 → 頻譜圖
        ↓  augment_data()          資料增強（加噪、平移）
        ↓  split_dataset()         分割為 train / val
    data/dataset/train/    (可以訓練了！)
    data/dataset/val/      (用來驗證)
"""

import os
import shutil
import random
import numpy as np
import warnings
from src.config import Config, ensure_dirs
from src.audio_utils import load_audio, normalize_audio, is_valid_audio
from src.noise_reduction import spectral_subtraction
from src.spectrogram import audio_to_mel_spectrogram, save_spectrogram_image

warnings.filterwarnings("ignore")


# =====================================================================
# Khait et al. 資料集的分類對應表
# =====================================================================
# 原始資料夾名稱 → 我們的分類名稱
KHAIT_MAPPING = {
    "Empty Pot": "normal",           # 空盆（無植物）→ 正常/對照組
    "Tomato Dry": "thirsty",         # 番茄缺水 → 缺水
    "Greenhouse Noises": "noise",    # 溫室噪音 → 噪音
}
# 注意：Tomato Cut (660 筆) 暫不使用，未來可作為第四類別


def prepare_raw_data(source_dir: str) -> dict:
    """
    步驟 1：將 Khait et al. 原始資料複製並分類到 data/raw/ 底下。

    這個函式會在 data/raw/ 建立以下結構：
        data/raw/normal/   ← 來自 Empty Pot
        data/raw/thirsty/  ← 來自 Tomato Dry
        data/raw/noise/    ← 來自 Greenhouse Noises

    參數：
        source_dir (str): PlantSounds 資料夾的路徑
            例如 "F:\\PlantSounds"

    回傳：
        dict: 每個類別複製了多少檔案
            例如 {"normal": 1036, "thirsty": 1622, "noise": 1378}
    """
    ensure_dirs()
    stats = {}

    for original_name, our_name in KHAIT_MAPPING.items():
        src_path = os.path.join(source_dir, original_name)
        dst_path = os.path.join(Config.RAW_AUDIO_DIR, our_name)
        os.makedirs(dst_path, exist_ok=True)

        if not os.path.exists(src_path):
            print(f"[!] 找不到 {src_path}，跳過")
            stats[our_name] = 0
            continue

        # 取得所有 WAV 檔案
        wav_files = [f for f in os.listdir(src_path) if f.endswith(".wav")]
        copied = 0

        for filename in wav_files:
            src_file = os.path.join(src_path, filename)
            dst_file = os.path.join(dst_path, filename)

            # 如果目的地已經有了，就不重複複製
            if not os.path.exists(dst_file):
                shutil.copy2(src_file, dst_file)

            copied += 1

        stats[our_name] = copied
        print(f"[OK] {original_name} -> {our_name}: {copied} 個檔案")

    return stats


def generate_spectrograms(apply_noise_reduction: bool = True) -> dict:
    """
    步驟 2：將 data/raw/ 中的所有 WAV 檔案轉換為頻譜圖。

    處理流程（對每個 WAV 檔案）：
        1. 載入音訊
        2. 驗證是否有效
        3. [可選] 頻譜減法抗噪
        4. 正規化
        5. 轉換為 Mel-Spectrogram
        6. 儲存為 PNG 圖片

    參數：
        apply_noise_reduction (bool): 是否啟用頻譜減法抗噪

    回傳：
        dict: 每個類別產生了多少張頻譜圖
    """
    ensure_dirs()
    stats = {}

    for class_name in Config.CLASSES:
        input_dir = os.path.join(Config.RAW_AUDIO_DIR, class_name)
        output_dir = os.path.join(Config.SPECTROGRAM_DIR, class_name)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(input_dir):
            print(f"[!] 找不到 {input_dir}，跳過")
            stats[class_name] = 0
            continue

        wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
        count = 0

        print(f"\n[>>] 正在處理【{class_name}】類別 ({len(wav_files)} 個音檔)...")

        for i, filename in enumerate(wav_files):
            try:
                audio_path = os.path.join(input_dir, filename)
                base_name = filename.replace(".wav", "")

                # 載入音訊
                y, sr = load_audio(audio_path)

                # 驗證
                if not is_valid_audio(y):
                    continue

                # 抗噪
                if apply_noise_reduction:
                    y = spectral_subtraction(y, sr)

                # 正規化
                y = normalize_audio(y)

                # ===== 原始版頻譜圖 =====
                mel_db = audio_to_mel_spectrogram(y, sr)
                save_path = os.path.join(output_dir, f"{base_name}_ori.png")
                save_spectrogram_image(mel_db, sr, save_path)
                count += 1

                # ===== 資料增強 1：加入微量高斯噪音 =====
                noise = np.random.randn(len(y))
                y_noisy = y + Config.NOISE_FACTOR * noise
                mel_db_noisy = audio_to_mel_spectrogram(y_noisy, sr)
                save_path = os.path.join(output_dir, f"{base_name}_aug_noise.png")
                save_spectrogram_image(mel_db_noisy, sr, save_path)
                count += 1

                # ===== 資料增強 2：時間平移 =====
                shift = int(len(y) * Config.TIME_SHIFT_RATIO)
                y_shifted = np.roll(y, shift)
                mel_db_shifted = audio_to_mel_spectrogram(y_shifted, sr)
                save_path = os.path.join(output_dir, f"{base_name}_aug_shift.png")
                save_spectrogram_image(mel_db_shifted, sr, save_path)
                count += 1

                # 進度回報
                if (i + 1) % 100 == 0:
                    print(f"   ...已處理 {i + 1} / {len(wav_files)}")

            except Exception as e:
                print(f"   [X] 處理 {filename} 時發生錯誤: {e}")

        stats[class_name] = count
        print(f"   [OK] 【{class_name}】完成！產生 {count} 張頻譜圖")

    return stats


def split_dataset(train_ratio: float = None) -> dict:
    """
    步驟 3：將頻譜圖分割為訓練集和驗證集。

    分割策略：
        1. 隨機打亂每個類別的圖片順序
        2. 按比例分割（預設 80/20）
        3. 複製到 data/dataset/train/ 和 data/dataset/val/

    參數：
        train_ratio (float): 訓練集佔比，預設 Config.TRAIN_SPLIT (0.8)

    回傳：
        dict: 各類別的分割統計
    """
    if train_ratio is None:
        train_ratio = Config.TRAIN_SPLIT

    ensure_dirs()
    stats = {}

    for class_name in Config.CLASSES:
        src_dir = os.path.join(Config.SPECTROGRAM_DIR, class_name)
        train_dir = os.path.join(Config.TRAIN_DIR, class_name)
        val_dir = os.path.join(Config.VAL_DIR, class_name)

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            print(f"[!] 找不到 {src_dir}，跳過")
            continue

        # 清空舊資料（避免新舊混雜）
        for f in os.listdir(train_dir):
            os.remove(os.path.join(train_dir, f))
        for f in os.listdir(val_dir):
            os.remove(os.path.join(val_dir, f))

        # 取得所有 PNG
        images = [f for f in os.listdir(src_dir) if f.endswith(".png")]

        if len(images) == 0:
            print(f"[!] 【{class_name}】沒有圖片，跳過")
            continue

        # 隨機打亂
        random.shuffle(images)

        # 分割
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # 複製檔案
        for img in train_images:
            shutil.copy2(os.path.join(src_dir, img), os.path.join(train_dir, img))
        for img in val_images:
            shutil.copy2(os.path.join(src_dir, img), os.path.join(val_dir, img))

        stats[class_name] = {
            "total": len(images),
            "train": len(train_images),
            "val": len(val_images),
        }
        print(f"[OK] 【{class_name}】 {len(train_images)} train / {len(val_images)} val")

    return stats


# =====================================================================
# 主程式：一鍵執行完整資料管線
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("[START] 番茄聲學資料處理管線")
    print("=" * 60)

    # 步驟 1
    print("\n[1/3] 複製並分類原始資料...")
    raw_stats = prepare_raw_data(r"F:\PlantSounds")

    # 步驟 2
    print("\n[2/3] 轉換頻譜圖 + 抗噪 + 資料增強...")
    spec_stats = generate_spectrograms(apply_noise_reduction=True)

    # 步驟 3
    print("\n[3/3] 分割訓練集與驗證集...")
    split_stats = split_dataset()

    # 最終統計
    print("\n" + "=" * 60)
    print("[DONE] 資料管線執行完畢！最終統計：")
    print("=" * 60)
    for cls, info in split_stats.items():
        print(f"  {cls}: {info['train']} train / {info['val']} val (total: {info['total']})")
