"""
模型訓練主程式

使用方式：
    # 在專案根目錄執行
    python -m src.train

這個腳本會：
    1. 從 data/dataset/train 和 data/dataset/val 讀取頻譜圖
    2. 建構 MobileNetV2 模型
    3. 訓練模型（含 EarlyStopping、ReduceLR 等策略）
    4. 儲存最佳模型為 .keras 格式
    5. 自動轉換為 TFLite 量化模型
"""

import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)
from src.config import Config, ensure_dirs
from src.model import build_model


def main():
    ensure_dirs()

    # ==================================================================
    # 1. 載入資料集
    # ==================================================================
    print("[1/5] 正在讀取訓練集與驗證集...")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        Config.TRAIN_DIR,
        shuffle=True,
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMG_SIZE,
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        Config.VAL_DIR,
        shuffle=False,
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMG_SIZE,
    )

    # 顯示偵測到的類別
    class_names = train_dataset.class_names
    print(f"[INFO] 偵測到 {len(class_names)} 種類別：{class_names}")

    # 效能優化：預讀取加速
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # ==================================================================
    # 2. 建構模型
    # ==================================================================
    print("\n[2/5] 正在建構 MobileNetV2 模型...")
    model = build_model()

    # 顯示模型摘要
    model.summary()

    # ==================================================================
    # 3. 設定 Callbacks（訓練策略）
    # ==================================================================
    callbacks = [
        # EarlyStopping：如果驗證損失連續 N 輪沒有改善，就提前停止
        EarlyStopping(
            monitor="val_loss",
            patience=Config.EARLY_STOP_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        # ModelCheckpoint：每次驗證準確率創新高，就自動存檔
        ModelCheckpoint(
            Config.BEST_MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        # ReduceLROnPlateau：如果驗證損失停滯，就自動降低學習率
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=Config.REDUCE_LR_FACTOR,
            patience=Config.REDUCE_LR_PATIENCE,
            min_lr=Config.REDUCE_LR_MIN,
            verbose=1,
        ),
    ]

    # ==================================================================
    # 4. 開始訓練
    # ==================================================================
    print(f"\n[3/5] 開始訓練！最多 {Config.EPOCHS} 輪...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=Config.EPOCHS,
        callbacks=callbacks,
    )

    print(f"\n[4/5] 訓練完成！最佳模型已儲存至: {Config.BEST_MODEL_PATH}")

    # ==================================================================
    # 5. 匯出 TFLite 量化模型
    # ==================================================================
    print("\n[5/5] 正在匯出 TFLite 量化模型...")
    export_tflite()

    return history


def export_tflite():
    """
    將訓練好的 Keras 模型轉換為 TFLite 量化模型。

    量化效果：
        - 模型大小壓縮至約 1/4
        - 推論速度在 Raspberry Pi 上提升 2-3 倍
        - 精度損失通常小於 1%
    """
    # 載入最佳模型
    best_model = tf.keras.models.load_model(Config.BEST_MODEL_PATH)

    # 建立轉換器
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)

    # 啟用動態範圍量化（Float32 → Int8）
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 執行轉換
    tflite_model = converter.convert()

    # 儲存
    with open(Config.TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    # 比較大小
    import os
    keras_size = os.path.getsize(Config.BEST_MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(Config.TFLITE_MODEL_PATH) / (1024 * 1024)

    print(f"[OK] TFLite 模型已匯出: {Config.TFLITE_MODEL_PATH}")
    print(f"   Keras 模型大小:  {keras_size:.1f} MB")
    print(f"   TFLite 模型大小: {tflite_size:.1f} MB")
    print(f"   壓縮比: {keras_size / tflite_size:.1f}x")


if __name__ == "__main__":
    main()
