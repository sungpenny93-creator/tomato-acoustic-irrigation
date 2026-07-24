"""
模型評估模組 — 含混淆矩陣與完整分類報告

使用方式：
    python -m src.evaluate

輸出：
    1. 驗證集準確率
    2. 混淆矩陣（Confusion Matrix）
    3. 精確率（Precision）、召回率（Recall）、F1-Score
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from src.config import Config


def evaluate_model():
    """對驗證集進行完整評估，輸出分類報告和混淆矩陣。"""

    # 載入模型
    print(f"載入模型: {Config.BEST_MODEL_PATH}")
    model = tf.keras.models.load_model(Config.BEST_MODEL_PATH)

    # 載入驗證資料
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        Config.VAL_DIR,
        shuffle=False,
        batch_size=Config.BATCH_SIZE,
        image_size=Config.IMG_SIZE,
    )

    class_names = val_dataset.class_names
    print(f"類別: {class_names}")

    # ======= 基礎評估 =======
    loss, accuracy = model.evaluate(val_dataset)
    print(f"\n[RESULT] 驗證集損失: {loss:.4f}")
    print(f"[RESULT] 驗證集準確率: {accuracy * 100:.2f}%")

    # ======= 收集所有預測結果 =======
    y_true = []
    y_pred = []

    for images, labels in val_dataset:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ======= 分類報告 =======
    print("\n[REPORT] 分類報告：")
    print("=" * 60)
    report = classification_report(
        y_true, y_pred, target_names=class_names, digits=4
    )
    print(report)

    # ======= 混淆矩陣 =======
    cm = confusion_matrix(y_true, y_pred)
    print("[MATRIX] 混淆矩陣：")
    print(cm)

    # 儲存混淆矩陣圖片
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix - Tomato Acoustic Model")

    save_path = os.path.join(Config.MODEL_DIR, "confusion_matrix.png")
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n[OK] 混淆矩陣圖片已儲存: {save_path}")

    return accuracy, report, cm


if __name__ == "__main__":
    evaluate_model()
