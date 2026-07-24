"""
模型定義模組 — MobileNetV2 遷移學習

什麼是遷移學習（Transfer Learning）？
    想像一個已經會辨識貓狗的 AI 大腦（在 ImageNet 上訓練過的）。
    我們不需要從零教它認東西，只需要：
    1. 把它已經學會的「看圖能力」保留下來（凍結底層）
    2. 換上一顆新的「分類腦袋」（自訂的全連結層）
    3. 只訓練這顆新腦袋來辨識「頻譜圖」

    這樣做的好處：
    - 需要的資料量大幅減少
    - 訓練速度更快
    - 效果通常比從零訓練更好

為什麼選 MobileNetV2？
    計畫書指定的。它使用「深度可分離卷積」，
    參數量比傳統 CNN 少很多，非常適合部署在 Raspberry Pi 上。
"""

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from src.config import Config


def build_model() -> Model:
    """
    建構 MobileNetV2 遷移學習模型。

    模型架構：
        MobileNetV2（凍結前 100 層） → GlobalAveragePooling2D
        → Dropout(0.5) → Dense(num_classes, softmax)

    回傳：
        tf.keras.Model: 編譯好、準備訓練的模型
    """
    # ---------------------------------------------------------------
    # 第一步：載入 MobileNetV2「已經學過」的大腦
    # ---------------------------------------------------------------
    # include_top=False：移除原本的分類層（ImageNet 有 1000 類，我們只有 3 類）
    # weights='imagenet'：載入在 ImageNet 上預訓練好的權重
    base_model = MobileNetV2(
        input_shape=(Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3),
        include_top=False,
        weights="imagenet",
    )

    # ---------------------------------------------------------------
    # 第二步：決定哪些層要「解凍」進行微調
    # ---------------------------------------------------------------
    # 底層學的是通用特徵（邊緣、紋理），不需要改
    # 高層學的是語義特徵，需要微調來適應我們的頻譜圖
    base_model.trainable = True

    print(f"MobileNetV2 總共有 {len(base_model.layers)} 層")
    print(f"凍結前 {Config.FINE_TUNE_AT} 層，解凍剩餘 "
          f"{len(base_model.layers) - Config.FINE_TUNE_AT} 層進行微調")

    for layer in base_model.layers[:Config.FINE_TUNE_AT]:
        layer.trainable = False

    # ---------------------------------------------------------------
    # 第三步：搭建我們自己的「分類腦袋」
    # ---------------------------------------------------------------
    x = base_model.output

    # GlobalAveragePooling2D：將特徵圖壓縮成一個向量
    # 比 Flatten 好的地方：參數更少，不容易過擬合
    x = GlobalAveragePooling2D()(x)

    # Dropout：訓練時隨機關閉一半的神經元
    # 這迫使模型學會用不同的神經元組合來做判斷，提高泛化能力
    x = Dropout(Config.DROPOUT_RATE)(x)

    # 最後一層：輸出每個類別的機率
    # softmax 確保所有機率加起來 = 1
    predictions = Dense(Config.NUM_CLASSES, activation="softmax")(x)

    # 組合成完整模型
    model = Model(inputs=base_model.input, outputs=predictions)

    # ---------------------------------------------------------------
    # 第四步：編譯模型（設定學習方式）
    # ---------------------------------------------------------------
    model.compile(
        optimizer=Adam(learning_rate=Config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
