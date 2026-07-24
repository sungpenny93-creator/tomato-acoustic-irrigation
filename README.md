# 🍅 番茄生理聲學監測與自動化灌溉系統

> Tomato Physiological Acoustic Monitoring & Automated Irrigation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4-red.svg)](https://www.raspberrypi.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📖 專案簡介

本系統利用植物在水分逆境（Water Stress）下，木質部導管產生**氣穴現象（Cavitation）**所發出的特異性超音波（20–100kHz），作為判斷番茄缺水的直接生理指標。

系統結合**輕量化 CNN 模型（MobileNetV2）**與**邊緣運算（Raspberry Pi）**，實現即時聲紋辨識與自動灌溉控制，真正做到「按需給水」的精準農業。

### 核心技術
- 🎙️ 壓電感測器擷取番茄莖部微震動
- 📊 時頻譜轉換（Mel-Spectrogram）將聲音視覺化
- 🧠 MobileNetV2 輕量化 CNN 辨識聲紋特徵
- 🔧 頻譜減法（Spectral Subtraction）抗噪演算法
- 📦 TensorFlow Lite 量化部署於 Raspberry Pi
- 💧 GPIO 控制繼電器驅動自動灌溉設備

## 🏗️ 系統架構

```
壓電感測器 → 音訊擷取 → 頻譜減法抗噪 → Mel-Spectrogram → CNN 推論 → 灌溉控制
                                                                    ↓
                                                              LED 狀態指示
                                                              本地日誌記錄
```

## 📁 專案結構

```
tomato-acoustic-irrigation/
├── src/                    # 💻 電腦端：模型訓練
│   ├── config.py           #   全域參數設定
│   ├── audio_utils.py      #   音訊載入與處理
│   ├── spectrogram.py      #   頻譜圖轉換
│   ├── noise_reduction.py  #   頻譜減法抗噪
│   ├── dataset.py          #   資料集建構與分割
│   ├── model.py            #   MobileNetV2 模型定義
│   ├── train.py            #   模型訓練主程式
│   ├── evaluate.py         #   模型評估（混淆矩陣）
│   └── export_tflite.py    #   模型量化匯出
│
├── pi/                     # 🍓 樹莓派端：推論與控制
│   ├── config.json         #   執行設定
│   ├── audio_capture.py    #   即時音訊擷取
│   ├── preprocessor.py     #   即時前處理
│   ├── inference.py        #   TFLite 推論引擎
│   ├── gpio_controller.py  #   GPIO 硬體控制
│   ├── irrigation.py       #   灌溉決策邏輯
│   ├── logger.py           #   日誌記錄
│   └── main.py             #   主程式入口
│
├── data/                   # 📂 資料集（不上傳 Git）
├── models/                 # 🧠 訓練好的模型
├── notebooks/              # 📓 實驗用 Jupyter Notebook
├── scripts/                # 🔧 輔助腳本
├── tests/                  # 🧪 單元測試
├── docs/                   # 📖 文件
├── requirements-pc.txt     # 電腦端套件
└── requirements-pi.txt     # 樹莓派端套件
```

## 🚀 快速開始

### 電腦端（模型訓練）

```bash
# 1. 建立虛擬環境
python -m venv venv
venv\Scripts\activate          # Windows

# 2. 安裝套件
pip install -r requirements-pc.txt

# 3. 準備資料集（將 PlantSounds 資料放入 data/raw/）

# 4. 執行訓練管線
python -m src.train
```

### 樹莓派端（推論與控制）

```bash
# 1. 安裝套件
pip install -r requirements-pi.txt

# 2. 將訓練好的 .tflite 模型複製到 models/

# 3. 啟動監測系統
python -m pi.main
```

## 📊 資料集

本專案使用 [Khait et al. (2023)](https://doi.org/10.1016/j.cell.2023.03.009) 公開釋出的植物聲學資料集：

| 類別 | 數量 | 說明 |
|------|------|------|
| Tomato Dry | 1,622 | 番茄缺水（氣穴聲） |
| Tomato Cut | 660 | 番茄被切斷 |
| Empty Pot | 1,036 | 空盆（對照組） |
| Greenhouse Noises | 1,378 | 溫室環境噪音 |

## 📚 參考文獻

1. Khait, I., et al. (2023). "Sounds emitted by plants under stress are airborne and informative." *Cell*, 186(7).
2. Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."
3. Boll, S. (1979). "Suppression of acoustic noise in speech using spectral subtraction." *IEEE TASSP*.
4. TensorFlow Lite Documentation - Model Optimization.

## 📄 授權條款

本專案採用 [MIT License](LICENSE) 授權。
