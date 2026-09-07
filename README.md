# 🍅 番茄生理聲學監測與自動化灌溉系統

> Tomato Physiological Acoustic Monitoring & Automated Irrigation System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-4-red.svg)](https://www.raspberrypi.org/)
[![Flask](https://img.shields.io/badge/Flask-3.x-lightgrey.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 專案簡介

本系統利用 **Backyard Brains Plant SpikerBox** 擷取番茄的生物電訊號，結合 **MobileNetV2 輕量化 CNN 模型**，在 **Raspberry Pi** 上進行即時頻譜圖辨識，判定番茄是否缺水，並自動控制水泵灌溉。

整套系統透過 **Flask Web 控制面板**，讓使用者可以在手機或電腦上遠端監控與操控。

### 核心技術

- 🎙️ Plant SpikerBox 擷取番茄生物電位訊號
- 📊 時頻譜轉換（Mel-Spectrogram）將訊號視覺化
- 🧠 MobileNetV2 輕量化 CNN 辨識聲紋特徵
- 🔧 頻譜減法（Spectral Subtraction）抗噪演算法
- 📦 TensorFlow Lite 量化部署於 Raspberry Pi
- 💧 GPIO 控制繼電器驅動水泵自動灌溉
- 🌐 Flask Web 遠端控制面板（手機即時監控）

---

## 🏗️ 系統架構

```
                        ┌─────────────────────────────────────────┐
                        │           Raspberry Pi                  │
                        │                                         │
 番茄 → 電極 → SpikerBox USB ──→ 音訊擷取 (WAV)                    │
                        │              ↓                          │
                        │        ┌─── 存到資料集                    │
                        │        │    data/spikerbox_raw/          │
                        │        │                                │
                        │        └─── 即時推論管線                   │
                        │              ↓                          │
                        │        頻譜減法抗噪                       │
                        │              ↓                          │
                        │        Mel-Spectrogram → 自動存檔         │
                        │              ↓                          │
                        │        CNN 推論 (TFLite)                 │
                        │              ↓                          │
                        │     ┌─ thirsty → GPIO → 繼電器 → 水泵     │
                        │     ├─ normal  → 繼續監測                 │
                        │     └─ noise   → 忽略                    │
                        │              ↓                          │
                        │     Flask Web 控制面板 ←── 手機/電腦瀏覽器  │
                        └─────────────────────────────────────────┘
```

---

## 🔩 硬體材料清單

### 核心控制

| 元件 | 規格 / 型號 | 數量 | 用途 |
|------|------------|------|------|
| Raspberry Pi | 3B+ / 4 Model B | 1 | 主控板（AI 推論 + GPIO 控制） |
| Plant SpikerBox | Backyard Brains | 1 | 擷取番茄生物電訊號 |
| WeMos D1 (ESP8266) | WiFi UNO 開發板 | 2 | 備用 / 未來遠端感測節點 |
| USB 傳輸線 | USB-A to Micro-USB | 1 | Pi 供電 / SpikerBox 連接 |

### 感測器

| 元件 | 規格 / 型號 | 數量 | 用途 |
|------|------------|------|------|
| 壓電陶瓷震動感測模組 | KEYES 型號 1516 | 2 | 擷取番茄莖部氣穴聲波 |
| LM358 信號放大模組 | 雙運算放大器 | 1 | 放大壓電感測器微弱訊號 |
| 土壤濕度感測器 | 型號 0886（電阻式） | 1 | 偵測土壤含水量（輔助驗證） |
| DS18B20 溫度感測線 | 防水探頭，線長 1M | 1 | 監測環境 / 土壤溫度 |

### 灌溉控制

| 元件 | 規格 / 型號 | 數量 | 用途 |
|------|------------|------|------|
| 5V 單路繼電器模組 | MTARDK50011 | 2 | GPIO 控制水泵開關 |
| DC 微型潛水泵 | DC 3~5V 臥式沉水馬達 | 2 | 抽水灌溉 |
| 1N4007 二極體 | 整流 / 飛輪二極體 | 若干 | 吸收馬達反電動勢，保護電路 |
| PC817 4路光電隔離模組 | 型號 1736 | 1 | Pi 與動力電路之間的電氣隔離 |

---

## ⚡ 電路接線指南

### 控制迴路（低壓 5V）

用 2 條杜邦線把 Raspberry Pi 和繼電器模組連起來，並由外部電源供電：

```
   Raspberry Pi                    繼電器模組
   ┌──────────┐                  ┌──────────────┐
   │ (不接)   │                  │  VCC (接外部 5V)│
   │          │                  │              │
   │  Pin 9  ─┼── 黑線 ────────→│  GND (共地)   │
   │  (GND)   │                  │              │
   │  Pin 11 ─┼── 黃線 ────────→│  IN (信號)    │
   │ (GPIO 17)│                  │              │
   └──────────┘                  └──────────────┘
```

### 動力迴路（DC 5V）

```
   電源 (+) ──→ 繼電器 COM
   繼電器 NO ──→ 水泵 紅線 (+)
   水泵 黑線 (-) ──→ 電源 (-)

   ★ 1N4007 二極體反向並聯在水泵兩端 ★
     銀環端（陰極） → 接水泵紅線 (+) 端
     另一端（陽極） → 接水泵黑線 (-) 端
```

### 1N4007 接法圖示

```
    水泵 紅線 (+) ─────────── 水泵 黑線 (-)
               │                    │
               └──┤ Cathode(銀環) ├──┘
                  │    1N4007     │
                  └── Anode ─────┘

    方向：跟電流反向，平常不導通
    功能：吸收馬達斷電瞬間的反電動勢
```

### SpikerBox 連接

```
    Plant SpikerBox ──── USB 線 ──── Pi USB 孔

    Pi 會自動辨識為 USB 音訊裝置。
    系統會自動搜尋「SpikerBox」關鍵字來偵測裝置。
```

### 完整電路迴路

```
 ┌──────────── 控制迴路（低壓）─────────────┐
 │   外部電源 5V (+)     → 繼電器 VCC      │
 │   Pi Pin 9  (GND)     → 繼電器 GND      │
 │   外部電源 GND (-)    → 繼電器 GND      │
 │   Pi Pin 11 (GPIO 17) → 繼電器 IN       │
 └──────────────────────────────────────────┘

 ┌──────────── 動力迴路（DC 5V）────────────┐
 │   電源 (+)      → 繼電器 COM             │
 │   繼電器 NO     → 水泵 紅線 (+)           │
 │   水泵 黑線 (-) → 電源 (-)                │
 │   ★ 1N4007 反向並聯在水泵兩端 ★           │
 └──────────────────────────────────────────┘

 ┌──────────── 訊號擷取 ────────────────────┐
 │   SpikerBox USB → Pi USB 孔              │
 └──────────────────────────────────────────┘
```

---

## 📁 專案結構

```
tomato-acoustic-irrigation/
├── src/                        # 💻 電腦端：模型訓練
│   ├── config.py               #   全域參數設定
│   ├── audio_utils.py          #   音訊載入與處理
│   ├── spectrogram.py          #   頻譜圖轉換
│   ├── noise_reduction.py      #   頻譜減法抗噪
│   ├── dataset.py              #   資料集建構與分割
│   ├── model.py                #   MobileNetV2 模型定義
│   ├── train.py                #   模型訓練主程式
│   ├── evaluate.py             #   模型評估（混淆矩陣）
│   └── export_tflite.py        #   模型量化匯出
│
├── pi/                         # 🍓 樹莓派端：推論與控制
│   ├── config.json             #   執行設定（含 SpikerBox 參數）
│   ├── audio_capture.py        #   SpikerBox 音訊擷取（自動偵測）
│   ├── preprocessor.py         #   頻譜圖前處理（自動存檔）
│   ├── inference.py            #   TFLite 推論引擎
│   ├── gpio_controller.py      #   GPIO 硬體控制
│   ├── irrigation.py           #   灌溉決策邏輯（AI 版）
│   ├── sensors.py              #   DS18B20 溫度 + 電阻式土壤濕度（DO 數位模式）
│   ├── soil_irrigation.py      #   純門檻自動灌溉（不用 AI，基準系統）
│   ├── spectro_web.py          #   SpikerBox 即時頻譜圖 Web 檢視（診斷用）
│   ├── test_spikerbox.py       #   SpikerBox 診斷腳本（掃裝置 + 錄音 + 波形圖）
│   ├── dataset_manager.py      #   本地資料集管理
│   ├── web_controller.py       #   Flask Web 整合控制面板
│   ├── monitor.py              #   整合版主程式入口
│   ├── main.py                 #   純推論主程式入口（無 Web）
│   ├── templates/              #   Web 前端模板
│   │   └── dashboard.html
│   └── static/                 #   Web 靜態資源
│       ├── css/style.css
│       └── js/app.js
│
├── data/                       # 📂 資料集
│   ├── raw/                    #   Khait 原始音檔
│   ├── spectrograms/           #   訓練用頻譜圖
│   ├── spikerbox_raw/          #   SpikerBox 蒐集的原始錄音
│   └── spikerbox_spectrograms/ #   SpikerBox 產生的頻譜圖
│
├── models/                     # 🧠 訓練好的模型
│   └── best_model.tflite       #   量化後的 TFLite 模型
│
├── requirements-pc.txt         # 電腦端套件
└── requirements-pi.txt         # 樹莓派端套件（含 Flask）
```

---

## 🚀 操作手冊

### 1. 環境安裝

#### 電腦端（用於模型訓練）

```bash
# 建立虛擬環境
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Linux / Mac

# 安裝套件
pip install -r requirements-pc.txt
```

#### 樹莓派端

```bash
# 建立虛擬環境
python3 -m venv venv
source venv/bin/activate

# 安裝套件
pip install -r requirements-pi.txt
```

### 2. 模型訓練（電腦端）

```bash
# 步驟 1：將 PlantSounds 資料放入 data/raw/
#         結構：data/raw/Tomato Dry/、data/raw/Empty Pot/ ...

# 步驟 2：執行訓練
python -m src.train

# 步驟 3：匯出 TFLite 模型
python -m src.export_tflite

# 步驟 4：將 models/best_model.tflite 複製到樹莓派
scp models/best_model.tflite pi@<IP>:~/tomato-acoustic-irrigation/models/
```

### 3. 硬體組裝

請按照上方「⚡ 電路接線指南」完成以下步驟：

#### 水泵與繼電器接線
1. **接線前拔掉所有電源**
2. 外部電源供電：將外部 5V（如行動電源）接到繼電器的 **VCC** 與 **COM**。外部 GND 接到繼電器 **GND**。
3. Pi → 繼電器：**不要接 Pi 的 5V**，只接 Pin 9 (GND)→繼電器 **GND** (共地)、Pin 11 (GPIO 17)→繼電器 **IN**。
4. 繼電器 → 水泵：繼電器 **NO**→水泵紅線(+)、水泵黑線(-)→外部電源 **GND**(-)
4. **1N4007 反向並聯**在水泵兩端（銀環接紅線端）
5. 水泵放入水桶（完全泡水），軟管接到盆栽

#### Plant SpikerBox 接線與配置
1. **電源供應**：若使用標準版 SpikerBox，請在背面扣上 **9V 方形電池**並將紅色開關往上撥（綠燈亮起）。
2. **感測器連接**：將附贈的「黃色 RCA 接頭」感測線插進 SpikerBox 右側標示 **INPUT** 的銀色圓孔。
3. **連接植物**：
   - **紅色夾子（訊號極）**：夾在番茄的粗壯主莖上。若番茄太小，可用單芯線輕輕纏繞在莖上，再用夾子夾住單芯線，避免夾傷植物。
   - **黑色夾子（接地極）**：直接插進盆栽泥土裡，作為接地參考以消除環境雜訊。
4. **連接樹莓派**：用一條**支援資料傳輸**的手機充電線（USB-A 轉 Type-C），將 SpikerBox 連上樹莓派的 USB 孔。系統會自動透過 `/dev/ttyACM*`（序列埠）或 USB 音訊介面抓取資料。

### 4. 啟動系統（樹莓派端）

#### 方式 A：完整模式（推薦）— Web 面板 + AI 監測

```bash
cd ~/tomato-acoustic-irrigation
source venv/bin/activate

# 啟動整合版（Web 面板 + 監測功能）
python -m pi.monitor
```

然後用手機打開：`http://<樹莓派IP>:5000`

在儀表板上點「**啟動監測**」按鈕，系統會開始自動迴圈：
> 錄音 → 頻譜圖 → AI 推論 → 判定缺水 → 自動灌溉

#### 方式 B：純 Web 面板（僅手動控制水泵）

```bash
python -m pi.web_controller
```

#### 方式 C：純命令列模式（無 Web，適合自動排程）

```bash
python -m pi.main
```

### 5. Web 控制面板功能

| 功能 | 操作方式 |
|------|---------|
| 手動開啟水泵 | 點綠色「啟動水泵」按鈕 |
| 手動關閉水泵 | 點紅色「停止水泵」按鈕 |
| 定時灌溉 | 設定秒數 → 點「開始定時」 |
| 啟動 AI 自動監測 | 點「啟動監測」→ 系統自動錄音 + 推論 + 灌溉 |
| 停止 AI 自動監測 | 點「停止監測」 |
| 查看即時頻譜圖 | 啟動監測後自動顯示在「即時推論」區塊 |
| 查看推論結果 | 即時推論區塊顯示 noise / normal / thirsty 機率條 |
| 查看活動日誌 | 底部「活動日誌」自動記錄每次操作 |
| 查看資料集統計 | 系統資訊卡顯示已蒐集的資料筆數 |

### 6. API 端點一覽

| 方法 | 路徑 | 說明 |
|------|------|------|
| GET | `/` | 控制面板頁面 |
| GET | `/api/status` | 系統即時狀態 |
| POST | `/api/pump/on` | 手動開啟水泵 |
| POST | `/api/pump/off` | 手動關閉水泵 |
| POST | `/api/pump/cycle` | 定時灌溉（body: `{"duration": 秒數}`） |
| POST | `/api/monitor/start` | 啟動自動監測 |
| POST | `/api/monitor/stop` | 停止自動監測 |
| GET | `/api/inference/latest` | 最新推論結果 + 頻譜圖 (base64) |
| GET | `/api/history` | 推論歷史記錄 |
| GET | `/api/dataset/stats` | 資料集統計 |
| GET | `/api/log` | 活動日誌 |
| POST | `/api/log/clear` | 清除日誌 |

### 7. 資料集蒐集

系統啟動監測後，會自動將每次錄音存到 `data/spikerbox_raw/` 目錄：

```
data/spikerbox_raw/
├── 2026-08-28/
│   └── unlabeled/
│       ├── 194500_123.wav
│       ├── 195000_456.wav
│       └── ...
├── 2026-08-29/
│   └── unlabeled/
│       └── ...
```

頻譜圖會自動存到 `data/spikerbox_spectrograms/`。

累積足夠資料後（建議每類 50~100 筆），可以：
1. 手動將 `unlabeled/` 下的檔案分類到 `normal/` 和 `thirsty/`
2. 用 `python -m src.train` 重新訓練模型

### 8. 設定檔說明

`pi/config.json` 中可調整的參數：

```json
{
    "model_path": "models/best_model.tflite",
    "audio": {
        "sample_rate": 44100,          // 取樣率（Hz）— BYB「USB2 Digital Audio」實測為 44100
        "duration_seconds": 1.0,       // 每次錄音長度
        "listen_interval_seconds": 300  // 監測間隔（秒）
    },
    "spikerbox": {
        "device_keywords": ["SpikerBox", "Backyard", "Digital Audio", "USB Audio"],
        "sample_rate": 44100           // 要跟裝置實際取樣率一致，否則會退回模擬模式
    },
    "dataset": {
        "enabled": true,               // 是否自動儲存資料集
        "save_dir": "data/spikerbox_raw",
        "spectrogram_dir": "data/spikerbox_spectrograms"
    },
    "gpio": {
        "relay_pin": 17,               // 繼電器控制腳位
        "irrigation_duration_seconds": 5 // 自動灌溉時長
    },
    "logic": {
        "thirsty_threshold": 0.85,     // 缺水判定信心度門檻（AI 版）
        "consecutive_thirsty_required": 2 // 連續幾次才觸發灌溉
    },
    "soil": {                          // 土壤濕度自動灌溉（pi/soil_irrigation.py，不用 AI）
        "do_pin": 23,                  // 電阻式模組 DO 接的 GPIO（BCM）
        "power_pin": null,             // 選用：改由此 GPIO 供電，只在讀值時通電以減少腐蝕
        "dry_is_high": true,           // DO 在「乾」時為高電平？不同板子相反就改 false
        "read_interval_seconds": 300,  // 讀感測器間隔
        "consecutive_dry_required": 3, // 連續幾次「乾」才澆水
        "irrigation_duration_seconds": 5, // 每次澆水秒數
        "cooldown_seconds": 1800,      // 澆完後多久內不再澆
        "min_temp_c": 5.0              // 低於此溫度不澆水
    }
}
```

> ⚠️ **取樣率一定要對**：`AudioCapture` 偵測時會用 `sample_rate` 去驗證裝置，
> 若填的值裝置不支援（例如舊設定的 `10000`），偵測會失敗並**自動退回模擬模式**，
> 你看到的頻譜圖就會是模擬音檔而不是 SpikerBox。用 `python -m pi.test_spikerbox`
> 可以確認你這台的實際取樣率。

---

## 🧪 分步測試 / 診斷工具

系統很多零件，建議一個一個確認，不要一次全開。以下工具電腦或樹莓派都能跑，
沒接到硬體時會自動進「模擬模式」，可先看流程。

### A. 測 SpikerBox 有沒有在傳資料 — `pi/test_spikerbox.py`

```bash
python -m pi.test_spikerbox                 # 全自動：掃裝置 + 錄 5 秒 + 存波形圖
python -m pi.test_spikerbox --seconds 10    # 錄久一點
python -m pi.test_spikerbox --device 3      # 指定音訊裝置編號（先跑一次看清單）
python -m pi.test_spikerbox --port COM3     # 指定序列埠（樹莓派用 /dev/ttyACM0）
```

會列出所有音訊/序列埠裝置、自動挑出 SpikerBox、錄一段、印出判讀
（訊號是平的／削波／正常），並把波形圖 + 頻譜圖存到 `pi/diag_output/`。

### B. 網頁看即時頻譜圖 — `pi/spectro_web.py`

```bash
python -m pi.spectro_web
```

不需要模型、不碰 GPIO。啟動後在同一 Wi-Fi 下用手機/電腦打開
`http://<這台的IP>:5000`，畫面每約 1.5 秒更新一次波形 + 頻譜圖(dB)。
頁面頂端會顯示模式：

| 模式 | 意思 |
|------|------|
| `audio` | 有抓到 USB 音訊裝置（正常） |
| `serial` | 走序列埠的新版 SpikerBox |
| `mock` | ⚠️ 沒抓到硬體，正在放模擬音檔 → 回去跑 test_spikerbox 檢查 |

### C. 測感測器 — `pi/sensors.py`

```bash
python -m pi.sensors        # 連讀 5 次 DS18B20 溫度 + 電阻式土壤濕度（DO 模式）
```

### D. 純門檻自動灌溉（不用 AI）— `pi/soil_irrigation.py`

```bash
python -m pi.soil_irrigation
```

連續 N 次讀到「土壤乾」就啟動水泵幾秒，然後進入冷卻期。
參數在 `pi/config.json` 的 `"soil"` 區塊。這是最快能做出「會自動澆水」的路徑，
跟聲學 / AI 完全獨立，兩者可並行。接線與感測器校正見「感測器接線」一節。

### 感測器接線（DS18B20 + 電阻式土壤濕度）

```
DS18B20（溫度，1-Wire）
    VCC  → 3.3V (Pin 1)
    GND  → GND  (Pin 6)
    DATA → GPIO 4 (Pin 7)
    ★ DATA 與 3.3V 之間接 4.7kΩ 上拉電阻（必要）
    ★ 先啟用：sudo raspi-config → Interface Options → 1-Wire → Enable → 重開機

電阻式土壤濕度模組（數位 DO 模式，樹莓派沒有 ADC 不能讀 AO）
    VCC → 3.3V   ★ 不要接 5V！接 5V 時 DO 會輸出 5V 打壞 GPIO
    GND → GND
    DO  → GPIO 23 (Pin 16)   ← 可在 config.json 的 soil.do_pin 改
    AO  → 不接
    校正：把探頭插進「該澆水了」的乾土，轉模組上的藍色可調電阻到狀態 LED 剛好切換，
          再跑 python -m pi.sensors 確認乾土顯示「缺水」。乾濕相反就把 dry_is_high 改 false。
```

> 電阻式感測器長期泡土會電解腐蝕。想延壽：VCC 改接一隻 GPIO（例如 24），
> 設 `soil.power_pin: 24`，程式只在讀值前通電 2 秒。之後預算夠再換
> **電容式感測器 + MCP3008 / ADS1115（ADC）**。

---

## ⚠️ 安全注意事項

| 規則 | 原因 |
|------|------|
| 接線前**拔掉所有電源** | 避免短路 |
| 1N4007 **方向不能接反** | 接反會短路燒毀！銀環接紅線端 |
| 12V 不能接到 Pi 的任何腳位 | 會燒毀 Pi |
| 水泵**一定要泡水中**才能運轉 | 乾燒會壞 |
| 水泵附近接線做**防水處理** | 電工膠帶包好 |
| 第一次測試先**不接水管** | 確認繼電器有「喀」聲就好 |

---

## 📊 訓練資料集

### Khait 植物聲學資料集

本專案使用 [Khait et al. (2023)](https://doi.org/10.1016/j.cell.2023.03.009) 公開釋出的資料集作為基礎模型訓練：

| 類別 | 數量 | 說明 |
|------|------|------|
| Tomato Dry | 1,622 | 番茄缺水（氣穴聲） |
| Tomato Cut | 660 | 番茄被切斷 |
| Empty Pot | 1,036 | 空盆（對照組） |
| Greenhouse Noises | 1,378 | 溫室環境噪音 |

### SpikerBox 自建資料集

系統運行時自動蒐集的資料存在 `data/spikerbox_raw/`。
累積後可重新訓練專用模型，提升辨識準確度。

---

## 📚 參考文獻

1. Khait, I., et al. (2023). "Sounds emitted by plants under stress are airborne and informative." *Cell*, 186(7).
2. Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications."
3. Boll, S. (1979). "Suppression of acoustic noise in speech using spectral subtraction." *IEEE TASSP*.
4. TensorFlow Lite Documentation - Model Optimization.
5. Backyard Brains - Plant SpikerBox Documentation.

## 📄 授權條款

本專案採用 [MIT License](LICENSE) 授權。
