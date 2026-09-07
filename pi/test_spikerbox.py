"""
Plant SpikerBox 診斷腳本
========================
這支程式跟主系統完全獨立，不會 import src/，也不需要模型。
目的只有一個：確認 SpikerBox 到底有沒有在傳資料、傳的是什麼。

用法（電腦或樹莓派都可以跑）：

    python -m pi.test_spikerbox                 # 全自動：掃裝置 + 錄 5 秒 + 存波形圖
    python -m pi.test_spikerbox --seconds 10    # 錄 10 秒
    python -m pi.test_spikerbox --device 3      # 指定音訊裝置編號（先跑一次看清單）
    python -m pi.test_spikerbox --rate 44100    # 指定取樣率
    python -m pi.test_spikerbox --port COM3     # 指定序列埠（樹莓派用 /dev/ttyACM0）
    python -m pi.test_spikerbox --audio-only    # 只測 USB 音訊，跳過序列埠
    python -m pi.test_spikerbox --serial-only   # 只測序列埠，跳過 USB 音訊

輸出檔案會存在  pi/diag_output/  底下：
    spikerbox_test.wav   錄到的原始聲音
    spikerbox_test.png   波形圖 + 頻譜圖（用來看訊號長什麼樣）
"""

import argparse
import os
import sys
import time

import numpy as np

# Windows 終端機預設 cp950，中文會變亂碼；強制 stdout 用 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# --- 選用套件，缺了也能跑（只是功能少一點） ---
try:
    import sounddevice as sd
    HAS_SD = True
except Exception:
    HAS_SD = False

try:
    import serial
    import serial.tools.list_ports as list_ports
    HAS_SERIAL = True
except Exception:
    HAS_SERIAL = False

try:
    from scipy.io.wavfile import write as wav_write
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

try:
    import matplotlib
    matplotlib.use("Agg")            # 不開視窗，直接存檔
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False


DEVICE_KEYWORDS = ["spikerbox", "backyard", "digital audio", "usb audio", "usb2"]
OUT_DIR = os.path.join(os.path.dirname(__file__), "diag_output")


def p(msg=""):
    """跨平台安全 print（避免 Windows 終端機編碼錯誤）"""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode("ascii"))


def hr(title=""):
    p("\n" + "=" * 60)
    if title:
        p(title)
        p("=" * 60)


# ======================================================================
# 1. 列出所有裝置
# ======================================================================
def list_all_devices():
    hr("步驟 1：目前這台電腦看得到哪些裝置")

    if HAS_SERIAL:
        ports = list(list_ports.comports())
        if ports:
            p("[序列埠] 找到 %d 個：" % len(ports))
            for pt in ports:
                p("   %-12s  %s" % (pt.device, pt.description))
        else:
            p("[序列埠] 沒有找到任何序列埠")
    else:
        p("[序列埠] 未安裝 pyserial，略過（pip install pyserial）")

    if HAS_SD:
        p("")
        devs = sd.query_devices()
        p("[音訊裝置] 找到 %d 個（只有 in>0 的才能錄音）：" % len(devs))
        for i, d in enumerate(devs):
            mark = "  <== 可能是 SpikerBox" if _looks_like_spikerbox(d["name"]) and d["max_input_channels"] > 0 else ""
            p("   [%2d] in=%d out=%d  %6.0f Hz  %s%s" % (
                i, d["max_input_channels"], d["max_output_channels"],
                d["default_samplerate"], d["name"], mark))
    else:
        p("[音訊裝置] 未安裝 sounddevice，略過（pip install sounddevice）")


def _looks_like_spikerbox(name):
    low = name.lower()
    return any(k in low for k in DEVICE_KEYWORDS)


# ======================================================================
# 2. 序列埠測試
# ======================================================================
def test_serial(port, bauds):
    hr("步驟 2：測試序列埠（新版 USB-C Plant SpikerBox 走這條）")

    if not HAS_SERIAL:
        p("未安裝 pyserial，跳過")
        return

    candidates = [port] if port else [pt.device for pt in list_ports.comports()]
    if not candidates:
        p("沒有序列埠可測，跳過")
        return

    for pt in candidates:
        for baud in bauds:
            try:
                p("嘗試  %s  @ %d baud ..." % (pt, baud))
                s = serial.Serial(pt, baud, timeout=2)
                time.sleep(0.3)
                s.reset_input_buffer()
                data = s.read(512)
                s.close()
                if len(data) > 50:
                    hi = sum(1 for b in data if b & 0x80)
                    p("   收到 %d bytes（其中 %d 個是高位元組）" % (len(data), hi))
                    p("   >>> 這個埠有在傳資料。若要用序列埠模式，config 的 baud 設 %d" % baud)
                    return
                else:
                    p("   只收到 %d bytes，這個組合大概不對" % len(data))
            except Exception as e:
                p("   打不開：%s" % e)
    p(">>> 序列埠都沒讀到有效資料。你的 SpikerBox 多半是走 USB 音訊（看步驟 3）")


# ======================================================================
# 3. USB 音訊錄音測試
# ======================================================================
def pick_audio_device(device_arg):
    if device_arg is not None:
        return int(device_arg)
    devs = sd.query_devices()
    for i, d in enumerate(devs):
        if d["max_input_channels"] > 0 and _looks_like_spikerbox(d["name"]):
            return i
    # 找不到就用系統預設輸入
    default_in = sd.default.device[0]
    if default_in is not None and default_in >= 0:
        return int(default_in)
    return None


def test_audio(device_arg, rate_arg, seconds):
    hr("步驟 3：測試 USB 音訊錄音")

    if not HAS_SD:
        p("未安裝 sounddevice，跳過")
        return None

    dev_id = pick_audio_device(device_arg)
    if dev_id is None:
        p(">>> 找不到任何可錄音的裝置。確認 SpikerBox 有插好、開關打開、綠燈亮")
        return None

    info = sd.query_devices(dev_id)
    p("使用裝置：[%d] %s" % (dev_id, info["name"]))

    # 決定取樣率：優先用參數，其次試裝置預設，再試常見值
    if rate_arg:
        rates_to_try = [int(rate_arg)]
    else:
        rates_to_try = [int(info["default_samplerate"]), 10000, 44100, 48000]

    rate = None
    for r in rates_to_try:
        try:
            sd.check_input_settings(device=dev_id, samplerate=r, channels=1)
            rate = r
            break
        except Exception:
            continue
    if rate is None:
        p(">>> 這個裝置不接受單聲道錄音，可能選錯裝置了")
        return None

    p("取樣率：%d Hz   錄音長度：%d 秒" % (rate, seconds))
    p("錄音中...（如果要看訊號變化，現在去碰一下植物的葉子）")

    n = int(rate * seconds)
    try:
        rec = sd.rec(n, samplerate=rate, channels=1, dtype="float32", device=dev_id)
        sd.wait()
    except Exception as e:
        p(">>> 錄音失敗：%s" % e)
        return None

    x = rec.reshape(-1).astype(np.float32)
    analyze(x, rate)
    save_outputs(x, rate)
    return x, rate


# ======================================================================
# 4. 分析 + 判讀
# ======================================================================
def analyze(x, rate):
    hr("步驟 4：訊號分析")

    n = len(x)
    dur = n / rate
    xmin, xmax = float(x.min()), float(x.max())
    mean = float(x.mean())
    std = float(x.std())
    rms = float(np.sqrt(np.mean(x ** 2)))
    clip_ratio = float(np.mean(np.abs(x) > 0.98))

    p("樣本數        : %d" % n)
    p("實際長度      : %.2f 秒" % dur)
    p("數值範圍      : %.4f  ~  %.4f   (音訊滿刻度是 -1.0 ~ 1.0)" % (xmin, xmax))
    p("平均值            : %.5f" % mean)
    p("標準差(振幅) : %.5f" % std)
    p("RMS          : %.5f" % rms)
    p("削波比例      : %.1f%%   (打到滿刻度的樣本佔比)" % (clip_ratio * 100))

    # 頻譜：找主頻 + 看 50/60Hz 電源哼聲
    xw = x - mean
    spec = np.abs(np.fft.rfft(xw))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    band = freqs > 1.0                      # 忽略直流
    if band.any():
        peak_f = float(freqs[band][np.argmax(spec[band])])
        p("主要頻率      : %.1f Hz" % peak_f)
        for hum in (50.0, 60.0):
            if freqs[-1] >= hum:
                idx = np.argmin(np.abs(freqs - hum))
                lvl = spec[idx] / (np.median(spec[band]) + 1e-9)
                if lvl > 8:
                    p("   ⚠ %.0f Hz 電源哼聲很強（x%.0f）→ 黑夾子要確實插進土裡接地" % (hum, lvl))

    # ---- 判讀 ----
    hr("判讀")
    if std < 1e-4:
        p("❌ 訊號幾乎是一條平線。")
        p("   → 裝置沒在傳資料，或錄到的是靜音。")
        p("   → 檢查：USB 是資料線嗎？開關開了嗎？綠燈亮嗎？選對裝置編號嗎？")
    elif clip_ratio > 0.05:
        p("⚠ 訊號一直打到滿刻度（削波）。")
        p("   → 最常見原因：電極還沒夾到植物，輸入浮空亂跳。")
        p("   → 把紅夾子夾到莖（可先纏一小段電線再夾），黑夾子插進盆土，再錄一次。")
        p("   → 如果已經夾好還是這樣，SpikeRecorder 裡把增益(左邊 +/- 圓鈕)調小。")
    else:
        p("✅ 有起伏、沒有一直削波，看起來像真的在收訊號。")
        p("   → 打開 pi/diag_output/spikerbox_test.png 看波形：")
        p("     生物電位應該是『緩慢漂移 + 偶爾跳動』，不是整片白噪音。")
        p("     碰葉子/夾莖的前後如果波形有變，就對了。")


def save_outputs(x, rate):
    os.makedirs(OUT_DIR, exist_ok=True)

    wav_path = os.path.join(OUT_DIR, "spikerbox_test.wav")
    if HAS_SCIPY:
        i16 = np.clip(x * 32767, -32768, 32767).astype(np.int16)
        wav_write(wav_path, rate, i16)
        p("\n已存錄音 : %s" % wav_path)
    else:
        p("\n（未安裝 scipy，跳過存 WAV）")

    if HAS_PLT:
        png_path = os.path.join(OUT_DIR, "spikerbox_test.png")
        t = np.arange(len(x)) / rate
        xw = x - x.mean()
        spec = np.abs(np.fft.rfft(xw))
        freqs = np.fft.rfftfreq(len(x), 1.0 / rate)

        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        ax[0].plot(t, x, lw=0.5)
        ax[0].set_title("Waveform")
        ax[0].set_xlabel("time (s)")
        ax[0].set_ylabel("amplitude")
        ax[0].set_ylim(-1.05, 1.05)

        ax[1].semilogy(freqs[1:], spec[1:] + 1e-9, lw=0.5)
        ax[1].set_title("Spectrum")
        ax[1].set_xlabel("frequency (Hz)")
        ax[1].set_ylabel("magnitude (log)")

        fig.tight_layout()
        fig.savefig(png_path, dpi=110)
        plt.close(fig)
        p("已存波形圖 : %s" % png_path)
    else:
        p("（未安裝 matplotlib，跳過存波形圖）pip install matplotlib")


# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Plant SpikerBox 診斷工具")
    ap.add_argument("--seconds", type=float, default=5.0, help="錄音秒數（預設 5）")
    ap.add_argument("--device", default=None, help="音訊裝置編號（先跑一次看清單）")
    ap.add_argument("--rate", default=None, help="取樣率 Hz（預設自動）")
    ap.add_argument("--port", default=None, help="序列埠，例如 COM3 或 /dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=None, help="序列埠 baud（預設自動試幾個）")
    ap.add_argument("--audio-only", action="store_true", help="只測 USB 音訊")
    ap.add_argument("--serial-only", action="store_true", help="只測序列埠")
    args = ap.parse_args()

    hr("Plant SpikerBox 診斷開始")
    p("Python      : %s" % sys.version.split()[0])
    p("平台        : %s" % sys.platform)
    p("sounddevice : %s" % ("OK" if HAS_SD else "未安裝"))
    p("pyserial    : %s" % ("OK" if HAS_SERIAL else "未安裝"))
    p("matplotlib  : %s" % ("OK" if HAS_PLT else "未安裝"))

    list_all_devices()

    bauds = [args.baud] if args.baud else [230400, 222222, 500000, 115200]
    if not args.audio_only:
        test_serial(args.port, bauds)
    if not args.serial_only:
        test_audio(args.device, args.rate, int(args.seconds))

    hr("診斷結束")
    p("把 pi/diag_output/ 底下的 .png 和上面的判讀貼給我，我幫你看下一步。")


if __name__ == "__main__":
    main()
