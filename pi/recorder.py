"""
連續錄音器 — 長期資料蒐集專用
================================
只做一件事：定時從 SpikerBox 錄音並存檔。

**不做 AI 推論、不碰 GPIO、不會澆水。**
適合「連續 N 天不澆水」這種實驗，不用擔心系統自己把水澆下去毀掉實驗。

跟 web_controller 的自動監測差在哪：
  * 自動監測每次只錄 1 秒 → 這支預設錄 5 秒（時間軸切格子才有意義）
  * 自動監測會跑 CNN 推論並可能觸發水泵 → 這支完全不碰
  * 這支額外產生一份 CSV，記錄每筆的訊號品質（RMS / 峰值頻率 / 削波率）

那份 CSV 很重要：5 天下來如果電極鬆脫、接觸變差，可以從 CSV 看出
從哪個時間點開始訊號品質崩掉，把那段資料剔除，避免污染範本。

用法：
    python -m pi.recorder                        # 每 5 分鐘錄 5 秒，直到 Ctrl+C
    python -m pi.recorder --interval 600          # 改成每 10 分鐘錄一次
    python -m pi.recorder --duration 10           # 每次錄 10 秒
    python -m pi.recorder --days 5                # 錄滿 5 天自動停止
    python -m pi.recorder --label normal          # 直接存到 normal/ 而非 unlabeled/
    python -m pi.recorder --note "day0 澆水後"    # 在 CSV 備註欄留紀錄

檔案會存到：
    data/spikerbox_raw/<日期>/<label>/<時間>.wav
    data/spikerbox_raw/<日期>/recording_log.csv
"""

import os
import sys
import csv
import json
import time
import shutil
import signal
import argparse
import threading
from datetime import datetime

import numpy as np
from scipy.io.wavfile import read as wav_read

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from pi.audio_capture import AudioCapture


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── 預設參數 ───────────────────────────────────────
DEFAULT_DURATION = 5.0        # 每次錄幾秒
DEFAULT_INTERVAL = 300.0      # 每隔幾秒錄一次（300 = 5 分鐘）
DEFAULT_LABEL = "unlabeled"

# 收到 Ctrl+C 或 systemd 的 SIGTERM 時設起來，讓迴圈乾淨收尾
stop_event = threading.Event()


def _safe_print(msg: str):
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode("ascii"), flush=True)


def _handle_signal(signum, frame):
    _safe_print(f"\n[RECORDER] 收到停止訊號（{signum}），正在收尾...")
    stop_event.set()


def load_config():
    cfg_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── 訊號品質統計（寫進 CSV，事後篩資料用）──────────
def signal_stats(wav_path: str) -> dict:
    """讀回剛錄好的 WAV，算 RMS / 峰值頻率 / 削波比例。"""
    sr, data = wav_read(wav_path)
    x = np.asarray(data, dtype=np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        x = x / float(np.iinfo(data.dtype).max)

    if x.size == 0:
        return {"sample_rate": sr, "rms": 0.0, "peak_hz": 0.0, "clip_pct": 0.0}

    xw = x - float(np.mean(x))
    rms = float(np.sqrt(np.mean(xw ** 2)))
    clip_pct = float(np.mean(np.abs(x) > 0.98)) * 100.0

    spec = np.abs(np.fft.rfft(xw))
    freqs = np.fft.rfftfreq(len(xw), 1.0 / sr)
    band = freqs > 1.0          # 忽略直流
    peak_hz = float(freqs[band][np.argmax(spec[band])]) if band.any() else 0.0

    return {"sample_rate": int(sr), "rms": rms, "peak_hz": peak_hz, "clip_pct": clip_pct}


#  相對基準的漂移門檻：掉到基準的一半以下 / 漲到兩倍以上就示警
RMS_DROP_RATIO = 0.5
RMS_JUMP_RATIO = 2.0
BASELINE_SAMPLES = 10          # 用前幾筆的中位數當基準


def quality_flag(st: dict, baseline_rms: float = None) -> str:
    """
    把統計數字翻成一個好讀的品質標記，方便事後 grep。

    判斷順序（由嚴重到輕微）：
      1. CLIPPED / TOO_WEAK  — 絕對門檻，一定是壞的
      2. RMS_DROP / RMS_JUMP — 相對「本次錄音基準」的漂移
      3. HUM_HEAVY           — 市電雜訊主導（環境問題，資料通常還能用）

    為什麼要有第 2 項：
      絕對門檻抓不到「訊號正在死掉」。實測遇過電池快沒電時 RMS 從 0.154
      掉到 0.037（剩 1/4），但因為還沒低於 TOO_WEAK 的 0.005、主頻也剛好
      跑出 45~65Hz 的哼聲範圍，結果被標成「OK」—— 訊號在死掉卻顯示正常，
      非常誤導。改成跟「這次錄音自己的基準」比，才抓得到這種衰退。
    """
    rms = st["rms"]

    # 1. 絕對門檻：硬性失效
    if st["clip_pct"] > 1.0:
        return "CLIPPED"        # 爆表，電極亂跳或增益太大
    if rms < 0.005:
        return "TOO_WEAK"       # 太弱，接觸不良或裝置掉線

    # 2. 相對基準的漂移：抓電池衰退、電極鬆脫
    if baseline_rms and baseline_rms > 0:
        ratio = rms / baseline_rms
        if ratio < RMS_DROP_RATIO:
            return "RMS_DROP"   # 訊號正在衰退（電池沒電？電極鬆了？）
        if ratio > RMS_JUMP_RATIO:
            return "RMS_JUMP"   # 突然暴增（電極碰到東西？接觸改變？）

    # 3. 環境雜訊：不是故障，只是提醒接地可以更好
    if 45 <= st["peak_hz"] <= 65 and rms > 0.1:
        return "HUM_HEAVY"      # 市電哼聲主導，接地沒做好

    return "OK"


# ─── CSV 記錄 ───────────────────────────────────────
CSV_FIELDS = ["timestamp", "filename", "label", "duration_s", "sample_rate",
              "rms", "rms_ratio", "peak_hz", "clip_pct", "quality", "mode", "note"]


def append_csv(csv_path: str, row: dict):
    """
    附加一列到 CSV。若既有檔案的欄位跟現在的 CSV_FIELDS 不一致
    （例如程式更新過、新增了欄位），先把舊檔改名備份再開新的，
    避免新舊欄位混在同一個檔案裡導致欄位錯位。
    """
    if os.path.isfile(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8") as fh:
                old_header = fh.readline().strip().split(",")
            if old_header != CSV_FIELDS:
                backup = csv_path + ".old"
                os.replace(csv_path, backup)
                _safe_print(f"[RECORDER] CSV 欄位已更新，舊檔備份為 {os.path.basename(backup)}")
        except Exception:
            pass

    is_new = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)


# ─── 主迴圈 ─────────────────────────────────────────
def run(duration: float, interval: float, label: str, note: str,
        days: float, save_root: str):
    cfg = load_config()
    sb = cfg.get("spikerbox", {})

    audio = AudioCapture(
        sample_rate=sb.get("sample_rate", cfg["audio"]["sample_rate"]),
        duration=duration,
        device_keywords=sb.get("device_keywords", ["SpikerBox", "Backyard", "USB Audio"]),
        fallback_device=sb.get("fallback_device"),
    )

    if audio.mode == AudioCapture.MODE_MOCK:
        _safe_print("")
        _safe_print("⚠️  警告：沒有抓到 SpikerBox，現在是「模擬模式」！")
        _safe_print("    錄下來的不是真實資料，拿去建範本沒有意義。")
        _safe_print("    請先確認 ls /dev/ttyACM* 有東西，再重跑一次。")
        _safe_print("")

    deadline = time.time() + days * 86400 if days else None
    tmp_wav = os.path.join(os.path.dirname(__file__), "recorder_tmp.wav")
    count = 0
    bad_count = 0
    started = datetime.now()

    # 基準 RMS：用開頭幾筆（排除硬性失效的）的中位數，之後拿來偵測漂移。
    # 用中位數而不是平均，避免某一兩筆異常值把基準拉歪。
    baseline_rms = None
    baseline_pool = []

    try:
        while not stop_event.is_set():
            if deadline and time.time() >= deadline:
                _safe_print(f"\n[RECORDER] 已達設定時長（{days} 天），自動停止。")
                break

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H%M%S_%f")[:-3]

            target_dir = os.path.join(save_root, date_str, label)
            os.makedirs(target_dir, exist_ok=True)
            target_path = os.path.join(target_dir, f"{time_str}.wav")
            csv_path = os.path.join(save_root, date_str, "recording_log.csv")

            try:
                ok = audio.record_audio(tmp_wav)
                if ok and os.path.isfile(tmp_wav):
                    # 先錄到暫存再搬過去，避免資料夾裡出現寫到一半的破檔。
                    # 用 shutil.move 而不是 os.replace：save_root 若在外接
                    # 隨身碟（跨檔案系統），os.replace 會丟 WinError 17 / EXDEV。
                    shutil.move(tmp_wav, target_path)
                    st = signal_stats(target_path)
                    flag = quality_flag(st, baseline_rms)

                    # 還沒建立基準時，收集開頭幾筆沒有硬性失效的 RMS
                    if baseline_rms is None and flag not in ("CLIPPED", "TOO_WEAK"):
                        baseline_pool.append(st["rms"])
                        if len(baseline_pool) >= BASELINE_SAMPLES:
                            baseline_rms = float(np.median(baseline_pool))
                            _safe_print(
                                f"[RECORDER] 已建立基準 RMS = {baseline_rms:.4f}"
                                f"（前 {BASELINE_SAMPLES} 筆的中位數）。"
                                f"之後掉到 {baseline_rms*RMS_DROP_RATIO:.4f} 以下會標 RMS_DROP"
                            )

                    rms_ratio = (st["rms"] / baseline_rms) if baseline_rms else ""
                    if flag != "OK":
                        bad_count += 1
                    count += 1

                    append_csv(csv_path, {
                        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                        "filename": os.path.basename(target_path),
                        "label": label,
                        "duration_s": duration,
                        "sample_rate": st["sample_rate"],
                        "rms": round(st["rms"], 6),
                        "rms_ratio": round(rms_ratio, 3) if rms_ratio != "" else "",
                        "peak_hz": round(st["peak_hz"], 1),
                        "clip_pct": round(st["clip_pct"], 2),
                        "quality": flag,
                        "mode": audio.mode,
                        "note": note,
                    })

                    # 嚴重問題用 ✗，輕微提醒用 ⚠，正常用 ✓
                    mark = {"CLIPPED": "✗", "TOO_WEAK": "✗",
                            "RMS_DROP": "✗", "RMS_JUMP": "✗"}.get(flag,
                           "⚠" if flag != "OK" else "✓")
                    ratio_str = f" ratio={rms_ratio:.2f}" if rms_ratio != "" else ""
                    elapsed = (now - started)
                    _safe_print(
                        f"{mark} [{count:5d}] {now.strftime('%m-%d %H:%M:%S')} "
                        f"RMS={st['rms']:.4f}{ratio_str} peak={st['peak_hz']:6.1f}Hz "
                        f"clip={st['clip_pct']:4.1f}% {flag:10s} "
                        f"(已錄 {elapsed.days}天{elapsed.seconds//3600}時)"
                    )
                else:
                    _safe_print(f"✗ [{now.strftime('%H:%M:%S')}] 錄音失敗，跳過這輪")
            except Exception as e:
                _safe_print(f"✗ [{now.strftime('%H:%M:%S')}] 錯誤: {e}")

            stop_event.wait(interval)

    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except Exception:
                pass
        try:
            audio.close()
        except Exception:
            pass

        elapsed = datetime.now() - started
        _safe_print("")
        _safe_print("=" * 58)
        _safe_print(f"[RECORDER] 結束。共錄 {count} 筆，其中 {bad_count} 筆品質有問題。")
        _safe_print(f"           執行時間: {elapsed.days} 天 "
                    f"{elapsed.seconds//3600} 小時 {(elapsed.seconds//60)%60} 分")
        _safe_print(f"           資料位置: {save_root}")
        _safe_print("=" * 58)


def main():
    ap = argparse.ArgumentParser(description="連續錄音器（純錄音，不碰 AI 和水泵）")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                    help=f"每次錄幾秒（預設 {DEFAULT_DURATION}）")
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                    help=f"每隔幾秒錄一次（預設 {DEFAULT_INTERVAL} = 5 分鐘）")
    ap.add_argument("--label", default=DEFAULT_LABEL,
                    help=f"存到哪個子資料夾（預設 {DEFAULT_LABEL}）")
    ap.add_argument("--note", default="", help="備註，會寫進 CSV")
    ap.add_argument("--days", type=float, default=0,
                    help="錄滿幾天自動停止（0 = 一直錄到手動停止）")
    ap.add_argument("--save-root", default=None, help="資料根目錄")
    args = ap.parse_args()

    cfg = load_config()
    save_root = args.save_root or os.path.join(
        PROJECT_ROOT, cfg["dataset"]["save_dir"])

    signal.signal(signal.SIGINT, _handle_signal)
    try:
        signal.signal(signal.SIGTERM, _handle_signal)   # systemctl stop 用
    except Exception:
        pass

    per_file_kb = args.duration * cfg.get("spikerbox", {}).get("sample_rate", 10000) * 2 / 1024
    per_day = 86400 / args.interval

    _safe_print("=" * 58)
    _safe_print("[RECORDER] 連續錄音器 — 純錄音，不做推論、不碰水泵")
    _safe_print("=" * 58)
    _safe_print(f"  每次錄音長度 : {args.duration} 秒")
    _safe_print(f"  錄音間隔     : {args.interval} 秒（每天約 {per_day:.0f} 筆）")
    _safe_print(f"  存到          : {os.path.join(save_root, '<日期>', args.label)}")
    _safe_print(f"  預估空間     : 每筆約 {per_file_kb:.0f} KB，每天約 "
                f"{per_file_kb * per_day / 1024:.0f} MB")
    if args.days:
        _safe_print(f"  自動停止     : {args.days} 天後")
    if args.note:
        _safe_print(f"  備註          : {args.note}")
    _safe_print("=" * 58)
    _safe_print("  按 Ctrl+C 停止\n")

    run(args.duration, args.interval, args.label, args.note, args.days, save_root)


if __name__ == "__main__":
    main()
