"""
色塊分布相似度比對 — 不需要 CNN 模型的輕量分類
================================================
教授建議的方法：把頻譜圖切成「時間 × 頻率」的格子（色塊），
每一格算一個平均能量 → 得到一張「色塊分布」。
先用已知狀況的錄音建立範本，新訊號進來就比對它跟哪個範本最像。

跟 CNN 的差別：
  * 不需要 250kHz 的 Khait 資料，直接吃 SpikerBox 的 10kHz
  * 不需要 GPU 訓練，每類錄十幾筆就能起步
  * 看得懂為什麼這樣判（色塊數字可以印出來直接對照）

用法：
    python -m pi.similarity_classifier check           # 看目前有哪些已標記資料
    python -m pi.similarity_classifier build           # 用已標記資料建立範本
    python -m pi.similarity_classifier test <wav檔>    # 測單一檔案屬於哪一類

資料要放這裡（沿用系統原本就在存的資料夾）：
    data/spikerbox_raw/<日期>/normal/*.wav     ← 正常（澆足水）
    data/spikerbox_raw/<日期>/thirsty/*.wav    ← 缺水
    data/spikerbox_raw/<日期>/unlabeled/*.wav  ← 還沒分類，會被略過

⚠️ 標籤一定要用「跟 SpikerBox 無關的方法」決定 —— 土壤濕度感測器讀數、
   肉眼看葉片有沒有萎凋、停水幾天。不能用聲音自己判斷自己，那是循環論證。
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
from scipy.io.wavfile import read as wav_read
from scipy.signal import spectrogram as sp_spectrogram, iirnotch, filtfilt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


# ─── 參數 ───────────────────────────────────────────
N_FREQ_BANDS = 6        # 頻率方向切幾格
N_TIME_SLICES = 8       # 時間方向切幾格
MAX_FREQ = 500.0        # 只看 0~500Hz（植物電位訊號都在低頻）
MAINS_HZ = 60.0         # 市電頻率（台灣 60Hz，日本東部/歐洲是 50）
NOTCH_HARMONICS = 3     # 連同 2 倍、3 倍頻一起濾掉
APPLY_NOTCH = True      # 是否濾除市電雜訊

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "spikerbox_raw")
DEFAULT_PROFILE_PATH = os.path.join(os.path.dirname(__file__), "reference_profiles.json")

# 只有這些資料夾名稱會被當成「已標記」，其他（unlabeled 等）一律略過
VALID_LABELS = ("normal", "thirsty", "noise")


def _safe_print(msg: str):
    try:
        print(msg)
    except UnicodeEncodeError:
        print(str(msg).encode("ascii", "replace").decode("ascii"))


# ─── 音訊讀取與前處理 ───────────────────────────────
def load_wav(path: str):
    """讀 WAV → (單聲道 float32 [-1,1], 取樣率)。"""
    sr, data = wav_read(path)
    x = np.asarray(data, dtype=np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        x = x / float(np.iinfo(data.dtype).max)
    elif x.size and np.max(np.abs(x)) > 1.5:
        x = x / 32768.0
    return x, sr


def remove_mains_hum(x: np.ndarray, sr: int,
                     mains_hz: float = MAINS_HZ,
                     harmonics: int = NOTCH_HARMONICS) -> np.ndarray:
    """
    用 notch filter 濾掉市電雜訊（含諧波）。

    為什麼要濾：電極接觸不良時 60Hz 哼聲會非常大（我們實測過 RMS 0.8、
    削波 55%），那是環境雜訊不是植物訊號，不濾掉的話色塊分布會被它主導，
    兩種狀況的差異全部被蓋掉。
    """
    y = x.copy()
    nyq = sr / 2.0
    for k in range(1, harmonics + 1):
        f0 = mains_hz * k
        if f0 >= nyq * 0.95:
            break
        try:
            b, a = iirnotch(w0=f0 / nyq, Q=30.0)
            y = filtfilt(b, a, y)
        except Exception:
            pass  # 濾波失敗就保持原樣，不讓整個流程掛掉
    return y


# ─── 核心：把頻譜圖變成色塊分布 ──────────────────────
def spectro_blocks(x: np.ndarray, sr: int,
                   n_freq: int = N_FREQ_BANDS,
                   n_time: int = N_TIME_SLICES,
                   max_freq: float = MAX_FREQ,
                   apply_notch: bool = APPLY_NOTCH) -> np.ndarray:
    """
    把一段音訊轉成「色塊分布」矩陣，形狀 (n_freq, n_time)。

    步驟：
      1. 濾除市電雜訊
      2. 算頻譜圖（dB）
      3. 只取 0~max_freq 這段
      4. 切成 n_freq × n_time 個方格，每格取平均
      5. 做 z-score 正規化（減平均、除標準差）

    第 5 步很重要：不同次錄音的電極接觸鬆緊會讓整體音量差很多，
    正規化之後比較的才是「分布形狀」而不是「誰比較大聲」。
    """
    if len(x) < 64:
        raise ValueError(f"音訊太短（{len(x)} 個 sample），無法分析")

    if apply_notch:
        x = remove_mains_hum(x, sr)

    xw = x - float(np.mean(x))
    nperseg = min(1024, max(64, len(xw) // (n_time * 2)))
    noverlap = int(nperseg * 0.5)
    f, t, S = sp_spectrogram(xw, fs=sr, nperseg=nperseg, noverlap=noverlap)
    S_db = 10.0 * np.log10(S + 1e-12)

    # 只保留感興趣的頻段
    keep = f <= min(max_freq, sr / 2.0)
    if keep.sum() < n_freq:
        keep = np.ones_like(f, dtype=bool)   # 頻率解析度不夠就用全部
    S_db = S_db[keep, :]

    if S_db.shape[1] < n_time:
        # 時間框太少（音訊很短），用重複補到夠
        reps = int(np.ceil(n_time / max(1, S_db.shape[1])))
        S_db = np.repeat(S_db, reps, axis=1)

    # 切格子取平均
    freq_edges = np.linspace(0, S_db.shape[0], n_freq + 1).astype(int)
    time_edges = np.linspace(0, S_db.shape[1], n_time + 1).astype(int)
    grid = np.zeros((n_freq, n_time), dtype=np.float64)
    for i in range(n_freq):
        f0, f1 = freq_edges[i], max(freq_edges[i + 1], freq_edges[i] + 1)
        for j in range(n_time):
            t0, t1 = time_edges[j], max(time_edges[j + 1], time_edges[j] + 1)
            grid[i, j] = float(np.mean(S_db[f0:f1, t0:t1]))

    # z-score 正規化：比形狀，不比音量
    std = float(np.std(grid))
    grid = (grid - float(np.mean(grid))) / (std if std > 1e-9 else 1.0)
    return grid


def wav_to_blocks(path: str) -> np.ndarray:
    x, sr = load_wav(path)
    return spectro_blocks(x, sr)


# ─── 建立範本 ───────────────────────────────────────
def scan_labeled_files(data_root: str = DEFAULT_DATA_ROOT) -> dict:
    """掃描 data/spikerbox_raw/<日期>/<label>/*.wav，回傳 {label: [檔案路徑...]}。"""
    found = {}
    if not os.path.isdir(data_root):
        return found
    for label in VALID_LABELS:
        files = sorted(glob.glob(os.path.join(data_root, "*", label, "*.wav")))
        if files:
            found[label] = files
    return found


def build_profiles(data_root: str = DEFAULT_DATA_ROOT,
                   out_path: str = DEFAULT_PROFILE_PATH,
                   min_samples: int = 3) -> dict:
    """
    讀所有已標記的錄音 → 算色塊分布 → 每類取平均當範本 → 存成 JSON。
    """
    found = scan_labeled_files(data_root)
    if not found:
        _safe_print(f"[ERROR] 在 {data_root} 找不到任何已標記的錄音。")
        _safe_print("        請先把音檔分類到 <日期>/normal/ 和 <日期>/thirsty/ 資料夾。")
        return {}

    profiles = {}
    for label, files in found.items():
        grids = []
        for fp in files:
            try:
                grids.append(wav_to_blocks(fp))
            except Exception as e:
                _safe_print(f"  [跳過] {os.path.basename(fp)}: {e}")
        if len(grids) < min_samples:
            _safe_print(f"  [警告] 「{label}」只有 {len(grids)} 筆（建議至少 {min_samples} 筆），"
                        f"範本會很不穩定")
        if not grids:
            continue
        stack = np.stack(grids)
        profiles[label] = {
            "mean": stack.mean(axis=0).tolist(),
            "std": stack.std(axis=0).tolist(),
            "n_samples": len(grids),
        }
        _safe_print(f"  ✓ 「{label}」: {len(grids)} 筆 → 已建立範本")

    if profiles:
        meta = {
            "n_freq": N_FREQ_BANDS,
            "n_time": N_TIME_SLICES,
            "max_freq": MAX_FREQ,
            "mains_hz": MAINS_HZ,
            "notch": APPLY_NOTCH,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"meta": meta, "profiles": profiles}, fh,
                      ensure_ascii=False, indent=2)
        _safe_print(f"\n範本已存檔: {out_path}")
    return profiles


def load_profiles(path: str = DEFAULT_PROFILE_PATH) -> dict:
    """讀回範本檔，沒有就回傳空 dict。"""
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("profiles", {})
    except Exception as e:
        _safe_print(f"[WARNING] 範本讀取失敗: {e}")
        return {}


# ─── 分類 ───────────────────────────────────────────
def classify_grid(grid: np.ndarray, profiles: dict) -> dict:
    """
    把一張色塊分布跟所有範本比對。

    回傳:
        {
          "prediction": "thirsty",
          "confidence": 0.78,
          "scores": {"normal": 0.22, "thirsty": 0.78},   # 相似度，加總為 1
          "distances": {"normal": 12.4, "thirsty": 3.1}, # 原始距離，越小越像
        }
    沒有範本時回傳 prediction=None。
    """
    if not profiles:
        return {"prediction": None, "confidence": 0.0, "scores": {}, "distances": {},
                "message": "尚未建立範本，請先執行 build"}

    distances = {}
    for label, prof in profiles.items():
        ref = np.array(prof["mean"], dtype=np.float64)
        if ref.shape != grid.shape:
            continue
        # 歐氏距離：每格相減、平方、加總、開根號
        distances[label] = float(np.sqrt(np.sum((grid - ref) ** 2)))

    if not distances:
        return {"prediction": None, "confidence": 0.0, "scores": {}, "distances": {},
                "message": "範本格子大小跟目前設定不符，請重新 build"}

    # 距離轉相似度：距離越小分數越高，全部加總正規化為 1
    inv = {k: 1.0 / (d + 1e-6) for k, d in distances.items()}
    total = sum(inv.values())
    scores = {k: v / total for k, v in inv.items()}
    best = max(scores, key=scores.get)

    return {
        "prediction": best,
        "confidence": scores[best],
        "scores": scores,
        "distances": distances,
    }


def classify_wav(path: str, profiles: dict = None) -> dict:
    if profiles is None:
        profiles = load_profiles()
    return classify_grid(wav_to_blocks(path), profiles)


def classify_audio(x: np.ndarray, sr: int, profiles: dict = None) -> dict:
    """給即時錄音用的入口（直接吃 numpy 陣列，不用先存檔）。"""
    if profiles is None:
        profiles = load_profiles()
    return classify_grid(spectro_blocks(x, sr), profiles)


# ─── CLI ────────────────────────────────────────────
def cmd_check(args):
    _safe_print("=" * 58)
    _safe_print("已標記資料盤點")
    _safe_print("=" * 58)
    found = scan_labeled_files(args.data_root)
    if not found:
        _safe_print(f"在 {args.data_root} 找不到任何已標記的錄音。\n")
        unlabeled = glob.glob(os.path.join(args.data_root, "*", "unlabeled", "*.wav"))
        _safe_print(f"未標記的錄音: {len(unlabeled)} 筆")
        _safe_print("\n下一步：把音檔搬到對應資料夾，例如")
        _safe_print(f"  {os.path.join(args.data_root, '2026-09-20', 'normal')}/")
        _safe_print(f"  {os.path.join(args.data_root, '2026-09-20', 'thirsty')}/")
        return
    total = 0
    for label, files in found.items():
        _safe_print(f"  {label:10s}: {len(files):4d} 筆")
        total += len(files)
    _safe_print(f"  {'合計':10s}: {total:4d} 筆")
    unlabeled = glob.glob(os.path.join(args.data_root, "*", "unlabeled", "*.wav"))
    if unlabeled:
        _safe_print(f"\n（另有 {len(unlabeled)} 筆未標記，不會被使用）")


def cmd_build(args):
    _safe_print("=" * 58)
    _safe_print("建立色塊分布範本")
    _safe_print("=" * 58)
    _safe_print(f"  格子大小: {N_FREQ_BANDS} 頻段 × {N_TIME_SLICES} 時間段")
    _safe_print(f"  頻率範圍: 0~{MAX_FREQ:.0f} Hz")
    _safe_print(f"  市電濾除: {'開啟 ' + str(MAINS_HZ) + 'Hz' if APPLY_NOTCH else '關閉'}")
    _safe_print("")
    profiles = build_profiles(args.data_root, args.profile, args.min_samples)
    if len(profiles) < 2:
        _safe_print("\n⚠️ 至少要有兩類（例如 normal 和 thirsty）才能做比對。")


def cmd_test(args):
    profiles = load_profiles(args.profile)
    if not profiles:
        _safe_print(f"[ERROR] 找不到範本檔 {args.profile}，請先執行 build")
        return
    if not os.path.isfile(args.wav):
        _safe_print(f"[ERROR] 找不到檔案: {args.wav}")
        return

    grid = wav_to_blocks(args.wav)
    result = classify_grid(grid, profiles)

    _safe_print("=" * 58)
    _safe_print(f"測試檔案: {os.path.basename(args.wav)}")
    _safe_print("=" * 58)
    _safe_print("\n這筆訊號的色塊分布（已正規化，數字大=該格能量相對高）:")
    for row in grid:
        _safe_print("  " + "  ".join(f"{v:6.2f}" for v in row))

    _safe_print(f"\n判定結果: 【{result['prediction']}】  信心度 {result['confidence']*100:.1f}%")
    _safe_print("\n跟各範本的距離（越小越像）:")
    for label in sorted(result["distances"], key=result["distances"].get):
        d = result["distances"][label]
        s = result["scores"][label]
        n = profiles[label].get("n_samples", "?")
        _safe_print(f"  {label:10s} 距離={d:7.2f}  相似度={s*100:5.1f}%  (範本由 {n} 筆平均)")

    _safe_print("\n⚠️ 提醒：範本筆數太少時，這個結果只能參考，不能當結論。")


def main():
    ap = argparse.ArgumentParser(description="色塊分布相似度比對")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="錄音資料根目錄")
    ap.add_argument("--profile", default=DEFAULT_PROFILE_PATH, help="範本 JSON 路徑")
    sub = ap.add_subparsers(dest="cmd")

    p_check = sub.add_parser("check", help="盤點已標記的資料")
    p_check.set_defaults(func=cmd_check)

    p_build = sub.add_parser("build", help="用已標記資料建立範本")
    p_build.add_argument("--min-samples", type=int, default=3, help="每類最少幾筆才算穩定")
    p_build.set_defaults(func=cmd_build)

    p_test = sub.add_parser("test", help="測單一 WAV 檔")
    p_test.add_argument("wav", help="要測試的 WAV 檔路徑")
    p_test.set_defaults(func=cmd_test)

    args = ap.parse_args()
    if not getattr(args, "func", None):
        ap.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
