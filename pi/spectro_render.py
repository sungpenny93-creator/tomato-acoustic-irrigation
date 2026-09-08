"""
波形 + 頻譜圖繪製 — 共用函式
============================
pi/spectro_web.py（獨立頁面）和 pi/web_controller.py（儀表板）都用這裡的函式，
避免兩邊各寫一份。純函式，不碰 Flask、不碰硬體。
"""

import io
import base64

import numpy as np
from scipy.io.wavfile import read as _wav_read
from scipy.signal import spectrogram as _scipy_spectrogram

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def wav_to_mono_float(path: str):
    """讀 WAV → (單聲道 float32 陣列 [-1,1], 取樣率)。"""
    sr, data = _wav_read(path)
    x = np.asarray(data, dtype=np.float32)
    if x.ndim > 1:
        x = x[:, 0]
    if np.issubdtype(data.dtype, np.integer):
        x = x / float(np.iinfo(data.dtype).max)
    elif x.size and np.max(np.abs(x)) > 1.5:
        x = x / 32768.0
    return x, sr


def stats_text(x: np.ndarray, sr: int) -> str:
    """一行文字統計：取樣率 / 樣本數 / RMS / 主頻 / 削波比例。"""
    n = len(x)
    if n == 0:
        return f"SR={sr}Hz  (無資料)"
    xw = x - float(np.mean(x))
    spec = np.abs(np.fft.rfft(xw))
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    band = freqs > 1.0
    peak_f = float(freqs[band][np.argmax(spec[band])]) if band.any() else 0.0
    rms = float(np.sqrt(np.mean(xw ** 2)))
    clip = float(np.mean(np.abs(x) > 0.98)) * 100
    return (f"SR={sr}Hz  samples={n}  RMS={rms:.4f}  "
            f"peak={peak_f:.0f}Hz  clip={clip:.1f}%")


def render_waveform_spectrogram_b64(x: np.ndarray, sr: int,
                                    max_freq: float = None) -> str:
    """
    畫「上：波形，下：頻譜圖(dB)」，回傳 PNG 的 base64 字串。
    max_freq: 頻譜圖 y 軸上限（Hz）。None = 顯示到奈奎斯特頻率。
             生物電位訊號建議帶 max_freq=500 之類，才看得清楚。
    """
    n = len(x)
    if n == 0:
        x = np.zeros(int(sr * 0.1), dtype=np.float32)
        n = len(x)

    t_axis = np.arange(n) / sr
    xw = x - float(np.mean(x))

    nperseg = min(1024, max(128, n // 8))
    f, t, Sxx = _scipy_spectrogram(xw, fs=sr, nperseg=nperseg,
                                   noverlap=int(nperseg * 0.75))
    Sxx_db = 10.0 * np.log10(Sxx + 1e-12)

    fig, ax = plt.subplots(2, 1, figsize=(11, 6),
                           gridspec_kw={"height_ratios": [1, 2]})
    ax[0].plot(t_axis, x, lw=0.5, color="#2b8")
    ax[0].set_xlim(0, t_axis[-1] if n else 1)
    ax[0].set_ylim(-1.05, 1.05)
    ax[0].set_ylabel("amplitude")
    ax[0].set_title("Waveform")

    mesh = ax[1].pcolormesh(t, f, Sxx_db, shading="gouraud", cmap="magma")
    if max_freq:
        ax[1].set_ylim(0, min(max_freq, sr / 2))
    ax[1].set_ylabel("frequency (Hz)")
    ax[1].set_xlabel("time (s)")
    ax[1].set_title("Spectrogram (dB)")
    fig.colorbar(mesh, ax=ax[1], pad=0.01)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")
