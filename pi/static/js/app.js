/**
 * 番茄灌溉系統 — 前端互動邏輯（整合版）
 * ========================================
 * 功能：水泵控制、自動監測、即時頻譜圖、AI 推論顯示、資料集統計
 */

// ──── 全域狀態 ────
let pollTimer = null;
let countdownTimer = null;
let countdownRemaining = 0;
let countdownTotal = 0;

// ──── 初始化 ────
document.addEventListener("DOMContentLoaded", () => {
    fetchStatus();
    fetchLog();
    fetchInference();
    fetchLiveSpectro();
    fetchDatasetStats();
    // 每 2 秒輪詢
    pollTimer = setInterval(() => {
        fetchStatus();
        fetchLog();
        fetchInference();
        fetchLiveSpectro();
    }, 2000);
    // 資料集統計每 30 秒更新
    setInterval(fetchDatasetStats, 30000);
});

// ──── API 呼叫工具 ────
async function apiCall(url, method = "GET", body = null) {
    try {
        const opts = { method, headers: { "Content-Type": "application/json" } };
        if (body) opts.body = JSON.stringify(body);
        const res = await fetch(url, opts);
        const data = await res.json();
        updateConnection(true);
        return data;
    } catch (e) {
        updateConnection(false);
        return null;
    }
}

// ──── 連線狀態 ────
function updateConnection(ok) {
    const badge = document.getElementById("connection-badge");
    const label = badge.querySelector(".label");
    if (ok) {
        badge.classList.remove("disconnected");
        label.textContent = "已連線";
    } else {
        badge.classList.add("disconnected");
        label.textContent = "離線";
    }
}

// ──── 系統狀態輪詢 ────
async function fetchStatus() {
    const data = await apiCall("/api/status");
    if (!data) return;

    // 水泵環圈
    const ring = document.getElementById("pump-ring");
    const label = document.getElementById("pump-label");
    if (data.pump_on) {
        ring.classList.add("active");
        label.classList.add("active");
        label.textContent = "運轉中";
    } else {
        ring.classList.remove("active");
        label.classList.remove("active");
        label.textContent = "已停止";
    }

    // 監測狀態
    const monBadge = document.getElementById("monitor-badge");
    if (data.monitoring) {
        monBadge.textContent = "監測中";
        monBadge.className = "monitor-badge monitoring";
    } else {
        monBadge.textContent = "未啟動";
        monBadge.className = "monitor-badge";
    }

    // 系統資訊
    document.getElementById("info-pin").textContent = `GPIO ${data.relay_pin}`;
    document.getElementById("info-cycles").textContent = data.total_cycles;
    document.getElementById("info-last").textContent = data.last_irrigation || "尚未灌溉";

    const sec = data.uptime_seconds;
    if (sec < 60) document.getElementById("info-uptime").textContent = `${sec} 秒`;
    else if (sec < 3600) document.getElementById("info-uptime").textContent = `${Math.floor(sec / 60)} 分`;
    else document.getElementById("info-uptime").textContent = `${Math.floor(sec / 3600)} 時 ${Math.floor((sec % 3600) / 60)} 分`;
}

// ──── 推論結果 ────
async function fetchInference() {
    const data = await apiCall("/api/inference/latest");
    if (!data) return;

    // 頻譜圖
    const img = document.getElementById("spectrogram-img");
    const placeholder = document.getElementById("spectrogram-placeholder");
    if (data.spectrogram_b64) {
        img.src = `data:image/png;base64,${data.spectrogram_b64}`;
        img.style.display = "block";
        placeholder.style.display = "none";
    }

    // 推論結果
    if (data.inference) {
        const inf = data.inference;
        const labelEl = document.getElementById("inference-label");
        const confEl = document.getElementById("inference-confidence");

        const labelMap = { normal: "🌿 正常", thirsty: "🔥 缺水", noise: "📢 噪音" };
        labelEl.textContent = labelMap[inf.prediction] || inf.prediction;
        labelEl.className = `inference-label pred-${inf.prediction}`;
        confEl.textContent = `${(inf.confidence * 100).toFixed(1)}%`;

        // 機率條
        const probs = inf.probabilities || {};
        updateProbBar("normal", probs.normal || 0);
        updateProbBar("thirsty", probs.thirsty || 0);
        updateProbBar("noise", probs.noise || 0);
    }
}

// ──── 即時波形/頻譜圖（不需 AI 模型）────
async function fetchLiveSpectro() {
    const data = await apiCall("/api/spectro");
    if (!data) return;

    const img = document.getElementById("live-spectro-img");
    const placeholder = document.getElementById("live-spectro-placeholder");
    if (data.png) {
        img.src = `data:image/png;base64,${data.png}`;
        img.style.display = "block";
        if (placeholder) placeholder.style.display = "none";
    }

    const modeEl = document.getElementById("live-spectro-mode");
    if (modeEl) {
        const label = data.mode === "mock" ? "⚠️ 模擬(無硬體)"
            : data.mode === "serial" ? "序列埠"
            : data.mode === "audio" ? "USB 音訊" : (data.mode || "—");
        modeEl.textContent = data.ts ? `${label} · ${data.ts}` : label;
        modeEl.className = "monitor-badge" + (data.mode === "mock" ? "" : " monitoring");
    }

    const statsEl = document.getElementById("live-spectro-stats");
    if (statsEl) statsEl.textContent = data.stats_text || "";
}

function updateProbBar(name, value) {
    const pct = (value * 100).toFixed(1);
    const fill = document.getElementById(`prob-${name}`);
    const val = document.getElementById(`prob-${name}-val`);
    if (fill) fill.style.width = `${pct}%`;
    if (val) val.textContent = `${pct}%`;
}

// ──── 資料集統計 ────
async function fetchDatasetStats() {
    const data = await apiCall("/api/dataset/stats");
    if (!data) return;
    document.getElementById("info-dataset").textContent = data.total || 0;

    // 頻譜圖數（通過 spectrogram 統計 API 或直接從 stats）
    // 暫時用 total 代替
    const specEl = document.getElementById("info-spectrograms");
    if (specEl) {
        // 計算 by_label 的總數作為近似
        specEl.textContent = data.total || 0;
    }
}

// ──── 日誌輪詢 ────
async function fetchLog() {
    const data = await apiCall("/api/log");
    if (!data || !data.log) return;

    const list = document.getElementById("log-list");
    if (data.log.length === 0) {
        list.innerHTML = '<li class="log-empty">尚無活動記錄</li>';
        return;
    }

    const iconMap = {
        "水泵開啟": "💧", "水泵關閉": "🔴", "系統啟動": "🚀",
        "定時灌溉開始": "💧", "定時灌溉完成": "✅", "日誌已清除": "🗑️",
        "自動監測啟動": "🤖", "自動監測已停止": "⏹️",
        "自動灌溉啟動": "💧", "自動灌溉完成": "✅",
        "錄音完成": "🎤", "錄音失敗": "❌", "監測錯誤": "⚠️",
    };

    list.innerHTML = data.log.map(entry => {
        // 找推論結果的圖示
        let icon = "📋";
        for (const [key, val] of Object.entries(iconMap)) {
            if (entry.action.includes(key)) { icon = val; break; }
        }
        if (entry.action.includes("推論")) {
            if (entry.action.includes("thirsty") || entry.action.includes("缺水")) icon = "🔥";
            else if (entry.action.includes("normal") || entry.action.includes("正常")) icon = "🌿";
            else if (entry.action.includes("noise") || entry.action.includes("噪音")) icon = "📢";
            else icon = "🧠";
        }
        return `
            <li class="log-item">
                <span class="log-time">${entry.time}</span>
                <div class="log-content">
                    <div class="log-action">${icon} ${entry.action}</div>
                    ${entry.detail ? `<div class="log-detail">${entry.detail}</div>` : ""}
                </div>
            </li>`;
    }).join("");
}

// ──── 水泵控制 ────
async function pumpOn() {
    const data = await apiCall("/api/pump/on", "POST");
    if (data) showToast(data.message, data.success ? "success" : "error");
}

async function pumpOff() {
    const data = await apiCall("/api/pump/off", "POST");
    if (data) showToast(data.message, data.success ? "success" : "error");
}

async function pumpCycle() {
    const dur = parseInt(document.getElementById("duration-input").value) || 5;
    const data = await apiCall("/api/pump/cycle", "POST", { duration: dur });
    if (data && data.success) {
        showToast(data.message, "success");
        startCountdown(data.duration);
    } else if (data) {
        showToast(data.message, "error");
    }
}

// ──── 監測控制 ────
async function monitorStart() {
    const data = await apiCall("/api/monitor/start", "POST");
    if (data) showToast(data.message, data.success ? "success" : "error");
}

async function monitorStop() {
    const data = await apiCall("/api/monitor/stop", "POST");
    if (data) showToast(data.message, data.success ? "success" : "error");
}

// ──── 倒數計時 ────
function startCountdown(seconds) {
    countdownTotal = seconds;
    countdownRemaining = seconds;
    const bar = document.getElementById("countdown-bar");
    const fill = document.getElementById("countdown-fill");
    const text = document.getElementById("countdown-text");

    bar.classList.add("visible");
    fill.style.width = "100%";

    if (countdownTimer) clearInterval(countdownTimer);
    countdownTimer = setInterval(() => {
        countdownRemaining--;
        if (countdownRemaining <= 0) {
            clearInterval(countdownTimer);
            bar.classList.remove("visible");
            return;
        }
        const pct = (countdownRemaining / countdownTotal) * 100;
        fill.style.width = `${pct}%`;
        text.textContent = `${countdownRemaining}s`;
    }, 1000);
}

// ──── 持續時間控制 ────
function adjustDuration(delta) {
    const input = document.getElementById("duration-input");
    let val = parseInt(input.value) || 5;
    val = Math.max(1, Math.min(300, val + delta));
    input.value = val;
}

// ──── 日誌清除 ────
async function clearLog() {
    await apiCall("/api/log/clear", "POST");
}

// ──── Toast 通知 ────
function showToast(msg, type = "info") {
    const container = document.getElementById("toast-container");
    const icons = { success: "✅", error: "❌", info: "ℹ️" };
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-icon">${icons[type]}</span><span class="toast-msg">${msg}</span>`;
    container.appendChild(toast);
    setTimeout(() => {
        toast.classList.add("leaving");
        setTimeout(() => toast.remove(), 300);
    }, 3500);
}
