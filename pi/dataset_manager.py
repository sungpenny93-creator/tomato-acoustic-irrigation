"""
資料集管理模組 — 管理 SpikerBox 蒐集的本地資料集
=================================================
功能：
- 統計各類別的錄音筆數
- 列出近期蒐集的檔案
- 提供資料集摘要（供 Web API 呼叫）
"""

import os
import json
from datetime import datetime


class DatasetManager:
    """管理本地蒐集的 SpikerBox 資料集"""

    LABELS = ["unlabeled", "noise", "normal", "thirsty"]

    def __init__(self, raw_dir: str, spectrogram_dir: str):
        self.raw_dir = raw_dir
        self.spectrogram_dir = spectrogram_dir
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.spectrogram_dir, exist_ok=True)

    def get_stats(self) -> dict:
        """
        取得資料集統計摘要。
        回傳格式：
        {
            "total": 123,
            "by_label": {"unlabeled": 50, "noise": 20, "normal": 30, "thirsty": 23},
            "by_date": {"2026-08-28": 45, "2026-08-27": 78},
            "raw_dir": "...",
        }
        """
        stats = {
            "total": 0,
            "by_label": {},
            "by_date": {},
            "raw_dir": self.raw_dir,
        }

        if not os.path.isdir(self.raw_dir):
            return stats

        for date_folder in sorted(os.listdir(self.raw_dir)):
            date_path = os.path.join(self.raw_dir, date_folder)
            if not os.path.isdir(date_path):
                continue

            date_count = 0
            for label_folder in os.listdir(date_path):
                label_path = os.path.join(date_path, label_folder)
                if not os.path.isdir(label_path):
                    continue

                wav_files = [f for f in os.listdir(label_path) if f.endswith(".wav")]
                count = len(wav_files)
                date_count += count

                stats["by_label"][label_folder] = stats["by_label"].get(label_folder, 0) + count

            if date_count > 0:
                stats["by_date"][date_folder] = date_count
                stats["total"] += date_count

        return stats

    def get_recent_files(self, limit: int = 20) -> list:
        """取得最近蒐集的檔案列表"""
        all_files = []

        if not os.path.isdir(self.raw_dir):
            return all_files

        for date_folder in os.listdir(self.raw_dir):
            date_path = os.path.join(self.raw_dir, date_folder)
            if not os.path.isdir(date_path):
                continue
            for label_folder in os.listdir(date_path):
                label_path = os.path.join(date_path, label_folder)
                if not os.path.isdir(label_path):
                    continue
                for f in os.listdir(label_path):
                    if f.endswith(".wav"):
                        full_path = os.path.join(label_path, f)
                        all_files.append({
                            "path": full_path,
                            "date": date_folder,
                            "label": label_folder,
                            "filename": f,
                            "size_kb": round(os.path.getsize(full_path) / 1024, 1),
                        })

        # 按檔名（時間戳）倒序排列，取最近 N 筆
        all_files.sort(key=lambda x: x["filename"], reverse=True)
        return all_files[:limit]

    def get_spectrogram_stats(self) -> dict:
        """取得頻譜圖統計"""
        count = 0
        if os.path.isdir(self.spectrogram_dir):
            for f in os.listdir(self.spectrogram_dir):
                if f.endswith((".png", ".jpg")):
                    count += 1
        return {"total": count, "dir": self.spectrogram_dir}


if __name__ == "__main__":
    import json as _json
    # 測試
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mgr = DatasetManager(
        raw_dir=os.path.join(project_root, "data", "spikerbox_raw"),
        spectrogram_dir=os.path.join(project_root, "data", "spikerbox_spectrograms"),
    )
    print(_json.dumps(mgr.get_stats(), indent=2, ensure_ascii=False))
