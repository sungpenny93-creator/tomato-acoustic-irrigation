import time
import json
import sys
import os

# 為了確保在 Windows 上也能從根目錄以 python -m pi.main 執行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi.audio_capture import AudioCapture
from pi.preprocessor import AudioPreprocessor
from pi.inference import TFLiteInferenceEngine
from pi.gpio_controller import GPIOController
from pi.irrigation import IrrigationDecisionMaker

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    print("=" * 50)
    print("🌱 番茄生理聲學監測與自動化灌溉系統 (Edge Node) 🌱")
    print("=" * 50)
    
    # 1. 載入設定
    config = load_config()
    print("[SYSTEM] 設定檔載入完成。")

    # 2. 初始化各個模組
    # 修正模型路徑為絕對路徑且統一斜線方向，避免 TFLite C++ 引擎在 Windows 報錯
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_abs_path = os.path.normpath(os.path.join(project_root, config["model_path"]))
    
    audio = AudioCapture(
        sample_rate=config["audio"]["sample_rate"],
        duration=config["audio"]["duration_seconds"]
    )
    preprocessor = AudioPreprocessor()
    engine = TFLiteInferenceEngine(model_path=model_abs_path)
    gpio = GPIOController(relay_pin=config["gpio"]["relay_pin"])
    decision_maker = IrrigationDecisionMaker(
        thirsty_threshold=config["logic"]["thirsty_threshold"],
        consecutive_required=config["logic"]["consecutive_thirsty_required"]
    )
    
    listen_interval = config["audio"]["listen_interval_seconds"]
    irrigation_duration = config["gpio"]["irrigation_duration_seconds"]
    temp_wav_path = os.path.join(os.path.dirname(__file__), "live_audio.wav")

    print("\n🚀 系統初始化完成，開始 24 小時監測迴圈 (按 Ctrl+C 停止)...\n")

    try:
        while True:
            # Step 1: 錄音
            success = audio.record_audio(temp_wav_path)
            
            if success:
                # Step 2: 前處理 (抗噪 + 頻譜圖)
                input_tensor = preprocessor.process(temp_wav_path)
                
                # Step 3: 模型推論
                result = engine.predict(input_tensor)
                
                # Step 4: 決策邏輯
                needs_water = decision_maker.process_prediction(result)
                
                # Step 5: 動作
                if needs_water:
                    print("💧 [ACTION] 啟動抽水馬達進行灌溉！")
                    gpio.run_irrigation_cycle(irrigation_duration)
                    print(f"😴 [ACTION] 剛澆完水，暫停監聽 60 秒讓水吸收...")
                    time.sleep(60) # 澆完水後強制休息，避免一直連環觸發
                    
            print(f"⏳ 等待 {listen_interval} 秒後進行下一次監聽...\n")
            time.sleep(listen_interval)

    except KeyboardInterrupt:
        print("\n🛑 收到終止訊號，系統關閉中...")
    finally:
        # 確保 GPIO 資源被釋放，繼電器被關閉
        gpio.cleanup()
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)
        print("✅ 系統已安全關閉。")

if __name__ == "__main__":
    main()
