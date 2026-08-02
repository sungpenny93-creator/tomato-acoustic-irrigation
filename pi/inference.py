import numpy as np

# 在樹莓派上為了省資源，通常只安裝 tflite_runtime。
# 但為了讓你在 Windows 也可以無縫測試，我們做一個 Fallback 機制：
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    print("[INFO] 找不到 tflite_runtime，改用完整版 tensorflow.lite 進行推論測試。")
    from tensorflow.lite.python.interpreter import Interpreter

class TFLiteInferenceEngine:
    def __init__(self, model_path: str):
        self.model_path = model_path
        # 讀取檔案內容為 bytes (為了解決 Windows 下 tflite C++ 引擎不支援中文路徑的 Bug)
        with open(model_path, "rb") as f:
            model_data = f.read()
            
        # 載入 TFLite 模型
        self.interpreter = Interpreter(model_content=model_data)
        self.interpreter.allocate_tensors()

        # 取得輸入與輸出的張量資訊
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # 定義我們的分類標籤 (順序必須跟訓練時 class_names 的順序一致: [noise, normal, thirsty])
        self.class_names = ["noise", "normal", "thirsty"]
        
        print(f"[MODEL] TFLite 模型載入成功 ({model_path})")

    def predict(self, input_data: np.ndarray) -> dict:
        """
        傳入 (1, 224, 224, 3) 的圖片矩陣，回傳各類別機率與最終結果。
        """
        # 將資料餵給模型輸入層
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # 執行推論
        self.interpreter.invoke()

        # 取得輸出層結果 (這是 logits 分數)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        # 套用 Softmax 轉成機率
        probabilities = np.exp(output_data) / np.sum(np.exp(output_data), axis=1, keepdims=True)
        probabilities = probabilities[0] # 取出第一個 batch 的結果

        # 整理輸出結果
        predicted_idx = np.argmax(probabilities)
        result = {
            "prediction": self.class_names[predicted_idx],
            "confidence": float(probabilities[predicted_idx]),
            "probabilities": {
                name: float(prob) for name, prob in zip(self.class_names, probabilities)
            }
        }
        
        return result

if __name__ == "__main__":
    # 測試用
    # 建立一個全黑的假圖片來測試模型是否能跑
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    engine = TFLiteInferenceEngine(model_path="../models/best_model.tflite")
    print(engine.predict(dummy_input))
