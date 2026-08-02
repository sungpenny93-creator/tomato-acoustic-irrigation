import datetime

class IrrigationDecisionMaker:
    def __init__(self, thirsty_threshold: float = 0.85, consecutive_required: int = 2):
        self.thirsty_threshold = thirsty_threshold
        self.consecutive_required = consecutive_required
        self.consecutive_count = 0

    def process_prediction(self, prediction_result: dict) -> bool:
        """
        處理最新的預測結果。
        如果達到連續缺水標準，則回傳 True (代表需要啟動馬達澆水)
        否則回傳 False
        """
        label = prediction_result["prediction"]
        confidence = prediction_result["confidence"]
        
        # 紀錄一下當下時間
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if label == "thirsty" and confidence >= self.thirsty_threshold:
            self.consecutive_count += 1
            print(f"[{current_time}] ⚠️ 偵測到缺水！(信心度: {confidence*100:.1f}%) | 累積警告: {self.consecutive_count}/{self.consecutive_required}")
            
            if self.consecutive_count >= self.consecutive_required:
                # 達到標準，需要澆水，將計數器歸零避免重複狂澆
                self.consecutive_count = 0
                return True
        else:
            if self.consecutive_count > 0:
                print(f"[{current_time}] 聽起來不是缺水 (預測為 {label})。缺水計數器歸零。")
            else:
                print(f"[{current_time}] 植物狀態正常 ({label}, 信心度: {confidence*100:.1f}%)")
            self.consecutive_count = 0
            
        return False
