import yaml
import openai

class POICleanerAgent:
    def __init__(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        # Sử dụng thư viện openai làm chuẩn, bạn có thể thay bằng thư viện google-generativeai
        openai.api_key = self.config['api_keys']['llm_api_key']
        self.system_prompt = """
        Bạn là chuyên gia chuẩn hóa dữ liệu bản đồ.
        Nhiệm vụ: So sánh 2 bản ghi POI và xác định xem chúng có phải là cùng một địa điểm thực tế hay không.
        Trả lời theo định dạng JSON: {"is_duplicate": true/false, "reason": "Lý do ngắn gọn"}
        """

    def resolve_entity(self, poi_record_1, poi_record_2):
        prompt = f"Bản ghi A: {poi_record_1}\nBản ghi B: {poi_record_2}\nChúng có phải là một không?"
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini", # Hoặc gemini-1.5-flash
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"{{\"is_duplicate\": false, \"reason\": \"Lỗi API: {str(e)}\"}}"