import yaml
import openai

class VisionDescriberAgent:
    def __init__(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        openai.api_key = self.config['api_keys']['llm_api_key']
        self.system_prompt = """
        Bạn là chuyên gia kiến trúc đô thị. Hãy nhìn vào bức ảnh POI này và mô tả ngắn gọn (dưới 50 từ):
        1. Loại công trình/cửa hàng.
        2. Đặc điểm kiến trúc hoặc biển hiệu (màu sắc, vật liệu).
        Ví dụ: "Cửa hàng có biển hiệu màu đỏ, chữ vàng, có nhiều bàn ghế gỗ bên ngoài."
        """

    def describe_image(self, image_url):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o", # Model hỗ trợ vision
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": [
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]}
                ],
                max_tokens=100
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return "Không thể trích xuất mô tả hình ảnh."