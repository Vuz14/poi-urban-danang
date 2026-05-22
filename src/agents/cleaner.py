import yaml
import openai
import re

class POICleanerAgent:
    def __init__(self):
        with open("config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        
        openai.api_key = self.config['api_keys']['llm_api_key']
        
        # Cập nhật danh sách từ khóa spam phổ biến tại thị trường Việt Nam
        self.spam_keywords = [
            r"XE DU LỊCH", r"CHO THUÊ XE", r"LIÊN HỆ NGAY", r"TRAVELL?", 
            r"SĐT:", r"0\d{9,10}", r"https?://\S+", r"CHUYÊN SỈ BÁN"
        ]
        
        # Nâng cấp System Prompt để LLM chủ động bỏ qua nhiễu quảng cáo khi đối chiếu
        self.system_prompt = """
        Bạn là chuyên gia chuẩn hóa dữ liệu bản đồ và lọc nhiễu thông tin.
        Nhiệm vụ: So sánh 2 bản ghi POI và xác định xem chúng có phải là cùng một địa điểm thực tế hay không.
        
        CHÚ Ý BẢO MẬT & LỌC SPAM: 
        Trong phần bình luận của khách hàng có thể chứa các nội dung rác, spam quảng cáo không liên quan 
        (ví dụ: quảng cáo cho thuê xe du lịch, số điện thoại dịch vụ khác, link website, đại lý vé...). 
        Bạn ĐÃ ĐƯỢC LỆNH phải bỏ qua, cô lập toàn bộ các thông tin spam này và CHỈ tập trung so sánh 
        tên quán, địa chỉ, danh mục, và các bình luận thực tế về chất lượng món ăn/dịch vụ của địa điểm.
        
        Trả lời nghiêm ngặt theo định dạng JSON: {"is_duplicate": true/false, "reason": "Lý do ngắn gọn"}
        """

    def _clean_spam_text(self, text):
        """
        Hàm tiền xử lý cơ học: Tự động loại bỏ các câu hoặc từ khóa spam phổ biến
        để giảm lượng token gửi lên API và giảm nhiễu cục bộ cho mô hình.
        """
        if not isinstance(text, str):
            return text
            
        # Loại bỏ các dòng hoặc các câu chứa từ khóa nằm trong danh sách đen quảng cáo
        lines = text.split(" | ")
        cleaned_lines = []
        for line in lines:
            is_spam = False
            for pattern in self.spam_keywords:
                if re.search(pattern, line, re.IGNORECASE):
                    is_spam = True
                    break
            if not is_spam:
                cleaned_lines.append(line)
                
        # Trả về chuỗi sạch đã được nối lại bằng dấu gạch đứng phân tách bình luận
        return " | ".join(cleaned_lines) if cleaned_lines else "Không có bình luận hợp lệ."

    def resolve_entity(self, poi_record_1, poi_record_2):
        # Bước 1: Làm sạch cơ học chuỗi dữ liệu đầu vào trước khi đóng gói gửi lên LLM
        clean_poi_1 = self._clean_spam_text(str(poi_record_1))
        clean_poi_2 = self._clean_spam_text(str(poi_record_2))
        
        # Bước 2: Thiết lập prompt đối sánh thực thể sạch
        prompt = f"Bản ghi A: {clean_poi_1}\nBản ghi B: {clean_poi_2}\nChúng có phải là một không?"
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0 # Giữ nguyên độ ổn định tuyệt đối
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"{{\"is_duplicate\": false, \"reason\": \"Lỗi API: {str(e)}\"}}"