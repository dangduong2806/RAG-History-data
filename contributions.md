# Đóng góp của các thành viên (Team Contributions)
## 1. Nguyễn Đăng Dương
- **Gán nhãn dữ liệu (Data Annotation):** Biên soạn và gán nhãn các cặp câu QA từ 1 đến 13 (Tập trung vào chủ đề Lịch sử ĐHQGHN và Sứ mệnh - Tầm nhìn).
- **Thu thập dữ liệu (Web Crawling):** Viết script (`src/data_collection/crawler.py`) sử dụng thư viện `requests` và `BeautifulSoup4` để tự động thu thập và trích xuất văn bản từ 5 trang chuyên mục của VNU.
- **Tiền xử lý (Data Cleaning):** Xây dựng các tập luật làm sạch phức tạp bằng RegEx trong `src/data_processing/data_cleaner.py`. Xử lý chuyển đổi dữ liệu dạng bảng biểu (chỉ tiêu chiến lược, thống kê nhân sự) thành các cấu trúc câu văn tự nhiên.
- **Phân mảnh ngữ nghĩa (Semantic Chunking):** Cài đặt thuật toán chia văn bản thông minh (`src/data_processing/chunker.py`) theo các mốc thời gian, số La Mã và đơn vị tổ chức thay vì chia theo số lượng ký tự.
- **Viết báo cáo (Report):** Đảm nhiệm viết Mục 2 (Dữ liệu) và Mục 4 (Tập hỏi-đáp) của báo cáo khoa học.

## 2. Bùi Hải Đăng
- **Gán nhãn dữ liệu (Data Annotation):** Biên soạn và gán nhãn các cặp câu QA từ 14 đến 26 (Tập trung vào chủ đề Chiến lược phát triển). Đồng thời, cùng Tôn Thành Đạt độc lập gán nhãn chéo tập 8 câu hỏi ngẫu nhiên để tính toán các chỉ số đồng thuận IAA.
- **Hệ thống truy xuất (Dense Retrieval):** Cài đặt logic truy xuất cốt lõi trong `src/model/rag_pipeline.py`. Tích hợp mô hình nhúng `BAAI/bge-m3` và thiết lập cơ sở dữ liệu vector `FAISS` để tìm kiếm tương đồng.
- **Tích hợp Reranking:** Tích hợp mô hình Cross-Encoder (`BAAI/bge-reranker-base`) để chấm điểm và lọc lại từ top-10 chunks của FAISS xuống top-3 chunks tốt nhất.
- **Sinh câu trả lời (LLM Generation):** Cấu hình mô hình ngôn ngữ `Qwen/Qwen2.5-7B-Instruct` qua thư viện `transformers`. Thiết kế prompt với các ràng buộc ngữ cảnh và chuẩn bị các ví dụ few-shot.
- **Viết báo cáo (Report):** Đảm nhiệm viết Mục 3 (Phương pháp) và thiết kế sơ đồ khối kiến trúc hệ thống (Flowchart).

## 3. Tôn Thành Đạt
- **Gán nhãn dữ liệu (Data Annotation):** Biên soạn và gán nhãn các cặp câu QA từ 27 đến 40 (Tập trung vào chủ đề Thi đua khen thưởng và Số liệu thống kê). Phối hợp cùng Bùi Hải Đăng thực hiện đánh giá chéo để tính điểm IAA.
- **Các chỉ số đánh giá (Evaluation Metrics):** Cài đặt các hàm chuẩn hóa văn bản (Normal/Canonical) và code thuật toán đo lường 3 chỉ số tiêu chuẩn: Exact Match, Token-level F1, Recall (`src/evaluation/evaluate.py`).
- **Kiểm thử hệ thống (End-to-End Testing):** Viết script `src/evaluation/run_eval.py` để tự động hóa luồng chạy: đẩy câu hỏi test vào RAG pipeline, lưu kết quả tự động ra file và tính điểm.
- **Thực nghiệm Baseline:** Chạy các thực nghiệm để so sánh và phân tích độ hiệu quả giữa hệ thống RAG đề xuất với mô hình Closed-book LLM.
- **Viết báo cáo & Quản lý dự án:** Đảm nhiệm viết Mục 5 (Đánh giá) và Mục 6 (Phân tích lỗi & Thảo luận). Quản lý repository trên GitHub, hoàn thiện `README.md` và đóng gói nộp bài.