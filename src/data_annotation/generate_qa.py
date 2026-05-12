"""
Bước 4: Tạo dữ liệu QA chất lượng cao cho test/train
Dữ liệu được sinh thủ công dựa trên nội dung đã crawl và cleaned.
Tất cả câu trả lời đều khớp chính xác với dữ liệu raw/cleaned.
"""
import os
import random
import sys
sys.stdout.reconfigure(encoding='utf-8')

TRAIN_DIR = "data/train"
TEST_DIR = "data/test"


def setup_dirs():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)


# ========================================================
# BỘ DỮ LIỆU QA CHẤT LƯỢNG CAO (khớp 100% với dữ liệu đã crawl)
# Bao phủ cả 5 chủ đề: Lịch sử, Sứ mạng, Chiến lược, Khen thưởng, Thống kê
# ========================================================

QA_DATA = [
    # === LỊCH SỬ (12 câu) ===
    ("Đại học Đông Dương được thành lập vào năm nào?",
     "Năm 1906"),
    ("Trụ sở của Đại học Đông Dương đặt tại đâu?",
     "Số 19 phố Lê Thánh Tông, quận Hoàn Kiếm, thành phố Hà Nội"),
    ("Trường Đại học Quốc gia Việt Nam khai giảng khóa đầu tiên vào ngày nào?",
     "15 tháng 11 năm 1945"),
    ("Ai chủ tọa lễ khai giảng khóa đầu tiên của Trường Đại học Quốc gia Việt Nam?",
     "Chủ tịch Hồ Chí Minh"),
    ("Trường Đại học Tổng hợp Hà Nội được thành lập theo quyết định nào?",
     "Quyết định số 2183/TC ngày 04 tháng 6 năm 1956 của Chính phủ"),
    ("ĐHQGHN được thành lập trên cơ sở sắp xếp lại những trường đại học nào?",
     "Trường Đại học Tổng hợp Hà Nội, Trường Đại học Sư phạm Hà Nội I và Trường Đại học Sư phạm Ngoại ngữ Hà Nội"),
    ("Nghị định thành lập ĐHQGHN có số hiệu và ngày ban hành là gì?",
     "Nghị định số 97/CP ngày 10 tháng 12 năm 1993"),
    ("Trường Đại học Công nghệ thuộc ĐHQGHN được thành lập vào năm nào?",
     "Năm 2004"),
    ("Trường Đại học Kinh tế thuộc ĐHQGHN được thành lập trên cơ sở nào?",
     "Nâng cấp Khoa Kinh tế"),
    ("Trường Đại học Y Dược là trường đại học thành viên thứ mấy của ĐHQGHN?",
     "Thứ tám"),
    ("ĐHQGHN chính thức chuyển trụ sở làm việc tới Hòa Lạc vào ngày nào?",
     "19 tháng 5 năm 2022"),
    ("Tính đến tháng 01/2025, ĐHQGHN có bao nhiêu đơn vị sự nghiệp công lập thành viên và trực thuộc?",
     "34 đơn vị"),

    # === SỨ MỆNH - TẦM NHÌN (4 câu) ===
    ("Sứ mệnh của ĐHQGHN là gì?",
     "Hội tụ và nuôi dưỡng hiền tài, phát triển nhân lực và tri thức tinh hoa, dẫn dắt đổi mới sáng tạo, thực hiện chiến lược quốc gia và cùng kiến tạo tương lai nhân loại."),
    ("Tầm nhìn đến năm 2045 của ĐHQGHN là gì?",
     "Trở thành đại học nghiên cứu đẳng cấp thế giới, vận hành như một tập đoàn tri thức; nơi hội tụ nhân tài, văn hóa và tri thức tinh hoa, dẫn dắt đổi mới sáng tạo của Thủ đô và cả nước, đóng góp quan trọng hiện thực hóa tầm nhìn, mục tiêu phát triển quốc gia."),
    ("Giá trị cốt lõi của ĐHQGHN là gì?",
     "Tiên phong - Sáng tạo - Xuất sắc - Nhân văn - Phụng sự"),
    ("ĐHQGHN hoạt động theo cơ chế gì?",
     "Cơ chế tự chủ, tự chịu trách nhiệm cao"),

    # === CHIẾN LƯỢC PHÁT TRIỂN (10 câu) ===
    ("Mục tiêu tổng quát của chiến lược phát triển ĐHQGHN giai đoạn 2026-2030 là gì?",
     "Hiện đại hóa toàn diện, phát triển ĐHQGHN thành đại học nghiên cứu hàng đầu khu vực; là một trung tâm đào tạo nhân tài quốc gia, đóng vai trò hạt nhân tri thức, đổi mới sáng tạo của Thủ đô Hà Nội và cả nước."),
    ("ĐHQGHN đặt mục tiêu thuộc nhóm bao nhiêu đại học hàng đầu Châu Á vào năm 2030?",
     "Nhóm 100"),
    ("ĐHQGHN đặt mục tiêu thuộc nhóm bao nhiêu đại học hàng đầu thế giới vào năm 2035?",
     "Nhóm 300"),
    ("Có bao nhiêu trụ cột chiến lược trong chiến lược phát triển ĐHQGHN?",
     "6 trụ cột"),
    ("Trụ cột chiến lược đầu tiên của ĐHQGHN là gì?",
     "Quản trị tối ưu và kiến tạo"),
    ("Mục tiêu tỷ lệ giảng viên có trình độ tiến sĩ của ĐHQGHN vào năm 2030 là bao nhiêu?",
     "80%"),
    ("Khu đô thị đại học Hòa Lạc được phát triển theo mô hình gì?",
     "Đô thị tri thức và công nghệ cao"),
    ("Mục tiêu quy mô dân số tri thức làm việc thường xuyên tại Hòa Lạc vào năm 2030 là bao nhiêu?",
     "60.000"),
    ("Mục tiêu tổng giá trị quỹ hiến tặng ĐHQGHN vào năm 2030 là bao nhiêu?",
     "500 tỉ VNĐ"),
    ("Cơ cấu tuyển sinh đại học, thạc sĩ và tiến sĩ mục tiêu năm 2030 là bao nhiêu?",
     "6:3:1"),

    # === THI ĐUA KHEN THƯỞNG (6 câu) ===
    ("ĐHQGHN có bao nhiêu Huân chương Sao vàng?",
     "1"),
    ("ĐHQGHN có bao nhiêu Huân chương Hồ Chí Minh?",
     "3"),
    ("ĐHQGHN có bao nhiêu Nhà giáo Nhân dân?",
     "62"),
    ("ĐHQGHN có bao nhiêu Nhà giáo Ưu tú?",
     "137"),
    ("Số học sinh của ĐHQGHN đạt giải thưởng Quốc tế, Khu vực là bao nhiêu?",
     "249"),
    ("Số học sinh của ĐHQGHN đạt giải Quốc gia là bao nhiêu?",
     "637"),

    # === SỐ LIỆU THỐNG KÊ (8 câu) ===
    ("Tổng số người làm việc tại ĐHQGHN tính đến 31/12/2024 là bao nhiêu?",
     "5.281 người"),
    ("Tổng số viên chức của ĐHQGHN là bao nhiêu?",
     "3.073 viên chức"),
    ("Trường Đại học Khoa học Tự nhiên có bao nhiêu viên chức?",
     "490 viên chức"),
    ("Trường Đại học Ngoại ngữ có tổng số bao nhiêu người làm việc?",
     "777 người"),
    ("Trường Đại học Công nghệ có bao nhiêu vị trí việc làm được giao?",
     "345 vị trí"),
    ("Bệnh viện Đại học Y Dược có bao nhiêu viên chức?",
     "299 viên chức"),
    ("Trường Quốc tế có tổng số bao nhiêu người làm việc?",
     "208 người"),
    ("Cơ quan ĐHQGHN (VP, các Ban chức năng, Khối Đảng-đoàn thể) có bao nhiêu người làm việc?",
     "176 người"),
]


def write_qa_files(qa_pairs):
    random.seed(42)
    shuffled = list(qa_pairs)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * 0.7)
    train_qa = shuffled[:split_idx]
    test_qa = shuffled[split_idx:]

    # Save train
    with open(os.path.join(TRAIN_DIR, 'questions.txt'), 'w', encoding='utf-8') as fq, \
         open(os.path.join(TRAIN_DIR, 'reference_answers.txt'), 'w', encoding='utf-8') as fa:
        for q, a in train_qa:
            fq.write(q + '\n')
            fa.write(a + '\n')

    # Save test
    with open(os.path.join(TEST_DIR, 'questions.txt'), 'w', encoding='utf-8') as fq, \
         open(os.path.join(TEST_DIR, 'reference_answers.txt'), 'w', encoding='utf-8') as fa:
        for q, a in test_qa:
            fq.write(q + '\n')
            fa.write(a + '\n')

    print(f"Tổng: {len(qa_pairs)} cặp QA")
    print(f"Train: {len(train_qa)} cặp → {TRAIN_DIR}/")
    print(f"Test:  {len(test_qa)} cặp → {TEST_DIR}/")


if __name__ == "__main__":
    setup_dirs()
    write_qa_files(QA_DATA)
