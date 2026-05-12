"""
Bước 1: Data Cleaning
Đọc dữ liệu raw từ data/raw/, làm sạch chuyên sâu, xuất ra data/cleaned/
"""
import os
import re
import sys
sys.stdout.reconfigure(encoding='utf-8')

RAW_DIR = "data/raw"
CLEANED_DIR = "data/cleaned"


def clean_general(text):
    """Làm sạch chung cho mọi file."""
    # Chuẩn hóa line ending
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    # Loại bỏ khoảng trắng thừa cuối dòng
    text = '\n'.join(line.rstrip() for line in text.split('\n'))
    # Loại bỏ nhiều dòng trống liên tiếp (giữ tối đa 1 dòng trống)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def clean_lich_su(text):
    """Làm sạch dữ liệu Lịch sử VNU."""
    lines = text.split('\n')
    cleaned_lines = []

    # Các pattern caption ảnh cần loại bỏ
    caption_patterns = [
        r'^Đại học Đông Dương được chụp từ',
        r'^Tòa nhà trụ sở của Đại học Đông Dương',
        r'^Lễ ra mắt ĐHQGHN tại',
        r'^quận Hoàn Kiếm, thành phố Hà Nội$',
        r'^Nhà điều hành Đại học Quốc gia',
        r'^\(được khánh thành',
        r'^Quận Cầu Giấy',
        r'^Nhà Điều hành tại Hòa Lạc',
        r'^tại trụ sở mới ở Hòa Lạc',
        r'^Phối cảnh khu Trung tâm',
        r'^Tòa Nhà HT1 và HT2',
        r'^ở nội thành Hà Nội',
        r'^Một góc khuôn viên ĐHQGHN',
        r'^chính thức đi vào hoạt động',
        r'^Dưới đây là một số sự kiện',
        r'^:$',
        r'^Ngày$',
        r'^23 tháng 10 năm 2022',
        r'^Lễ kỷ niệm',
        r'^năm truyền thống',
        r'^Lao động hạng Nhất$',
        r'^Tại$',
    ]

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append('')
            continue

        # Bỏ caption ảnh
        skip = False
        for pattern in caption_patterns:
            if re.match(pattern, stripped):
                skip = True
                break
        if skip:
            continue

        cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)

    # Gộp các dòng bị ngắt giữa chừng (dòng ngắn + dòng tiếp theo không bắt đầu bằng "Năm")
    # Nhưng giữ nguyên các mốc năm
    result_lines = []
    for line in text.split('\n'):
        if not line.strip():
            result_lines.append('')
        elif result_lines and result_lines[-1] and not re.match(r'^(Năm |Giai đoạn |\d{4})', line) and len(result_lines[-1]) < 80 and not result_lines[-1].endswith('.') and not re.match(r'^(Năm |Giai đoạn |NHỮNG )', result_lines[-1]):
            result_lines[-1] += ' ' + line.strip()
        else:
            result_lines.append(line)

    return clean_general('\n'.join(result_lines))


def clean_su_mang(text):
    """Làm sạch dữ liệu Sứ mạng - Tầm nhìn."""
    # File này đã khá sạch, chỉ cần chuẩn hóa
    return clean_general(text)


def clean_chien_luoc(text):
    """Làm sạch dữ liệu Chiến lược phát triển."""
    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        # Bỏ dòng rác
        if stripped in ['>>>', '.']:
            continue
        if stripped.startswith('[Infographic]'):
            continue
        if stripped == 'Tải bản đầy đủ':
            continue
        if stripped == 'tại đây':
            continue
        cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)

    # Chuyển bảng chỉ tiêu chiến lược thành dạng có cấu trúc
    # Nhận diện phần bảng (bắt đầu từ "IV. Hệ thống chỉ tiêu chiến lược")
    parts = text.split('IV. Hệ thống chỉ tiêu chiến lược')
    if len(parts) == 2:
        main_text = parts[0].strip()
        table_text = parts[1].strip()
        structured_table = convert_strategy_table(table_text)
        text = main_text + '\n\nIV. HỆ THỐNG CHỈ TIÊU CHIẾN LƯỢC\n\n' + structured_table

    return clean_general(text)


def convert_strategy_table(table_text):
    """Chuyển bảng chỉ tiêu chiến lược sang dạng dễ đọc."""
    lines = [l.strip() for l in table_text.split('\n') if l.strip()]

    # Bỏ header cột
    skip_headers = ['TT', 'Tên chỉ số', 'Mục tiêu 2030', 'Mục tiêu 2035',
                    '(a)', '(b)', '(c)']

    result = []
    i = 0
    current_group = ""

    while i < len(lines):
        line = lines[i]

        if line in skip_headers:
            i += 1
            continue

        # Nhận diện nhóm chỉ tiêu (Quản trị, Đội ngũ, v.v.)
        if line in ['Quản trị', 'Đội ngũ', 'Chuyên môn', 'Tài chính',
                     'Công nghệ', 'Đô thị Hòa Lạc']:
            current_group = line
            result.append(f"\n--- {current_group} ---")
            i += 1
            continue

        # Nhận diện mã chỉ tiêu (1.1, 2.3, v.v.)
        if re.match(r'^\d+\.\d+$', line):
            code = line
            i += 1
            if i < len(lines):
                name = lines[i]
                i += 1
                target_2030 = lines[i] if i < len(lines) else "N/A"
                i += 1
                target_2035 = lines[i] if i < len(lines) else "N/A"
                i += 1
                result.append(f"Chỉ tiêu {code} ({current_group}): {name} - Mục tiêu 2030: {target_2030}, Mục tiêu 2035: {target_2035}.")
            continue

        # Bỏ qua số thứ tự nhóm đơn lẻ (1, 2, 3...)
        if re.match(r'^\d+$', line) and len(line) <= 2:
            i += 1
            continue

        i += 1

    # Thêm phần V. Tổ chức thực hiện nếu chưa có
    return '\n'.join(result)


def clean_thi_dua(text):
    """Chuyển bảng Thi đua khen thưởng thành câu tự nhiên."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    # Bỏ header
    skip = ['TT', 'Danh mục', 'Số lượng']

    result_sentences = [
        "THÀNH TÍCH THI ĐUA KHEN THƯỞNG CỦA ĐHQGHN",
        ""
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        if line in skip:
            i += 1
            continue

        # Pattern: STT. Tên danh mục \n Số lượng
        match_stt = re.match(r'^(\d+)\.$', line)
        if match_stt:
            i += 1
            if i < len(lines):
                name = lines[i]
                i += 1
                if i < len(lines) and re.match(r'^[\d.,]+$', lines[i]):
                    count = lines[i].replace('.', '')
                    i += 1
                    result_sentences.append(f"ĐHQGHN có {count} {name}.")
                else:
                    result_sentences.append(f"ĐHQGHN được trao tặng danh hiệu {name}.")
            continue

        i += 1

    return clean_general('\n'.join(result_sentences))


def clean_so_lieu(text):
    """Chuyển bảng Số liệu thống kê thành câu tự nhiên."""
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    result_sentences = [
        "SỐ LIỆU THỐNG KÊ NHÂN SỰ CỦA ĐHQGHN (cập nhật đến 31/12/2024)",
        ""
    ]

    # Bỏ header
    skip_headers = ['TT', 'Tên cơ quan, đơn vị', 'Số vị trí việc làm đã được giao hoặc phê duyệt',
                    'Tổng số', 'Trong đó', 'Viên chức', 'Người lao động',
                    '(a)', '(b)', '(c)', '(1 )= (2)+(3)', '(2)', '(3)',
                    'Cập nhật đến 31/12/2024']

    i = 0
    while i < len(lines):
        line = lines[i]

        if line in skip_headers:
            i += 1
            continue

        # Nhận diện STT đơn vị (1, 2, 3... hoặc số lớn hơn)
        if re.match(r'^\d+$', line) and int(line) <= 40:
            stt = line
            i += 1
            # Đọc tên đơn vị (có thể trên 2 dòng)
            name_parts = []
            while i < len(lines) and not re.match(r'^\d{2,}$', lines[i]):
                if re.match(r'^\d+$', lines[i]) and int(lines[i]) <= 40:
                    break
                if lines[i].startswith('(ko kể'):
                    name_parts.append(lines[i])
                    i += 1
                    continue
                name_parts.append(lines[i])
                i += 1

            if not name_parts:
                continue

            name = ' '.join(name_parts)

            # Đọc 4 số: vị trí, tổng số, viên chức, người lao động
            numbers = []
            while i < len(lines) and re.match(r'^[\d.,]+$', lines[i]) and len(numbers) < 4:
                numbers.append(lines[i].replace('.', ''))
                i += 1

            if len(numbers) >= 4:
                result_sentences.append(
                    f"{name}: {numbers[0]} vị trí việc làm được giao, "
                    f"tổng số {numbers[1]} người làm việc "
                    f"(gồm {numbers[2]} viên chức và {numbers[3]} người lao động)."
                )
            elif len(numbers) >= 1:
                result_sentences.append(
                    f"{name}: {numbers[0]} vị trí việc làm."
                )
            continue

        # Dòng Tổng
        if line == 'Tổng':
            i += 1
            numbers = []
            while i < len(lines) and re.match(r'^[\d.,]+$', lines[i]):
                numbers.append(lines[i].replace('.', ''))
                i += 1
            if len(numbers) >= 4:
                result_sentences.append(
                    f"\nTổng toàn ĐHQGHN: {numbers[0]} vị trí việc làm được giao, "
                    f"tổng số {numbers[1]} người làm việc "
                    f"(gồm {numbers[2]} viên chức và {numbers[3]} người lao động)."
                )
            continue

        i += 1

    return clean_general('\n'.join(result_sentences))


def main():
    os.makedirs(CLEANED_DIR, exist_ok=True)

    cleaners = {
        'lich-su.txt': clean_lich_su,
        'su-mang-tam-nhin.txt': clean_su_mang,
        'chien-luoc-phat-trien.txt': clean_chien_luoc,
        'thi-dua-khen-thuong.txt': clean_thi_dua,
        'so-lieu-thong-ke.txt': clean_so_lieu,
    }

    for filename, cleaner_fn in cleaners.items():
        raw_path = os.path.join(RAW_DIR, filename)
        if not os.path.exists(raw_path):
            print(f"[SKIP] {raw_path} không tồn tại")
            continue

        with open(raw_path, 'r', encoding='utf-8') as f:
            raw_text = f.read()

        cleaned_text = cleaner_fn(raw_text)

        out_path = os.path.join(CLEANED_DIR, filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)

        print(f"[OK] {filename}: {len(raw_text)} → {len(cleaned_text)} chars")


if __name__ == "__main__":
    main()
