"""
Bước 2: Smart Chunking
Cắt dữ liệu cleaned thành chunks thông minh theo ngữ nghĩa, lưu dạng JSONL.
"""
import os
import re
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

CLEANED_DIR = "data/cleaned"
CHUNKS_DIR = "data/chunks"
CHUNKS_FILE = os.path.join(CHUNKS_DIR, "chunks.jsonl")

# Map topic
TOPIC_MAP = {
    'lich-su.txt': 'Lịch sử ĐHQGHN',
    'su-mang-tam-nhin.txt': 'Sứ mệnh - Tầm nhìn ĐHQGHN',
    'chien-luoc-phat-trien.txt': 'Chiến lược phát triển ĐHQGHN',
    'thi-dua-khen-thuong.txt': 'Thi đua khen thưởng ĐHQGHN',
    'so-lieu-thong-ke.txt': 'Số liệu thống kê ĐHQGHN',
}


def chunk_lich_su(text, source):
    """Chunk lịch sử theo mốc năm."""
    chunks = []

    # Phần giới thiệu (trước NHỮNG MỐC LỊCH SỬ)
    intro_split = text.split('NHỮNG MỐC LỊCH SỬ QUAN TRỌNG CỦA ĐHQGHN')
    if len(intro_split) == 2:
        intro = intro_split[0].strip()
        if intro:
            chunks.append({
                "content": intro,
                "source": source,
                "topic": TOPIC_MAP[source],
                "section": "Giới thiệu chung về ĐHQGHN"
            })
        main_text = intro_split[1].strip()
    else:
        main_text = text

    # Tách theo mốc năm: "Năm XXXX" hoặc "Giai đoạn XXXX"
    year_pattern = r'(?=^(?:Năm \d{4}|Giai đoạn \d{4}))'
    year_sections = re.split(year_pattern, main_text, flags=re.MULTILINE)

    current_chunk = ""
    current_years = []

    for section in year_sections:
        section = section.strip()
        if not section:
            continue

        # Trích năm
        year_match = re.match(r'^(?:Năm |Giai đoạn )(\d{4})', section)
        year_label = year_match.group(0) if year_match else ""

        # Nếu chunk hiện tại + section mới quá dài (> 1500 chars), lưu và bắt đầu chunk mới
        if current_chunk and len(current_chunk) + len(section) > 1500:
            chunks.append({
                "content": current_chunk.strip(),
                "source": source,
                "topic": TOPIC_MAP[source],
                "section": f"Lịch sử: {', '.join(current_years)}"
            })
            current_chunk = ""
            current_years = []

        current_chunk += "\n\n" + section if current_chunk else section
        if year_label:
            current_years.append(year_label)

    # Chunk cuối
    if current_chunk.strip():
        chunks.append({
            "content": current_chunk.strip(),
            "source": source,
            "topic": TOPIC_MAP[source],
            "section": f"Lịch sử: {', '.join(current_years)}" if current_years else "Lịch sử: phần cuối"
        })

    return chunks


def chunk_su_mang(text, source):
    """Sứ mạng ngắn → 1 chunk duy nhất."""
    return [{
        "content": text.strip(),
        "source": source,
        "topic": TOPIC_MAP[source],
        "section": "Sứ mệnh, Tầm nhìn, Giá trị cốt lõi"
    }]


def chunk_chien_luoc(text, source):
    """Chunk chiến lược theo section/trụ cột."""
    chunks = []

    # Tách theo heading La Mã (I., II., III., IV., V.)
    section_pattern = r'(?=^(?:I{1,3}V?|VI{0,3})\.\s)'
    sections = re.split(section_pattern, text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        # Trích heading
        heading_match = re.match(r'^(I{1,3}V?|VI{0,3})\.\s*(.*?)$', section, re.MULTILINE)
        heading = heading_match.group(2).strip() if heading_match else "Chiến lược phát triển"

        # Nếu section quá dài, chia nhỏ theo sub-section (1., 2., 3.,...)
        if len(section) > 1500:
            sub_pattern = r'(?=^\d+\.\s+[A-ZĐ])'
            sub_sections = re.split(sub_pattern, section, flags=re.MULTILINE)

            current_chunk = ""
            for sub in sub_sections:
                sub = sub.strip()
                if not sub:
                    continue
                if current_chunk and len(current_chunk) + len(sub) > 1500:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "source": source,
                        "topic": TOPIC_MAP[source],
                        "section": f"Chiến lược: {heading}"
                    })
                    current_chunk = ""
                current_chunk += "\n\n" + sub if current_chunk else sub

            if current_chunk.strip():
                chunks.append({
                    "content": current_chunk.strip(),
                    "source": source,
                    "topic": TOPIC_MAP[source],
                    "section": f"Chiến lược: {heading}"
                })
        else:
            chunks.append({
                "content": section,
                "source": source,
                "topic": TOPIC_MAP[source],
                "section": f"Chiến lược: {heading}"
            })

    return chunks


def chunk_thi_dua(text, source):
    """Thi đua khen thưởng → 1 chunk (đã ngắn sau cleaning)."""
    return [{
        "content": text.strip(),
        "source": source,
        "topic": TOPIC_MAP[source],
        "section": "Thành tích thi đua khen thưởng"
    }]


def chunk_so_lieu(text, source):
    """Số liệu thống kê → chia theo nhóm đơn vị."""
    lines = text.strip().split('\n')
    chunks = []

    # Header
    header = lines[0] if lines else ""

    # Chia thành nhóm: Trường ĐH, Viện, Trung tâm, Ban, Cơ quan
    groups = {
        "Trường Đại học": [],
        "Viện nghiên cứu": [],
        "Trung tâm và đơn vị phục vụ": [],
        "Ban quản lý và Cơ quan": [],
        "Tổng kết": [],
    }

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        if line.startswith("Trường ") or line.startswith("Khoa "):
            groups["Trường Đại học"].append(line)
        elif line.startswith("Viện "):
            groups["Viện nghiên cứu"].append(line)
        elif line.startswith("Trung tâm ") or line.startswith("Bệnh viện") or line.startswith("Nhà Xuất bản") or line.startswith("Tạp chí"):
            groups["Trung tâm và đơn vị phục vụ"].append(line)
        elif line.startswith("Ban ") or line.startswith("Cơ quan "):
            groups["Ban quản lý và Cơ quan"].append(line)
        elif line.startswith("Tổng "):
            groups["Tổng kết"].append(line)

    for group_name, group_lines in groups.items():
        if group_lines:
            content = header + "\n\n" + "\n".join(group_lines)
            chunks.append({
                "content": content,
                "source": source,
                "topic": TOPIC_MAP[source],
                "section": f"Thống kê nhân sự: {group_name}"
            })

    return chunks


def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    chunkers = {
        'lich-su.txt': chunk_lich_su,
        'su-mang-tam-nhin.txt': chunk_su_mang,
        'chien-luoc-phat-trien.txt': chunk_chien_luoc,
        'thi-dua-khen-thuong.txt': chunk_thi_dua,
        'so-lieu-thong-ke.txt': chunk_so_lieu,
    }

    all_chunks = []

    for filename, chunker_fn in chunkers.items():
        filepath = os.path.join(CLEANED_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[SKIP] {filepath} không tồn tại")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = chunker_fn(text, filename)
        all_chunks.extend(chunks)
        print(f"[OK] {filename} → {len(chunks)} chunks")

    # Lưu dạng JSONL
    with open(CHUNKS_FILE, 'w', encoding='utf-8') as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')

    print(f"\nTổng cộng: {len(all_chunks)} chunks → {CHUNKS_FILE}")

    # In thống kê kích thước
    sizes = [len(c["content"]) for c in all_chunks]
    print(f"Kích thước chunk: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)//len(sizes)}")


if __name__ == "__main__":
    main()
