"""
Step 2: smart chunking
Split cleaned data into semantically meaningful chunks and save as JSONL.
"""
import json
import os
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

CLEANED_DIR = os.path.join(PROJECT_ROOT, "data", "cleaned")
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "data", "chunks")
CHUNKS_FILE = os.path.join(CHUNKS_DIR, "chunks.jsonl")

TARGET_CHUNK_SIZE = 900
HARD_MAX_CHUNK_SIZE = 1200

TOPIC_MAP = {
    "lich-su.txt": "Lịch sử ĐHQGHN",
    "su-mang-tam-nhin.txt": "Sứ mệnh - Tầm nhìn ĐHQGHN",
    "chien-luoc-phat-trien.txt": "Chiến lược phát triển ĐHQGHN",
    "thi-dua-khen-thuong.txt": "Thi đua khen thưởng ĐHQGHN",
    "so-lieu-thong-ke.txt": "Số liệu thống kê ĐHQGHN",
}


def make_chunk(content, source, section):
    return {
        "content": content.strip(),
        "source": source,
        "topic": TOPIC_MAP[source],
        "section": section,
    }


def split_by_sentences(text, max_size=HARD_MAX_CHUNK_SIZE):
    text = text.strip()
    if len(text) <= max_size:
        return [text] if text else []

    pieces = re.split(r"(?<=[\.\!\?\;])\s+", text)
    chunks = []
    current = ""

    for piece in pieces:
        piece = piece.strip()
        if not piece:
            continue

        if len(piece) > max_size:
            words = piece.split()
            long_current = ""
            for word in words:
                candidate = f"{long_current} {word}".strip()
                if long_current and len(candidate) > max_size:
                    chunks.append(long_current)
                    long_current = word
                else:
                    long_current = candidate
            if long_current:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(long_current)
            continue

        candidate = f"{current} {piece}".strip()
        if current and len(candidate) > max_size:
            chunks.append(current)
            current = piece
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def split_semantic_block(text, target_size=TARGET_CHUNK_SIZE, hard_max=HARD_MAX_CHUNK_SIZE):
    text = text.strip()
    if not text:
        return []
    if len(text) <= hard_max:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    for para in paragraphs:
        if len(para) > hard_max:
            sentence_chunks = split_by_sentences(para, max_size=hard_max)
            for sentence_chunk in sentence_chunks:
                candidate = f"{current}\n\n{sentence_chunk}".strip() if current else sentence_chunk
                if current and len(candidate) > target_size:
                    chunks.append(current)
                    current = sentence_chunk
                else:
                    current = candidate
            continue

        candidate = f"{current}\n\n{para}".strip() if current else para
        if current and len(candidate) > target_size:
            chunks.append(current)
            current = para
        else:
            current = candidate

    if current:
        chunks.append(current)

    normalized = []
    for chunk in chunks:
        if len(chunk) > hard_max:
            normalized.extend(split_by_sentences(chunk, max_size=hard_max))
        else:
            normalized.append(chunk)
    return normalized


def split_history_section(section_text):
    section_text = section_text.strip()
    if not section_text:
        return []

    heading_match = re.match(r"^(Năm \d{4}|Giai đoạn \d{4}(?:\s*-\s*\d{4})?)", section_text)
    heading = heading_match.group(1) if heading_match else "Mốc lịch sử"
    body = section_text[len(heading):].strip()

    if not body:
        return [(heading, section_text)]

    sub_entries = re.split(r"(?=^Ngày \d{1,2} tháng \d{1,2} năm \d{4})", body, flags=re.MULTILINE)
    sub_entries = [x.strip() for x in sub_entries if x.strip()]

    if len(sub_entries) <= 1:
        return [(heading, part) for part in split_semantic_block(section_text)]

    chunks = []
    current = heading
    part_id = 1
    for entry in sub_entries:
        candidate = f"{current}\n{entry}".strip()
        if current != heading and len(candidate) > TARGET_CHUNK_SIZE:
            chunks.append((f"{heading} (phan {part_id})", current))
            current = f"{heading}\n{entry}"
            part_id += 1
        else:
            current = candidate

    if current.strip():
        label = heading if part_id == 1 else f"{heading} (phần {part_id})"
        chunks.append((label, current))

    normalized = []
    for label, content in chunks:
        split_parts = split_semantic_block(content)
        if len(split_parts) == 1:
            normalized.append((label, split_parts[0]))
        else:
            for idx, part in enumerate(split_parts, start=1):
                normalized.append((f"{label} - phần {idx}", part))
    return normalized


def chunk_lich_su(text, source):
    chunks = []

    intro_split = text.split("NHỮNG MỐC LỊCH SỬ QUAN TRỌNG CỦA ĐHQGHN")
    if len(intro_split) == 2:
        intro = intro_split[0].strip()
        if intro:
            for idx, part in enumerate(split_semantic_block(intro), start=1):
                section = "Giới thiệu chung về ĐHQGHN"
                if idx > 1:
                    section = f"{section} - phần {idx}"
                chunks.append(make_chunk(part, source, section))
        main_text = intro_split[1].strip()
    else:
        main_text = text

    year_pattern = r"(?=^(?:Năm \d{4}|Giai đoạn \d{4}))"
    year_sections = re.split(year_pattern, main_text, flags=re.MULTILINE)

    pending_small = []
    pending_labels = []

    def flush_pending():
        nonlocal pending_small, pending_labels
        if pending_small:
            content = "\n\n".join(pending_small)
            section = f"Lich su: {', '.join(pending_labels)}"
            chunks.append(make_chunk(content, source, section))
            pending_small = []
            pending_labels = []

    for section in year_sections:
        section = section.strip()
        if not section:
            continue

        year_match = re.match(r"^(Năm \d{4}|Giai đoạn \d{4}(?:\s*-\s*\d{4})?)", section)
        year_label = year_match.group(1) if year_match else "Mốc lịch sử"

        split_parts = split_history_section(section)
        if len(split_parts) == 1 and len(split_parts[0][1]) < 500:
            candidate = ("\n\n".join(pending_small + [split_parts[0][1]])).strip()
            if pending_small and len(candidate) > TARGET_CHUNK_SIZE:
                flush_pending()
            pending_small.append(split_parts[0][1])
            pending_labels.append(year_label)
            continue

        flush_pending()
        for label, content in split_parts:
            chunks.append(make_chunk(content, source, f"Lịch sử: {label}"))

    flush_pending()
    return chunks


def chunk_su_mang(text, source):
    return [make_chunk(text.strip(), source, "Sứ mệnh, Tầm nhìn, Giá trị cốt lõi")]


def split_strategy_section(section_text, heading):
    numbered_subsections = re.split(
        r"(?=^\d+\.\s+(?:[A-ZĐ0-9]|Mục tiêu|Quan điểm|Hệ thống|Trụ cột|Chỉ tiêu))",
        section_text,
        flags=re.MULTILINE,
    )
    numbered_subsections = [x.strip() for x in numbered_subsections if x.strip()]

    if len(numbered_subsections) <= 1:
        return split_semantic_block(section_text)

    chunks = []
    current = ""
    for sub in numbered_subsections:
        candidate = f"{current}\n\n{sub}".strip() if current else sub
        if current and len(candidate) > TARGET_CHUNK_SIZE:
            chunks.append(current)
            current = sub
        else:
            current = candidate

    if current:
        chunks.append(current)

    normalized = []
    for chunk in chunks:
        normalized.extend(split_semantic_block(chunk))
    return normalized


def chunk_chien_luoc(text, source):
    chunks = []
    section_pattern = r"(?=^(?:I{1,3}V?|VI{0,3})\.\s)"
    sections = re.split(section_pattern, text, flags=re.MULTILINE)

    for section in sections:
        section = section.strip()
        if not section:
            continue

        heading_match = re.match(r"^(I{1,3}V?|VI{0,3})\.\s*(.*?)$", section, re.MULTILINE)
        heading = heading_match.group(2).strip() if heading_match else "Chien luoc phat trien"
        split_parts = split_strategy_section(section, heading)

        if len(split_parts) == 1:
            chunks.append(make_chunk(split_parts[0], source, f"Chiến lược: {heading}"))
        else:
            for idx, part in enumerate(split_parts, start=1):
                chunks.append(make_chunk(part, source, f"Chiến lược: {heading} - phần {idx}"))

    return chunks


def chunk_thi_dua(text, source):
    return [make_chunk(text.strip(), source, "Thành tích thi đua khen thưởng")]


def chunk_so_lieu(text, source):
    lines = text.strip().split("\n")
    chunks = []
    header = lines[0] if lines else ""

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
        elif (
            line.startswith("Trung tâm ")
            or line.startswith("Bệnh viện")
            or line.startswith("Nhà Xuất bản")
            or line.startswith("Tạp chí")
        ):
            groups["Trung tâm và đơn vị phục vụ"].append(line)
        elif line.startswith("Ban ") or line.startswith("Cơ quan "):
            groups["Ban quản lý và Cơ quan"].append(line)
        elif line.startswith("Tổng "):
            groups["Tổng kết"].append(line)

    for group_name, group_lines in groups.items():
        if group_lines:
            content = header + "\n\n" + "\n".join(group_lines)
            chunks.append(make_chunk(content, source, f"Thống kê nhân sự: {group_name}"))

    return chunks


def main():
    os.makedirs(CHUNKS_DIR, exist_ok=True)

    chunkers = {
        "lich-su.txt": chunk_lich_su,
        "su-mang-tam-nhin.txt": chunk_su_mang,
        "chien-luoc-phat-trien.txt": chunk_chien_luoc,
        "thi-dua-khen-thuong.txt": chunk_thi_dua,
        "so-lieu-thong-ke.txt": chunk_so_lieu,
    }

    all_chunks = []

    for filename, chunker_fn in chunkers.items():
        filepath = os.path.join(CLEANED_DIR, filename)
        if not os.path.exists(filepath):
            print(f"[SKIP] {filepath} không tồn tại")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunker_fn(text, filename)
        all_chunks.extend(chunks)
        print(f"[OK] {filename} -> {len(chunks)} chunks")

    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\nTổng cộng: {len(all_chunks)} chunks -> {CHUNKS_FILE}")
    sizes = [len(c["content"]) for c in all_chunks]
    print(f"Kích thước chunk: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes) // len(sizes)}")


if __name__ == "__main__":
    main()
