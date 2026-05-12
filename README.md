# Assignment 2: End-to-end NLP System Building — RAG for VNU

Hệ thống **Retrieval-Augmented Generation (RAG)** trả lời câu hỏi về Đại học Quốc gia Hà Nội (VNU).

## Kiến trúc Pipeline

```
Raw Text (crawl 5 trang VNU)
    │
    ▼  [Data Cleaning]
Cleaned Text (loại rác, chuẩn hóa bảng → câu tự nhiên)
    │
    ▼  [Smart Chunking]
Chunks (theo mốc năm / section / nhóm đơn vị)
    │
    ▼  [Embedding: BAAI/bge-m3]
FAISS Index (vector database)
    │
    ▼  [Retrieval: top-10]
Candidate Chunks
    │
    ▼  [Reranking: BAAI/bge-reranker-v2-m3]
Top-3 Chunks
    │
    ▼  [Generation: GPT 120b API]
Final Answer
```

## Project Structure

```text
.
├── data/
│   ├── raw/                    # Dữ liệu crawl thô từ VNU
│   ├── cleaned/                # Dữ liệu đã làm sạch
│   ├── chunks/                 # Chunks JSONL cho embedding
│   ├── test/                   # Test set (questions.txt, reference_answers.txt)
│   └── train/                  # Train set (questions.txt, reference_answers.txt)
├── src/
│   ├── data_collection/        # Web crawler
│   │   └── crawler.py
│   ├── data_processing/        # Data cleaning + chunking
│   │   ├── data_cleaner.py
│   │   └── chunker.py
│   ├── data_annotation/        # QA pair generator
│   │   └── generate_qa.py
│   ├── model/                  # RAG pipeline (FAISS + Reranking + LLM)
│   │   └── rag_pipeline.py
│   └── evaluation/             # Evaluation metrics
│       ├── evaluate.py
│       └── run_eval.py
├── faiss_index/                # FAISS vector index
├── system_outputs/             # RAG system outputs
├── contributions.md
├── github_url.txt
└── README.md
```

## Setup & Installation

```bash
pip install requests beautifulsoup4 sentence-transformers faiss-cpu openai
```

## Cấu hình API Key

```bash
# Windows CMD
set LLM_API_KEY=your_api_key_here

# Windows PowerShell
$env:LLM_API_KEY="your_api_key_here"

# Tùy chọn: đổi model hoặc base URL
set LLM_MODEL=gpt-120b
set LLM_API_BASE=https://api.openai.com/v1
```

## Chạy từng bước

### 1. Crawl dữ liệu
```bash
python src/data_collection/crawler.py
```

### 2. Làm sạch dữ liệu
```bash
python src/data_processing/data_cleaner.py
```

### 3. Chunking thông minh
```bash
python src/data_processing/chunker.py
```

### 4. Tạo dữ liệu QA (test/train)
```bash
python src/data_annotation/generate_qa.py
```

### 5. Test hệ thống RAG
```bash
python src/model/rag_pipeline.py
```

### 6. Chạy Evaluation đầy đủ
```bash
python src/evaluation/run_eval.py
```

## Models Used

| Component | Model | Source |
|-----------|-------|--------|
| Embedding | `BAAI/bge-m3` | HuggingFace |
| Reranker | `BAAI/bge-reranker-v2-m3` | HuggingFace |
| Generator | `GPT 120b` | API |
| Vector DB | FAISS (IndexFlatIP) | Meta AI |

## Evaluation Metrics

- **Exact Match (EM)**: Tỷ lệ câu trả lời khớp chính xác
- **F1 Score**: Token-level F1 giữa prediction và reference
- **Recall**: Token-level recall
