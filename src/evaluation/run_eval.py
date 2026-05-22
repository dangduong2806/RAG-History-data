"""
Chạy full evaluation: Load RAG v2.0 → chạy test set → lưu output → tính metrics
"""
import os
import sys
from dotenv import load_dotenv
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Import RAG System
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.model.rag_pipeline import RAGSystem
from src.evaluation.evaluate import evaluate


def main():
    test_q_path = "data/test/questions.txt"
    test_a_path = "data/test/reference_answers.txt"
    output_dir = "system_outputs"
    output_path = os.path.join(output_dir, "system_output_1.txt")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(test_q_path) or not os.path.exists(test_a_path):
        print("Không tìm thấy tập test. Vui lòng chạy:")
        print("  python src/data_annotation/generate_qa.py")
        return

    llm_api_key = (os.getenv("LLM_API_KEY") or "").strip()
    hf_token = (os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or "").strip()
    api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").lower()
    has_router_token = "huggingface.co" in api_base and hf_token and "your_hf_token" not in hf_token.lower()

    if not llm_api_key and not has_router_token:
        print("=" * 60)
        print("CẢNH BÁO: Chưa thiết lập token hop le cho LLM.")
        print("Neu dung Hugging Face Router, co the set HF_TOKEN.")
        print("Neu dung nha cung cap khac, set LLM_API_KEY.")
        print("=" * 60)

    print("\nKhởi tạo RAG System v2.0...")
    print(
        f"(BAAI/bge-m3 + FAISS + Reranking + "
        f"{os.getenv('LLM_MODEL', 'gpt-120b')})"
    )
    print()
    rag = RAGSystem()

    print(f"\nĐọc câu hỏi từ {test_q_path}...")
    with open(test_q_path, 'r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]

    print(f"Tổng số câu hỏi: {len(questions)}")
    print("\nĐang tạo câu trả lời qua RAG Pipeline...")
    print("-" * 60)

    outputs = []
    for i, q in enumerate(questions):
        print(f"\n[{i+1}/{len(questions)}] Q: {q}")
        ans = rag.ask(q, verbose=True)
        ans_clean = ans.replace('\n', ' ').strip()
        outputs.append(ans_clean)
        print(f"  → A: {ans_clean[:200]}...")

    with open(output_path, 'w', encoding='utf-8') as f:
        for out in outputs:
            f.write(out + "\n")

    print(f"\n{'=' * 60}")
    print(f"Đã lưu {len(outputs)} câu trả lời → {output_path}")
    print(f"{'=' * 60}")
    print("\nKẾT QUẢ ĐÁNH GIÁ (Evaluation Metrics)")
    print("=" * 60)

    evaluate(test_a_path, output_path)


if __name__ == "__main__":
    main()
