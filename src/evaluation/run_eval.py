"""
Chạy full evaluation: Load RAG v2.0 → chạy test set → lưu output → tính metrics
"""
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

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

    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        print("=" * 60)
        print("CẢNH BÁO: Chưa thiết lập LLM_API_KEY.")
        print("Câu trả lời sẽ là placeholder. Để có kết quả thực:")
        print("  set LLM_API_KEY=your_api_key_here")
        print("=" * 60)

    print("\nKhởi tạo RAG System v2.0...")
    print("(BAAI/bge-m3 + FAISS + Reranking + GPT 120b)")
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
