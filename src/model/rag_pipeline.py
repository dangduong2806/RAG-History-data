"""
RAG Pipeline v2.0
Pipeline: Chunks → BAAI/bge-m3 Embedding → FAISS → Retrieve(top-10) → Rerank(top-3) → GPT 120b → Answer
"""
import os
import json
import sys
import numpy as np
from dotenv import load_dotenv

# This pipeline uses PyTorch only. Avoid importing TensorFlow through
# transformers when TensorFlow/protobuf versions in the environment differ.
os.environ.setdefault("USE_TF", "0")

# Load biến môi trường từ file .env (nếu có)
load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')


import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


class RAGSystem:
    def __init__(
        self,
        chunks_file="data/chunks/chunks.jsonl",
        index_dir="faiss_index",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-base",
        top_k_retrieve=10,
        top_k_rerank=3,
        few_shot_questions_file = "data/train/questions.txt",
        few_shot_answers_file="data/train/reference_answers.txt",
        num_few_shot=5
    ):
        self.chunks_file = chunks_file
        self.index_dir = index_dir
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        
        self.num_few_shot = num_few_shot
        self.few_shot_examples = self._load_few_shot_examples(
            few_shot_questions_file,
            few_shot_answers_file,
            num_few_shot
        )

        # --- LLM API config ---
        self.model_name = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )

        self.model.eval()

        # --- Load chunks ---
        print(f"Loading chunks from {chunks_file}...")
        self.chunks = self._load_chunks()
        print(f"  → {len(self.chunks)} chunks loaded")

        # --- Load embedding model ---
        print(f"Loading embedding model: {embedding_model}...")
        self.embedder = SentenceTransformer(embedding_model)

        # --- Load reranker ---
        print(f"Loading reranker model: {reranker_model}...")
        self.reranker = CrossEncoder(reranker_model)

        # --- Build or load FAISS index ---
        self.index, self.chunk_texts = self._init_faiss()

    def _load_chunks(self):
        chunks = []
        with open(self.chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks
    
    def _load_few_shot_examples(self, questions_file, answers_file, max_examples):
        if max_examples <= 0:
            return []
        
        if not os.path.exists(questions_file) or not os.path.exists(answers_file):
            print("Không tìm thấy file data cho few-shot")
            return []

        with open(questions_file, "r", encoding="utf-8") as fq:
            questions = [line.strip() for line in fq if line.strip()]
        with open(answers_file, "r", encoding="utf-8") as fa:
            answers = [line.strip() for line in fa if line.strip()]

        examples = []
        for question, answer in zip(questions, answers):
            examples.append({
                "question": question,
                "answer": answer,
            })
        
        return examples[:max_examples]
        

    def _init_faiss(self):
        index_path = os.path.join(self.index_dir, "index.faiss")
        meta_path = os.path.join(self.index_dir, "chunks_meta.json")

        chunk_texts = [c["content"] for c in self.chunks]

        if os.path.exists(index_path) and os.path.exists(meta_path):
            print("Loading existing FAISS index...")
            index = faiss.read_index(index_path)
            with open(meta_path, 'r', encoding='utf-8') as f:
                stored_meta = json.load(f)
            # Kiểm tra xem index có khớp số chunks không
            if index.ntotal == len(chunk_texts):
                return index, chunk_texts
            else:
                print(f"  → Index mismatch ({index.ntotal} vs {len(chunk_texts)}), rebuilding...")

        print("Building FAISS index from chunks...")
        # Encode tất cả chunks
        embeddings = self.embedder.encode(
            chunk_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=8
        )
        embeddings = np.array(embeddings, dtype='float32')

        # Tạo FAISS index (Inner Product vì đã normalize → cosine similarity)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Lưu index
        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(index, index_path)
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({"num_chunks": len(chunk_texts)}, f)

        print(f"  → FAISS index built: {index.ntotal} vectors, dim={dim}")
        return index, chunk_texts

    def retrieve(self, query, top_k=None):
        """Bước 1: Retrieval - tìm top-K chunks gần nhất bằng FAISS."""
        if top_k is None:
            top_k = self.top_k_retrieve

        # Encode query
        query_embedding = self.embedder.encode(
            [query], normalize_embeddings=True
        ).astype('float32')

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append({
                    "rank": i + 1,
                    "score": float(score),
                    "content": self.chunks[idx]["content"],
                    "source": self.chunks[idx].get("source", ""),
                    "topic": self.chunks[idx].get("topic", ""),
                    "section": self.chunks[idx].get("section", ""),
                })
        return results

    def rerank(self, query, retrieved_docs, top_k=None):
        """Bước 2: Reranking - dùng cross-encoder để sắp xếp lại."""
        if top_k is None:
            top_k = self.top_k_rerank

        if not retrieved_docs:
            return []

        # Tạo pairs (query, doc) cho cross-encoder
        pairs = [(query, doc["content"]) for doc in retrieved_docs]
        rerank_scores = self.reranker.predict(pairs)

        # Gán điểm rerank vào mỗi doc
        for doc, score in zip(retrieved_docs, rerank_scores):
            doc["rerank_score"] = float(score)

        # Sắp xếp theo rerank_score giảm dần
        reranked = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def generate_answer(self, query, contexts):
        """Bước 3: Generation - gọi LLM API sinh câu trả lời."""
        context_text = ""
        for i, ctx in enumerate(contexts):
            context_text += f"\n--- Ngữ cảnh {i+1} (Nguồn: {ctx.get('topic', 'N/A')}) ---\n"
            context_text += ctx["content"] + "\n"
        
        few_shot_text = ""
        for i, example in enumerate(self.few_shot_examples):
            few_shot_text += f"\nVí dụ {i+1}:\n"
            few_shot_text += f"Câu hỏi: {example['question']}\n"
            few_shot_text += f"Trả lời: {example['answer']}\n"
        

        prompt = f"""Bạn là một chuyên gia về Đại học Quốc gia Hà Nội (ĐHQGHN/VNU). 
Dựa vào CÁC NGỮ CẢNH được cung cấp bên dưới, hãy trả lời câu hỏi một cách NGẮN GỌN, CHÍNH XÁC và ĐẦY ĐỦ.
Chỉ sử dụng thông tin từ ngữ cảnh. Nếu ngữ cảnh không chứa câu trả lời, hãy nói "Không tìm thấy thông tin trong tài liệu."

Dưới đây là một số ví dụ về cách trả lời:
{few_shot_text}

CÁC NGỮ CẢNH:
{context_text}

Câu hỏi: {query}

Trả lời ngắn gọn:"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "Bạn là trợ lý trả lời câu hỏi về Đại học Quốc gia Hà Nội. Trả lời ngắn gọn, chính xác dựa trên ngữ cảnh."
                },
                {
                    "role":"user",
                    "content": prompt
                }
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **model_inputs,
                    max_new_tokens=300,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]

            return response.strip()
        except Exception as e:
            return f"[Lỗi Qwen local: {str(e)}]"

    def ask(self, query, verbose=False):
        """Pipeline đầy đủ: Retrieve → Rerank → Generate."""
        # Bước 1: Retrieve
        retrieved = self.retrieve(query)
        if verbose:
            print(f"  [Retrieve] Top-{len(retrieved)} candidates found")

        # Bước 2: Rerank
        reranked = self.rerank(query, retrieved)
        if verbose:
            print(f"  [Rerank] Top-{len(reranked)} after reranking:")
            for doc in reranked:
                print(f"    - [{doc.get('section', '')}] score={doc['rerank_score']:.4f}")

        # Bước 3: Generate
        answer = self.generate_answer(query, reranked)
        return answer

    def test_retrieval(self, query, top_k=5):
        """Hàm test: chỉ retrieval + rerank, in ra kết quả để kiểm tra."""
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        retrieved = self.retrieve(query, top_k=top_k)
        print(f"\n--- Retrieval Results (top-{len(retrieved)}) ---")
        for doc in retrieved:
            print(f"  #{doc['rank']} [score={doc['score']:.4f}] {doc['section']}")
            print(f"    {doc['content'][:150]}...")

        reranked = self.rerank(query, retrieved, top_k=3)
        print(f"\n--- After Reranking (top-{len(reranked)}) ---")
        for i, doc in enumerate(reranked):
            print(f"  #{i+1} [rerank={doc['rerank_score']:.4f}] {doc['section']}")
            print(f"    {doc['content'][:150]}...")


if __name__ == "__main__":
    rag = RAGSystem()
    print("\n" + "="*60)
    print("RAG System v2.0 đã sẵn sàng!")
    print("="*60)

    # Test retrieval
    test_queries = [
        "Đại học Đông Dương được thành lập vào năm nào?",
        "Sứ mệnh của ĐHQGHN là gì?",
        "Trường Đại học Khoa học Tự nhiên có bao nhiêu viên chức?",
    ]
    for q in test_queries:
        rag.test_retrieval(q, top_k=5)
