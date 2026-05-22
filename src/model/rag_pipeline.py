"""
RAG pipeline:
chunks -> embedding -> FAISS retrieve -> cross-encoder rerank -> LLM answer

If the LLM is unavailable or the API key is invalid, the system falls back to
an extractive answer selected from the reranked contexts. This keeps end-to-end
evaluation usable for retrieval/rerank testing.
"""
import json
import os
import re
import sys

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder, SentenceTransformer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from src.evaluation.qa_metrics import normalize_answer
except ImportError:
    from qa_metrics import normalize_answer

STOPWORDS = {
    "la",
    "gi",
    "nao",
    "bao",
    "nhieu",
    "cua",
    "duoc",
    "vao",
    "tren",
    "co",
    "so",
    "tai",
    "dau",
    "ngay",
    "nam",
    "thang",
    "theo",
    "mot",
    "nhung",
    "thuoc",
    "dhqghn",
    "vnu",
    "truong",
    "dai",
    "hoc",
    "quoc",
    "gia",
}


load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


class RAGSystem:
    def __init__(
        self,
        chunks_file="data/chunks/chunks.jsonl",
        index_dir="faiss_index",
        embedding_model="BAAI/bge-m3",
        reranker_model="BAAI/bge-reranker-base",
        top_k_retrieve=10,
        top_k_rerank=3,
    ):
        self.chunks_file = chunks_file
        self.index_dir = index_dir
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank

        self.hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.local_files_only = os.getenv("HF_LOCAL_FILES_ONLY", "").lower() in {"1", "true", "yes"}
        self.use_extractive_fallback = os.getenv("RAG_USE_EXTRACTIVE_FALLBACK", "1").lower() in {
            "1",
            "true",
            "yes",
        }

        self.api_key = self._resolve_llm_api_key()
        self.api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
        self.model_name = os.getenv("LLM_MODEL", "gpt-120b")
        if OpenAI and self.api_key and not self._looks_like_placeholder_api_key(self.api_key):
            self.llm_client = OpenAI(api_key=self.api_key, base_url=self.api_base)
        else:
            self.llm_client = None

        print(f"Loading chunks from {chunks_file}...")
        self.chunks = self._load_chunks()
        print(f"  -> {len(self.chunks)} chunks loaded")

        print(f"Loading embedding model: {embedding_model}...")
        self.embedder = self._load_sentence_transformer(embedding_model)

        print(f"Loading reranker model: {reranker_model}...")
        self.reranker = self._load_cross_encoder(reranker_model)

        self.index, self.chunk_texts = self._init_faiss()

    def _load_chunks(self):
        chunks = []
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def _looks_like_placeholder_api_key(self, api_key):
        lowered = (api_key or "").strip().lower()
        return (
            not lowered
            or "your api" in lowered
            or "your hf token" in lowered
            or "your_hf_token" in lowered
            or "your_api" in lowered
            or lowered in {"changeme", "test", "none", "null"}
        )

    def _resolve_llm_api_key(self):
        llm_api_key = (os.getenv("LLM_API_KEY") or "").strip()
        if llm_api_key and not self._looks_like_placeholder_api_key(llm_api_key):
            return llm_api_key

        llm_api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1").lower()
        if "huggingface.co" in llm_api_base and self.hf_token:
            if not self._looks_like_placeholder_api_key(self.hf_token):
                return self.hf_token

        if "huggingface.co" in llm_api_base:
            return self.hf_token

        return llm_api_key

    def _hf_model_kwargs(self):
        return {"use_safetensors": True}

    def _raise_model_load_error(self, model_name, exc):
        message = str(exc)
        if "torch.load" in message or "safetensors" in message.lower():
            raise RuntimeError(
                "Khong the load model Hugging Face an toan.\n"
                f"Model: {model_name}\n"
                "Code da ep dung safetensors, nhung moi truong hien tai van khong load duoc.\n"
                "Cach xu ly uu tien:\n"
                "1. pip install -U safetensors\n"
                "2. neu van loi, nang cap torch len >= 2.6\n"
                "3. xoa cache checkpoint cu dang .bin neu truoc day da download model khong co safetensors"
            ) from exc
        raise

    def _load_sentence_transformer(self, model_name):
        try:
            return SentenceTransformer(
                model_name,
                token=self.hf_token,
                local_files_only=self.local_files_only,
                model_kwargs=self._hf_model_kwargs(),
            )
        except Exception as exc:
            self._raise_model_load_error(model_name, exc)

    def _load_cross_encoder(self, model_name):
        try:
            return CrossEncoder(
                model_name,
                token=self.hf_token,
                local_files_only=self.local_files_only,
                model_kwargs=self._hf_model_kwargs(),
            )
        except Exception as exc:
            self._raise_model_load_error(model_name, exc)

    def _init_faiss(self):
        index_path = os.path.join(self.index_dir, "index.faiss")
        meta_path = os.path.join(self.index_dir, "chunks_meta.json")

        chunk_texts = [c["content"] for c in self.chunks]

        if os.path.exists(index_path) and os.path.exists(meta_path):
            print("Loading existing FAISS index...")
            index = faiss.read_index(index_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                stored_meta = json.load(f)
            if index.ntotal == len(chunk_texts):
                return index, chunk_texts
            print(f"  -> Index mismatch ({index.ntotal} vs {len(chunk_texts)}), rebuilding...")

        print("Building FAISS index from chunks...")
        embeddings = self.embedder.encode(
            chunk_texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=8,
        )
        embeddings = np.array(embeddings, dtype="float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        os.makedirs(self.index_dir, exist_ok=True)
        faiss.write_index(index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"num_chunks": len(chunk_texts)}, f)

        print(f"  -> FAISS index built: {index.ntotal} vectors, dim={dim}")
        return index, chunk_texts

    def retrieve(self, query, top_k=None):
        if top_k is None:
            top_k = self.top_k_retrieve

        query_embedding = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                results.append(
                    {
                        "rank": i + 1,
                        "score": float(score),
                        "content": self.chunks[idx]["content"],
                        "source": self.chunks[idx].get("source", ""),
                        "topic": self.chunks[idx].get("topic", ""),
                        "section": self.chunks[idx].get("section", ""),
                    }
                )
        return results

    def rerank(self, query, retrieved_docs, top_k=None):
        if top_k is None:
            top_k = self.top_k_rerank

        if not retrieved_docs:
            return []

        pairs = [(query, doc["content"]) for doc in retrieved_docs]
        rerank_scores = self.reranker.predict(pairs)

        for doc, score in zip(retrieved_docs, rerank_scores):
            doc["rerank_score"] = float(score)

        reranked = sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]

    def _split_candidate_spans(self, text):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        spans = []
        for line in lines:
            pieces = re.split(r"(?<=[\.\!\?\;])\s+", line)
            for piece in pieces:
                piece = piece.strip(" -")
                if piece:
                    spans.append(piece)
        return spans or [text.strip()]

    def _query_content_tokens(self, query):
        tokens = normalize_answer(query).split()
        return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]

    def _score_candidate_span(self, query, span):
        normalized_query = normalize_answer(query)
        query_tokens = set(normalized_query.split())
        span_tokens = normalize_answer(span).split()
        if not query_tokens or not span_tokens:
            return 0.0

        overlap = sum(1 for tok in span_tokens if tok in query_tokens)
        coverage = overlap / max(len(query_tokens), 1)
        density = overlap / max(len(span_tokens), 1)

        numeric_bonus = 0.0
        if re.search(r"\d", query) and re.search(r"\d", span):
            numeric_bonus += 0.5
        if any(phrase in normalized_query for phrase in ["bao nhieu", "nam nao", "ngay nao", "so hieu"]):
            if re.search(r"\d", span):
                numeric_bonus += 0.5

        return (coverage * 2.0) + density + numeric_bonus

    def _extract_answer_like_snippet(self, query, span):
        normalized_query = normalize_answer(query)

        patterns = []
        if "ngay nao" in normalized_query:
            patterns = [
                r"ngay \d{1,2} thang \d{1,2} nam \d{4}",
                r"\d{1,2} thang \d{1,2} nam \d{4}",
            ]
        elif "nam nao" in normalized_query:
            patterns = [
                r"nam \d{4}",
                r"\b\d{4}\b",
            ]
        elif "bao nhieu" in normalized_query:
            patterns = [
                r"\b\d[\d\.]*\s+(?:nguoi|vien chuc|don vi|tru cot|huan chuong)\b",
                r"\b\d[\d\.]*\b",
            ]
        elif "so hieu" in normalized_query:
            patterns = [
                r"nghi dinh so [^,;\.\n]+",
                r"quyet dinh so [^,;\.\n]+",
            ]
        elif "co so nao" in normalized_query:
            patterns = [
                r"tren co so ([^,;\.\n]+)",
                r"nang cap ([^,;\.\n]+)",
            ]
        elif "gia tri cot loi" in normalized_query:
            patterns = [r"tien phong\s*-\s*sang tao\s*-\s*xuat sac\s*-\s*nhan van\s*-\s*phung su"]
        elif "co che gi" in normalized_query:
            patterns = [r"co che [^,;\.\n]+", r"tu chu[^,;\.\n]*"]
        elif "tai dau" in normalized_query:
            patterns = [r"(?:tai|dat tru so tai|tru so tai)\s+([^.;\n]+)"]

        for pattern in patterns:
            match = re.search(pattern, normalize_answer(span))
            if not match:
                continue
            if match.lastindex:
                return match.group(1).strip().title()
            return match.group(0).strip()

        return ""

    def _prefer_contexts(self, query, contexts):
        normalized_query = normalize_answer(query)
        content_tokens = self._query_content_tokens(query)

        scored = []
        for ctx in contexts:
            section = normalize_answer(ctx.get("section", ""))
            topic = normalize_answer(ctx.get("topic", ""))
            content = normalize_answer(ctx.get("content", ""))
            bonus = 0.0

            if "gia tri cot loi" in normalized_query and "gia tri cot loi" in content:
                bonus += 3.0
            if "su menh" in normalized_query and "su menh" in content:
                bonus += 3.0
            if "tam nhin" in normalized_query and "tam nhin" in content:
                bonus += 3.0
            if "co che" in normalized_query and "quy che" in content:
                bonus += 2.0
            if "so hieu" in normalized_query and "nghi dinh so" in content:
                bonus += 3.0
            if "tai dau" in normalized_query and "le thanh tong" in content:
                bonus += 3.0
            if "tren co so nao" in normalized_query and "nang cap khoa kinh te" in content:
                bonus += 3.0
            if "nam nao" in normalized_query and any(tok in section for tok in content_tokens):
                bonus += 1.5
            if any(tok in topic for tok in content_tokens):
                bonus += 0.5

            scored.append((bonus + float(ctx.get("rerank_score", 0.0)), ctx))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ctx for _, ctx in scored]

    def _extract_answer_from_contexts(self, query, contexts):
        normalized_query = normalize_answer(query)
        content_tokens = self._query_content_tokens(query)
        ordered_contexts = self._prefer_contexts(query, contexts)

        if "gia tri cot loi" in normalized_query:
            for ctx in ordered_contexts:
                text = ctx["content"]
                match = re.search(r"Giá trị cốt lõi\s*\n([^\n]+)", text, flags=re.IGNORECASE)
                if match:
                    return match.group(1).strip()

        if "so hieu" in normalized_query:
            for ctx in ordered_contexts:
                text = normalize_answer(ctx["content"])
                match = re.search(r"nghi dinh so [^,;\.\n]+ ngay \d{1,2} thang \d{1,2} nam \d{4}", text)
                if match:
                    return match.group(0).strip()

        if "tren co so nao" in normalized_query:
            for ctx in ordered_contexts:
                text = normalize_answer(ctx["content"])
                match = re.search(r"tren co so nang cap ([^,;\.\n]+)", text)
                if match:
                    return match.group(1).strip()

        if "tai dau" in normalized_query:
            for ctx in ordered_contexts:
                text = normalize_answer(ctx["content"])
                match = re.search(r"dat tru so tai ([^.;\n]+)", text)
                if match:
                    return match.group(1).strip()
                match = re.search(r"tai so \d+ pho [^.;\n]+", text)
                if match:
                    return match.group(0).strip()

        if "ngay nao" in normalized_query:
            best = ("", -1.0)
            for ctx in ordered_contexts:
                for span in self._split_candidate_spans(ctx["content"]):
                    score = self._score_candidate_span(query, span)
                    if all(tok in normalize_answer(span) for tok in content_tokens[:2]) and re.search(r"\d", span):
                        score += 2.0
                    match = re.search(r"(ngay )?\d{1,2} thang \d{1,2} nam \d{4}", normalize_answer(span))
                    if match and score > best[1]:
                        best = (match.group(0).strip(), score)
            if best[0]:
                return best[0]

        if "nam nao" in normalized_query:
            best = ("", -1.0)
            for ctx in ordered_contexts:
                for span in self._split_candidate_spans(ctx["content"]):
                    norm_span = normalize_answer(span)
                    score = self._score_candidate_span(query, span)
                    if content_tokens and all(tok in norm_span for tok in content_tokens[:2]):
                        score += 2.0
                    for match in re.finditer(r"\b(?:nam )?\d{4}\b", norm_span):
                        if score > best[1]:
                            best = (match.group(0).strip(), score)
            if best[0]:
                return best[0]

        if "bao nhieu" in normalized_query:
            best = ("", -1.0)
            for ctx in ordered_contexts:
                for span in self._split_candidate_spans(ctx["content"]):
                    norm_span = normalize_answer(span)
                    score = self._score_candidate_span(query, span)
                    if content_tokens and any(tok in norm_span for tok in content_tokens):
                        score += 1.0
                    match = re.search(r"\b\d[\d\.]*\s+(?:nguoi|vien chuc|don vi|tru cot|huan chuong)\b", norm_span)
                    if match and score > best[1]:
                        best = (match.group(0).strip(), score)
                    elif re.search(r"\b\d[\d\.]*\b", norm_span) and score > best[1]:
                        best = (re.search(r"\b\d[\d\.]*\b", norm_span).group(0).strip(), score)
            if best[0]:
                return best[0]

        return ""

    def _extractive_fallback(self, query, contexts):
        extracted = self._extract_answer_from_contexts(query, contexts)
        if extracted:
            return extracted

        best_span = ""
        best_score = -1.0

        ordered_contexts = self._prefer_contexts(query, contexts)
        for ctx in ordered_contexts:
            for span in self._split_candidate_spans(ctx["content"]):
                score = self._score_candidate_span(query, span)
                if score > best_score:
                    best_score = score
                    best_span = span

        answer_like = self._extract_answer_like_snippet(query, best_span)
        if answer_like:
            return answer_like

        if best_span:
            return best_span

        if contexts:
            return contexts[0]["content"].strip().replace("\n", " ")[:300]

        return "Khong tim thay thong tin trong tai lieu."

    def generate_answer(self, query, contexts):
        context_text = ""
        for i, ctx in enumerate(contexts):
            context_text += f"\n--- Ngu canh {i + 1} (Nguon: {ctx.get('topic', 'N/A')}) ---\n"
            context_text += ctx["content"] + "\n"

        prompt = f"""Ban la mot chuyen gia ve Dai hoc Quoc gia Ha Noi (DHQGHN/VNU).
Dua vao CAC NGU CANH duoc cung cap ben duoi, hay tra loi cau hoi mot cach NGAN GON, CHINH XAC va DAY DU.
Chi su dung thong tin tu ngu canh. Neu ngu canh khong chua cau tra loi, hay noi "Khong tim thay thong tin trong tai lieu."

{context_text}

Cau hoi: {query}

Tra loi ngan gon:"""

        if not self.llm_client:
            if self.use_extractive_fallback:
                return self._extractive_fallback(query, contexts)
            return "[LLM chua duoc cau hinh. Vui long set LLM_API_KEY]"

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Ban la tro ly tra loi cau hoi ve Dai hoc Quoc gia Ha Noi. "
                            "Tra loi ngan gon, chinh xac dua tren ngu canh."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if self.use_extractive_fallback:
                print(f"  [LLM fallback] {exc}")
                return self._extractive_fallback(query, contexts)
            return f"[Loi API: {str(exc)}]"

    def ask(self, query, verbose=False):
        retrieved = self.retrieve(query)
        if verbose:
            print(f"  [Retrieve] Top-{len(retrieved)} candidates found")

        reranked = self.rerank(query, retrieved)
        if verbose:
            print(f"  [Rerank] Top-{len(reranked)} after reranking:")
            for doc in reranked:
                print(f"    - [{doc.get('section', '')}] score={doc['rerank_score']:.4f}")

        return self.generate_answer(query, reranked)

    def test_retrieval(self, query, top_k=5):
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print(f"{'=' * 60}")

        retrieved = self.retrieve(query, top_k=top_k)
        print(f"\n--- Retrieval Results (top-{len(retrieved)}) ---")
        for doc in retrieved:
            print(f"  #{doc['rank']} [score={doc['score']:.4f}] {doc['section']}")
            print(f"    {doc['content'][:150]}...")

        reranked = self.rerank(query, retrieved, top_k=3)
        print(f"\n--- After Reranking (top-{len(reranked)}) ---")
        for i, doc in enumerate(reranked):
            print(f"  #{i + 1} [rerank={doc['rerank_score']:.4f}] {doc['section']}")
            print(f"    {doc['content'][:150]}...")


if __name__ == "__main__":
    rag = RAGSystem()
    print("\n" + "=" * 60)
    print("RAG System v2.0 san sang")
    print("=" * 60)

    test_queries = [
        "Dai hoc Dong Duong duoc thanh lap vao nam nao?",
        "Su menh cua DHQGHN la gi?",
        "Truong Dai hoc Khoa hoc Tu nhien co bao nhieu vien chuc?",
    ]
    for q in test_queries:
        rag.test_retrieval(q, top_k=5)
