from secret import API_KEY
from .vectorstore import VectorStore
from .detector import Detector

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings


class RAG:
    def __init__(
        self,
        model_name: str = "ministral-8b-latest",
        temperature: float = 0.0,
        k: int = 8,
        threshold = 0.5
    ):
        self.temperature = temperature
        self.model_name = model_name
        self.k = k
        self.threshold = threshold

        # LLM
        self.llm = ChatMistralAI(
            model=self.model_name,
            temperature=self.temperature,
            api_key=API_KEY
        )

        # Embeddings
        self.embedder = MistralAIEmbeddings(
            model="mistral-embed",
            api_key=API_KEY
        )

        # Vector store + detector
        self.vector_store = VectorStore()
        self.detector = Detector(self.vector_store)

        # ====== загрузка или тренировка ======
        try:
            self.detector.load_primary("primary_cpu.pt")
            print("Detector loaded.")
        except Exception:
            print("Detector not found. Training...")
            self.detector.train_primary(
                epochs=20,
                batch_size=128,
                lr=1e-3,
                weight_decay=1e-4,
                save_path="primary_cpu.pt"
            )
        # =====================================

    # ---------------------------------------------------------------------
    def make_prompt(self, query: str, context_prompt: str | None):
        system_prompt = (
            "You are a helpful assistant who provides short and precise advice "
            "about Python programming."
        )

        if context_prompt:
            final_prompt = (
                f"You can use the following similar questions as context:\n"
                f"{context_prompt}\n\n"
                f"Now answer the user question:\n"
                f"{query}"
            )
        else:
            final_prompt = (
                f"User question:\n{query}"
            )

        # LangChain формат сообщений:
        return [
            ("system", system_prompt),
            ("user", final_prompt)
        ]

    # ---------------------------------------------------------------------
    def get_response(self, query: str):
        # ===== embed query =====
        print("wait emb")
        try:
            query_emb = self.embedder.embed_query(query)
        except Exception as e:
            return f"Embedding error: {e}"
        print("We get emb")

        # ===== vector store =====
        try:
            context_dist, context_idx = self.vector_store.search(query_emb, self.k)
        except Exception as e:
            return f"Vector store error: {e}"
        print("We get context")
        # ===== detector =====
        try:
            is_jailbreak = self.detector.detect(query_emb, context_dist, context_idx)
        except Exception as e:
            return f"Detector error: {e}"
        print("det answer ", is_jailbreak)

        if is_jailbreak > self.threshold:
            return "Sorry, this prompt is not allowed."

        # ===== RAG context =====
        context_prompt = self.vector_store.make_prompt_context(context_dist, context_idx)

        print(context_prompt)

        # ===== Build prompt =====
        messages = self.make_prompt(query, context_prompt)

        print(messages)

        # ===== LLM =====
        try:
            answer = self.llm.invoke(messages)
        except Exception as e:
            return f"LLM error: {e}"

        return answer.content if hasattr(answer, "content") else answer