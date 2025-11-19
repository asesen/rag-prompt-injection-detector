import faiss
import pandas as pd
import numpy as np
import torch


class VectorStore:
    def __init__(
        self,
        dimension: int = 1024,
        train_path: str = "data/train_with_embeddings.parquet",
        val_path: str = "data/val_with_embeddings.parquet",
        test_path: str = "data/test_with_embeddings.parquet",
        context_path: str = "data/questions_with_embeddings.parquet"
    ):

        self.dimension = dimension

        # === Load main datasets ===
        self.train_df = pd.read_parquet(train_path)
        self.val_df = pd.read_parquet(val_path)
        self.test_df = pd.read_parquet(test_path)

        try:
            self.context_df = pd.read_parquet(context_path)
        except Exception:
            self.context_df = None

        # === Prepare embeddings ===

        # ------ Train embeddings ------
        train_emb_list = list(self.train_df['embedding'].values)
        train_emb_arr = np.vstack([
            np.array(e, dtype=np.float32) for e in train_emb_list
        ]).astype('float32')

        self.train_embeddings = train_emb_arr
        self.n_train = len(self.train_embeddings)

        # ------ Context embeddings ------
        if self.context_df is not None:
            ctx_emb_list = list(self.context_df['embedding'].values)
            ctx_emb_arr = np.vstack([
                np.array(e, dtype=np.float32) for e in ctx_emb_list
            ]).astype('float32')

            self.context_embeddings = ctx_emb_arr
            self.n_context = len(ctx_emb_arr)
        else:
            self.context_embeddings = None
            self.n_context = 0

        # === Build FAISS index ===
        self.index = faiss.IndexFlatL2(self.dimension)

        # Add train first
        self.index.add(self.train_embeddings)

        # Add context embeddings (if exist)
        if self.context_embeddings is not None:
            self.index.add(self.context_embeddings)
            print("Added context embeddings to FAISS index")

        self.use_faiss = True

    # ======================================================================
    # Basic search
    # ======================================================================

    def search(self, query_emb: np.ndarray, k: int = 5):
        """
        Returns FAISS distances and indices.
        """
        q = np.array(query_emb, dtype=np.float32).reshape(1, -1)
        dist, idx = self.index.search(q, k)
        return dist[0], idx[0]


    # ======================================================================
    # Build prompt context from context_df
    # ======================================================================

    def make_prompt_context(self,context_dist, context_idx):
        """
        Создает RAG-контекст на основе ближайших вопросов/ответов
        из context_df.
        Формат:
            Q: ...
            A: ...
        """
        blocks = []

        for d, faiss_i in zip(context_dist, context_idx):
          print(d, faiss_i, self.n_train)
          if faiss_i >= self.n_train:
              # context
              local_i = faiss_i - self.n_train
              label = 0
              source = "context"

              row = self.context_df.iloc[local_i]

              q = row.get("Question", "")
              a = row.get("Answer", "")

              blocks.append(f"Q: {q}\nA: {a}")


        if blocks == []:
            return None

        return "\n\n".join(blocks)

    def get_label(self, idx):
        if idx < self.n_train:
            return self.train_df['jailbreak'][idx]
        else:
            return 0

    # ======================================================================
    # Original methods required by detector (keep unchanged)
    # ======================================================================

    def give_train_data(self):
        return self.train_df

    def give_val_data(self):
        return self.val_df

    def give_test_data(self):
        return self.test_df