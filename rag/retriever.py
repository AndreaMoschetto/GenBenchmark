import faiss
import pickle
from sentence_transformers import SentenceTransformer
from constants import INDEX_PATH, META_PATH


class FaissRetriever:
    def __init__(self,
                 index_path=INDEX_PATH,
                 meta_path=META_PATH,
                 model_name="all-MiniLM-L6-v2",
                 num_docs: int = 5):

        self.k = num_docs
        print(f"Caricamento modello embedding ({model_name})...")
        self.model = SentenceTransformer(model_name)

        print(f"Caricamento indice FAISS da {index_path}...")
        self.index = faiss.read_index(index_path)

        print(f"Caricamento testi dei passaggi da {meta_path}...")
        with open(meta_path, "rb") as f:
            self.passages_text = pickle.load(f)

        print("✅ Retriever inizializzato.")

    def get_context(self, query: str) -> str:
        # 1. Convertiamo la query in vettore
        query_vector = self.model.encode([query]).astype("float32")

        # 2. Cerchiamo i k vettori più vicini in FAISS
        # distances conterrà le distanze L2, indices gli ID numerici
        distances, indices = self.index.search(query_vector, self.k)

        # 3. Recuperiamo i testi usando gli ID
        retrieved_docs = [self.passages_text[idx] for idx in indices[0] if idx != -1]

        # 4. Formattiamo come stringa unica per il prompt
        formatted_context = "\n\n---\n\n".join(retrieved_docs)
        return formatted_context


# --- Piccolo blocco di test se esegui il file direttamente ---
if __name__ == "__main__":
    retriever = FaissRetriever(num_docs=3)
    test_query = "how to become a notary oklahoma"
    context = retriever.get_context(test_query)
    print(f"\nQUERY: {test_query}\n")
    print("CONTESTO RECUPERATO:\n")
    print(context)
