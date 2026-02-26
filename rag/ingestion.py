import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from constants import INDEX_PATH, META_PATH, DATASET_PATH


def ingest_to_faiss(json_path=DATASET_PATH,
                    index_out=INDEX_PATH,
                    meta_out=META_PATH,
                    model_name="all-MiniLM-L6-v2"):

    print(f"Caricamento dati da {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Estraiamo tutti i passaggi unici per evitare duplicati nell'indice
    # Nota: la struttura di HuggingFace per MS MARCO ha 'passages'
    # come dict contenente 'passage_text' e 'is_selected'
    unique_passages = set()
    for row in dataset:
        passages_dict = row.get("passages", {})
        texts = passages_dict.get("passage_text", [])
        for text in texts:
            unique_passages.add(text)

    passages_list = list(unique_passages)
    print(f"Trovati {len(passages_list)} passaggi unici. Inizio embedding...")

    # download del modello
    model = SentenceTransformer(model_name)

    # Generazione degli embeddings
    embeddings = model.encode(passages_list, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Creiamo l'indicizzazione su FAISS (FlatL2 è il più semplice e fa ricerca esatta)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"Salvataggio dell'indice FAISS in {index_out}...")
    faiss.write_index(index, index_out)

    print(f"Salvataggio dei metadati (testi) in {meta_out}...")
    with open(meta_out, "wb") as f:
        pickle.dump(passages_list, f)

    print("Ingestion FAISS completata!")


if __name__ == "__main__":
    ingest_to_faiss()
