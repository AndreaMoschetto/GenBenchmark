import os
import json
import argparse
from tqdm import tqdm

from rag.retriever import FaissRetriever
from rag.generator import LocalGenerator
from constants import DATASET_PATH, RESULTS_DIR, MODELS_DIR


def run_benchmark(models_to_test, limit=None):
    print(f"Caricamento dataset da {DATASET_PATH}...")
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if limit:
        dataset = dataset[:limit]
        print(f"Esecuzione limitata a {limit} domande per test.")

    print("\nInizializzazione del Retriever (FAISS)...")
    retriever = FaissRetriever(num_docs=5)  # Recuperiamo i top 7 contesti

    for model_name in models_to_test:
        model_path = os.path.join(MODELS_DIR, model_name)

        # Saltiamo se il modello non è stato scaricato
        if not os.path.exists(model_path) or not os.listdir(model_path):
            print(f"\n⚠️ Modello {model_name} non trovato in {model_path}. Salto.")
            continue

        print("\n========================================")
        print(f"🚀 Avvio Benchmark per: {model_name}")
        print("========================================")

        # Inizializziamo il generatore caricandolo in RAM/VRAM
        generator = LocalGenerator(model_path=model_path)

        results = []

        # Create results folder if it doesn't exist
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # Iteriamo sulle domande con una barra di progresso
        for row in tqdm(dataset, desc=f"Generazione con {model_name}"):
            query = row["query"]
            ground_truth = row.get("well_formed_answer", "")

            # 1. Recupero del contesto
            context = retriever.get_context(query)

            # Splittiamo il contesto unico in una lista di testi per agevolare l'MRR
            retrieved_texts = context.split("\n\n---\n\n") if context else []

            # 2. Generazione della risposta
            generated_answer = generator.generate_answer(query, context)

            # 3. Salvataggio del record strutturato
            results.append({
                "query_id": row.get("query_id"),
                "query": query,
                "original_answers": row.get("answers", []),
                "well_formed_answer": ground_truth,
                "generated_answer": generated_answer,
                "retrieved_context": context,
                "retrieved_texts": retrieved_texts,          # Aggiunto per MRR
                "source_passages": row.get("passages", {}),  # Aggiunto per MRR (contiene is_selected)
                "query_type": row.get("query_type", "unknown")
            })

        # Liberiamo la memoria prima di passare al modello successivo
        generator.cleanup()

        # Salviamo i risultati in un file JSON dedicato a questo modello
        output_file = f"results_{model_name}.json"
        with open(os.path.join(RESULTS_DIR, output_file), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        print(f"✅ Benchmark completato per {model_name}. Risultati salvati in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegue il benchmark sui modelli scaricati.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Nomi delle cartelle dei modelli da testare (es. phi-3-mini llama-3.2-1b)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Numero massimo di domande da valutare (utile per test veloci)."
    )

    args = parser.parse_args()
    run_benchmark(args.models, args.limit)
