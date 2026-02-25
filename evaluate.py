import os
import json
import argparse
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

# Ignoriamo i warning di NLTK se le frasi sono troppo corte per i 4-grammi
warnings.filterwarnings("ignore")


class RAGEvaluator:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        print("Inizializzazione degli strumenti di valutazione...")
        # Inizializziamo ROUGE (usiamo ROUGE-L come nel paper di MS MARCO)
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # Inizializziamo il modello per la Semantic Similarity
        print(f"Caricamento modello embedding per la similarità: {embedding_model}")
        self.similarity_model = SentenceTransformer(embedding_model)

        # Funzione di smoothing per BLEU (evita punteggi a 0 se mancano n-grammi superiori)
        self.smoother = SmoothingFunction().method1

    def calculate_metrics(self, generated: str, references: list) -> dict:
        """
        Calcola ROUGE, BLEU e Cosine Similarity.
        Accetta una lista di reference (es. well_formed_answer + original_answers)
        e restituisce il punteggio massimo ottenuto tra tutte le reference.
        """
        if not generated or not references:
            return {"rougeL_fmeasure": 0.0, "bleu": 0.0, "semantic_similarity": 0.0}

        best_rouge = 0.0
        best_bleu = 0.0
        best_sim = 0.0

        # Embedding della risposta generata
        gen_emb = self.similarity_model.encode([generated])

        for ref in references:
            if not ref:
                continue

            # 1. ROUGE-L
            rouge_scores = self.scorer.score(ref, generated)
            f_measure = rouge_scores['rougeL'].fmeasure
            if f_measure > best_rouge:
                best_rouge = f_measure

            # 2. BLEU
            ref_tokens = ref.split()
            gen_tokens = generated.split()
            bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoother)
            if bleu_score > best_bleu:
                best_bleu = bleu_score

            # 3. Semantic Similarity (Cosine)
            ref_emb = self.similarity_model.encode([ref])
            sim_score = cosine_similarity(gen_emb, ref_emb)[0][0]
            if sim_score > best_sim:
                best_sim = float(sim_score)

        return {
            "rougeL_fmeasure": best_rouge,
            "bleu": best_bleu,
            "semantic_similarity": best_sim
        }


def evaluate_models(results_dir="results", output_csv="benchmark_summary.csv"):
    evaluator = RAGEvaluator()
    summary_data = []

    # Controllo che la cartella esista
    if not os.path.exists(results_dir):
        print(f"❌ Errore: La cartella '{results_dir}' non esiste. Assicurati di aver generato i file.")
        return

    # Cerchiamo tutti i file generati dal benchmark nella cartella specificata
    json_files = [f for f in os.listdir(results_dir) if f.startswith("results_") and f.endswith(".json")]

    if not json_files:
        print(f"⚠️ Nessun file di risultati trovato nella cartella '{results_dir}'. Esegui prima benchmark.py.")
        return

    for file_name in json_files:
        model_name = file_name.replace("results_", "").replace(".json", "")
        print(f"\n📊 Valutazione in corso per: {model_name}...")

        with open(os.path.join(results_dir, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)

        model_metrics = {"rougeL": [], "bleu": [], "similarity": []}

        for row in data:
            generated = row.get("generated_answer", "")

            # Costruiamo la lista delle reference GT
            references = []
            if row.get("well_formed_answer"):
                references.append(row["well_formed_answer"])
            if row.get("original_answers"):
                references.extend(row["original_answers"])

            # Se il modello ha risposto "I don't know" ma una risposta c'era, le metriche saranno basse
            scores = evaluator.calculate_metrics(generated, references)

            model_metrics["rougeL"].append(scores["rougeL_fmeasure"])
            model_metrics["bleu"].append(scores["bleu"])
            model_metrics["similarity"].append(scores["semantic_similarity"])

        # Aggreghiamo i risultati medi per il modello
        summary_data.append({
            "Model": model_name,
            "ROUGE-L (Mean)": np.mean(model_metrics["rougeL"]),
            "BLEU (Mean)": np.mean(model_metrics["bleu"]),
            "Semantic Similarity (Mean)": np.mean(model_metrics["similarity"])
        })

        print(f"✅ {model_name} valutato.")

    # Creiamo un DataFrame e salviamo i risultati
    df = pd.DataFrame(summary_data)
    df.to_csv(output_csv, index=False)

    print(f"\n🎉 Valutazione completata! Risultati riassuntivi salvati in {output_csv}")
    print("\n--- RISULTATI FINALI ---")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcola le metriche di valutazione sui risultati del benchmark.")
    # Default aggiornato a "results"
    parser.add_argument("--dir", type=str, default="results", help="Cartella contenente i file results_*.json")
    parser.add_argument("--output", type=str, default="benchmark_summary.csv", help="Nome del file CSV di output")
    args = parser.parse_args()

    evaluate_models(args.dir, args.output)
