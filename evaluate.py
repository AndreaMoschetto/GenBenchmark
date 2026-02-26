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
from tqdm import tqdm

# Ignoriamo i warning di NLTK se le frasi sono troppo corte per i 4-grammi
warnings.filterwarnings("ignore")


class RAGEvaluator:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        print("Inizializzazione degli strumenti di valutazione...")
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        print(f"Caricamento modello embedding per la similarità: {embedding_model}")
        self.similarity_model = SentenceTransformer(embedding_model)
        self.smoother = SmoothingFunction().method1

    def calculate_text_metrics(self, generated: str, references: list) -> dict:
        """
        Calcola ROUGE, BLEU e Cosine Similarity.
        Restituisce il punteggio massimo ottenuto tra tutte le referenze passate.
        """
        if not generated or not references:
            return {"rougeL": 0.0, "bleu": 0.0, "similarity": 0.0}

        best_rouge = 0.0
        best_bleu = 0.0
        best_sim = 0.0

        gen_emb = self.similarity_model.encode([generated])

        for ref in references:
            if not ref:
                continue

            # ROUGE-L
            f_measure = self.scorer.score(ref, generated)['rougeL'].fmeasure
            if f_measure > best_rouge:
                best_rouge = f_measure

            # BLEU
            ref_tokens = ref.split()
            gen_tokens = generated.split()
            bleu_score = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self.smoother)
            if bleu_score > best_bleu:
                best_bleu = bleu_score

            # Similarità Semantica
            ref_emb = self.similarity_model.encode([ref])
            sim_score = float(cosine_similarity(gen_emb, ref_emb)[0][0])
            if sim_score > best_sim:
                best_sim = sim_score

        return {
            "rougeL": best_rouge,
            "bleu": best_bleu,
            "similarity": best_sim
        }

    def calculate_mrr(self, retrieved_texts: list, source_passages: dict) -> float:
        """
        Calcola il Mean Reciprocal Rank (MRR).
        Verifica a quale indice della lista recuperata si trova il primo testo marcato come is_selected=1.
        """
        if not source_passages or not retrieved_texts:
            return 0.0

        # Estraiamo i testi dei passaggi considerati rilevanti dalla Ground Truth
        relevant_texts = []
        try:
            is_selected_list = source_passages.get("is_selected", [])
            passage_texts = source_passages.get("passage_text", [])
            for text, sel in zip(passage_texts, is_selected_list):
                if sel == 1:
                    relevant_texts.append(text)
        except Exception:
            pass

        if not relevant_texts:
            return 0.0

        # Controllo della posizione
        for index, ret_text in enumerate(retrieved_texts):
            # Usiamo un controllo in/contains per essere flessibili rispetto a piccoli spazi o artefatti di formattazione
            if any(rel_text.strip() in ret_text.strip() or ret_text.strip() in rel_text.strip() for rel_text in relevant_texts):
                return 1.0 / (index + 1)

        return 0.0


def evaluate_models(results_dir="results", output_csv="benchmark_summary.csv"):
    evaluator = RAGEvaluator()
    summary_data = []

    if not os.path.exists(results_dir):
        print(f"❌ Errore: La cartella '{results_dir}' non esiste. Assicurati di aver generato i file.")
        return

    json_files = [f for f in os.listdir(results_dir) if f.startswith("results_") and f.endswith(".json")]

    if not json_files:
        print(f"⚠️ Nessun file di risultati trovato nella cartella '{results_dir}'. Esegui prima benchmark.py.")
        return

    print("\n🔍 Calcolo del MRR globale (Metrica del Retriever)...")
    global_mrr = 0.0
    with open(os.path.join(results_dir, json_files[0]), "r", encoding="utf-8") as f:
        first_model_data = json.load(f)

    mrr_scores = []
    for row in first_model_data:
        retrieved = row.get("retrieved_texts", [])
        mrr_score = evaluator.calculate_mrr(retrieved, row.get("source_passages", {}))
        mrr_scores.append(mrr_score)

    global_mrr = np.mean(mrr_scores)
    print(f"🎯 MRR Globale (costante per tutti i modelli): {global_mrr:.4f}")

    for file_name in json_files:
        model_name = file_name.replace("results_", "").replace(".json", "")
        print(f"\n📊 Valutazione in corso per: {model_name}...")

        with open(os.path.join(results_dir, file_name), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Strutture dati separate per Well Formed, Original Answers e MRR
        metrics = {
            "wf_rouge": [], "wf_bleu": [], "wf_sim": [],
            "orig_rouge": [], "orig_bleu": [], "orig_sim": []
        }

        for row in tqdm(data, desc=f"Processing {file_name}"):
            generated = row.get("generated_answer", "")

            # 1. Valutazione rispetto a Well Formed Answer
            wf_refs = [row.get("well_formed_answer")] if row.get("well_formed_answer") else []
            wf_scores = evaluator.calculate_text_metrics(generated, wf_refs)
            metrics["wf_rouge"].append(wf_scores["rougeL"])
            metrics["wf_bleu"].append(wf_scores["bleu"])
            metrics["wf_sim"].append(wf_scores["similarity"])

            # 2. Valutazione rispetto a Original Answers
            orig_refs = row.get("original_answers", [])
            orig_scores = evaluator.calculate_text_metrics(generated, orig_refs)
            metrics["orig_rouge"].append(orig_scores["rougeL"])
            metrics["orig_bleu"].append(orig_scores["bleu"])
            metrics["orig_sim"].append(orig_scores["similarity"])

        # Aggregazione finale delle medie
        summary_data.append({
            "Model": model_name,
            "Retrieval_MRR": global_mrr,  # MRR è costante per tutti i modelli in questo setup
            "WF_ROUGE-L": np.mean(metrics["wf_rouge"]),
            "WF_BLEU": np.mean(metrics["wf_bleu"]),
            "WF_Similarity": np.mean(metrics["wf_sim"]),
            "Orig_ROUGE-L": np.mean(metrics["orig_rouge"]),
            "Orig_BLEU": np.mean(metrics["orig_bleu"]),
            "Orig_Similarity": np.mean(metrics["orig_sim"])
        })

        print(f"✅ {model_name} valutato.")

    # Creiamo un DataFrame e salviamo
    df = pd.DataFrame(summary_data)
    df.to_csv(output_csv, index=False)

    print(f"\n🎉 Valutazione completata! Risultati riassuntivi salvati in {output_csv}")
    print("\n--- RISULTATI FINALI ---")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcola le metriche di valutazione sui risultati del benchmark.")
    parser.add_argument("--dir", type=str, default="results", help="Cartella contenente i file results_*.json")
    parser.add_argument("--output", type=str, default="benchmark_summary.csv", help="Nome del file CSV di output")
    args = parser.parse_args()

    evaluate_models(args.dir, args.output)
