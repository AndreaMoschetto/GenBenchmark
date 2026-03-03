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
from constants import RESULTS_DIR

warnings.filterwarnings("ignore")


class RAGEvaluator:
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        print("Initializing evaluation tools...")
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        print(f"Loading embedding model for similarity: {embedding_model}")
        self.similarity_model = SentenceTransformer(embedding_model)
        self.smoother = SmoothingFunction().method1

    def calculate_text_metrics(self, generated: str, references: list) -> dict:
        """
        Calculates ROUGE, BLEU, and Cosine Similarity.
        Returns the maximum score obtained among all provided references.
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

            # Semantic Similarity
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
        Calculates the Mean Reciprocal Rank (MRR).
        Checks the index of the retrieved list where the first text marked as is_selected=1 is found.
        """
        if not source_passages or not retrieved_texts:
            return 0.0

        # Extract texts of passages considered relevant from the Ground Truth
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

        # Position check
        for index, ret_text in enumerate(retrieved_texts):
            # Use an in/contains check to be flexible regarding small spaces or formatting artifacts
            if any(rel_text.strip() in ret_text.strip() or ret_text.strip() in rel_text.strip() for rel_text in relevant_texts):
                return 1.0 / (index + 1)

        return 0.0


def evaluate_models(results_dir=RESULTS_DIR, output_csv="benchmark_summary.csv"):
    evaluator = RAGEvaluator()
    summary_data = []

    if not os.path.exists(results_dir):
        print(f"Error: The directory '{results_dir}' does not exist. Make sure you generated the files.")
        return

    json_files = [f for f in os.listdir(results_dir) if f.startswith("results_") and f.endswith(".json")]

    if not json_files:
        print(f"No result files found in directory '{results_dir}'. Run benchmark.py first.")
        return

    print("\nCalculating global MRR (Retriever Metric)...")
    global_mrr = 0.0
    with open(os.path.join(results_dir, json_files[0]), "r", encoding="utf-8") as f:
        first_model_data = json.load(f)

    mrr_scores = []
    for row in first_model_data:
        retrieved = row.get("retrieved_texts", [])
        mrr_score = evaluator.calculate_mrr(retrieved, row.get("source_passages", {}))
        mrr_scores.append(mrr_score)

    global_mrr = np.mean(mrr_scores)
    print(f"Global MRR (constant across all models): {global_mrr:.4f}")

    for file_name in json_files:
        model_name = file_name.replace("results_", "").replace(".json", "")
        print(f"\nEvaluating: {model_name}...")

        file_path = os.path.join(results_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Separate data structures for Well Formed, Original Answers, and MRR
        metrics = {
            "wf_rouge": [], "wf_bleu": [], "wf_sim": [],
            "orig_rouge": [], "orig_bleu": [], "orig_sim": []
        }

        for row in tqdm(data, desc=f"Processing {file_name}"):
            generated = row.get("generated_answer", "")

            # 1. Evaluate against Well Formed Answer
            wf_refs = [row.get("well_formed_answer")] if row.get("well_formed_answer") else []
            wf_scores = evaluator.calculate_text_metrics(generated, wf_refs)
            metrics["wf_rouge"].append(wf_scores["rougeL"])
            metrics["wf_bleu"].append(wf_scores["bleu"])
            metrics["wf_sim"].append(wf_scores["similarity"])

            # 2. Evaluate against Original Answers
            orig_refs = row.get("original_answers", [])
            orig_scores = evaluator.calculate_text_metrics(generated, orig_refs)
            metrics["orig_rouge"].append(orig_scores["rougeL"])
            metrics["orig_bleu"].append(orig_scores["bleu"])
            metrics["orig_sim"].append(orig_scores["similarity"])

            # 3. Calculate MRR for this specific row
            retrieved = row.get("retrieved_texts", [])
            row_mrr = evaluator.calculate_mrr(retrieved, row.get("source_passages", {}))

            # --- NEW: Inject the calculated metrics into the row dictionary ---
            row["metrics"] = {
                "mrr": row_mrr,
                "wf_rougeL": wf_scores["rougeL"],
                "wf_bleu": wf_scores["bleu"],
                "wf_similarity": wf_scores["similarity"],
                "orig_rougeL": orig_scores["rougeL"],
                "orig_bleu": orig_scores["bleu"],
                "orig_similarity": orig_scores["similarity"]
            }

        # --- NEW: Write the updated data back to the JSON file ---
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        # Final aggregation of means
        summary_data.append({
            "Model": model_name,
            "Retrieval_MRR": global_mrr,  # MRR is constant for all models in this setup
            "WF_ROUGE-L": np.mean(metrics["wf_rouge"]),
            "WF_BLEU": np.mean(metrics["wf_bleu"]),
            "WF_Similarity": np.mean(metrics["wf_sim"]),
            "Orig_ROUGE-L": np.mean(metrics["orig_rouge"]),
            "Orig_BLEU": np.mean(metrics["orig_bleu"]),
            "Orig_Similarity": np.mean(metrics["orig_sim"])
        })

        print(f"✅ {model_name} evaluated and JSON updated.")

    # Create the csv path if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Create a DataFrame and save
    df = pd.DataFrame(summary_data)
    df.to_csv(output_csv, index=False)

    print(f"\nEvaluation complete! Summary results saved to {output_csv}")
    print("\n--- FINAL RESULTS ---")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculates evaluation metrics on benchmark results.")
    parser.add_argument("--dir", type=str, default=RESULTS_DIR, help="Directory containing results_*.json files")
    parser.add_argument("--output", type=str, default="benchmark_summary.csv", help="Output CSV file name")
    args = parser.parse_args()

    evaluate_models(args.dir, args.output)
