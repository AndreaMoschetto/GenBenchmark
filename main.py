import yaml
import argparse
import sys
import os
from dotenv import load_dotenv
from utils.download_dataset import extract_well_formed_subset
from utils.download_models import download_models
from benchmark import run_benchmark
from evaluate import evaluate_models
from constants import RESULTS_DIR, OUTPUT_CSV

load_dotenv()


def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Errore nel caricamento del file di configurazione: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Orchestratore della Pipeline RAG")
    parser.add_argument("--config", type=str, default="config.yaml", help="Percorso del file YAML")
    args = parser.parse_args()

    print(f"Caricamento configurazione da: {args.config}")
    config = load_config(args.config)
    pipeline_steps = config.get("pipeline", {})

    if pipeline_steps.get("run_download_dataset", False):
        print("\n" + "=" * 40)
        print("[FASE 0]: DOWNLOAD DATASET")
        print("=" * 40)
        extract_well_formed_subset()
    else:
        print("\nSalto fase di download del dataset.")
    # 1. DOWNLOAD MODELS
    if pipeline_steps.get("run_download_models", False):
        print("\n" + "=" * 40)
        print("[FASE 1]: DOWNLOAD MODELLI")
        print("=" * 40)
        dl_cfg = config.get("download", {})
        hf_token = os.getenv("HF_TOKEN")
        download_models(dl_cfg.get("models", []), hf_token)
    else:
        print("\nSalto fase di download.")

    # 2. BENCHMARK
    if pipeline_steps.get("run_benchmark", False):
        print("\n" + "=" * 40)
        print("[FASE 2]: ESECUZIONE BENCHMARK")
        print("=" * 40)
        bench_cfg = config.get("benchmark", {})
        run_benchmark(bench_cfg.get("models", []), bench_cfg.get("limit"))
    else:
        print("\nSalto fase di benchmark.")

    # 3. EVALUATION
    if pipeline_steps.get("run_evaluation", False):
        print("\n" + "=" * 40)
        print("[FASE 3]: VALUTAZIONE METRICHE")
        print("=" * 40)
        evaluate_models(RESULTS_DIR, OUTPUT_CSV)
    else:
        print("\nSalto fase di valutazione.")

    print("\nEsecuzione della pipeline completata!")


if __name__ == "__main__":
    main()
