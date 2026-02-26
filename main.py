import yaml
import argparse
import sys

from download_models import download_models
from benchmark import run_benchmark
from evaluate import evaluate_models


def load_config(config_path):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"❌ Errore nel caricamento del file di configurazione: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Orchestratore della Pipeline RAG")
    parser.add_argument("--config", type=str, default="config.yaml", help="Percorso del file YAML")
    args = parser.parse_args()

    print(f"📄 Caricamento configurazione da: {args.config}")
    config = load_config(args.config)
    pipeline_steps = config.get("pipeline", {})

    # 1. DOWNLOAD
    if pipeline_steps.get("run_download", False):
        print("\n" + "=" * 40)
        print("📥 FASE 1: DOWNLOAD MODELLI")
        print("=" * 40)
        dl_cfg = config.get("download", {})
        download_models(dl_cfg.get("models", []), dl_cfg.get("token"))
    else:
        print("\n⏭️ Salto fase di download.")

    # 2. BENCHMARK
    if pipeline_steps.get("run_benchmark", False):
        print("\n" + "=" * 40)
        print("🚀 FASE 2: ESECUZIONE BENCHMARK")
        print("=" * 40)
        bench_cfg = config.get("benchmark", {})
        run_benchmark(bench_cfg.get("models", []), bench_cfg.get("limit"))
    else:
        print("\n⏭️ Salto fase di benchmark.")

    # 3. EVALUATION
    if pipeline_steps.get("run_evaluation", False):
        print("\n" + "=" * 40)
        print("📊 FASE 3: VALUTAZIONE METRICHE")
        print("=" * 40)
        eval_cfg = config.get("evaluation", {})
        evaluate_models(eval_cfg.get("results_dir", "results"), eval_cfg.get("output_csv", "benchmark_summary.csv"))
    else:
        print("\n⏭️ Salto fase di valutazione.")

    print("\n🎉 Esecuzione della pipeline completata!")


if __name__ == "__main__":
    main()
