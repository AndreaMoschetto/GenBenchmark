import os
import argparse
from huggingface_hub import snapshot_download

# Dizionario dei modelli: Nome "facile" -> ID del repository Hugging Face
MODELS_ZOO = {
    "llama-3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
    "gemma-2-2b": "google/gemma-2-2b-it",
    "ministral-3b": "mistralai/Ministral-3-3B-Instruct-2512"
}

TARGET_DIR = "./models_cache"


def download_models(models_to_download, hf_token):
    os.makedirs(TARGET_DIR, exist_ok=True)

    for nome_modello in models_to_download:
        if nome_modello not in MODELS_ZOO:
            print(f"⚠️ Modello '{nome_modello}' non trovato nel dizionario. Salto.")
            continue

        repo_id = MODELS_ZOO[nome_modello]
        save_path = os.path.join(TARGET_DIR, nome_modello)

        # Controllo se esiste già ed è popolato (evita di riscaricare)
        if os.path.exists(save_path) and len(os.listdir(save_path)) > 0:
            print(f"⏭️ Il modello {nome_modello} è già presente in {save_path}. Download saltato.")
            continue

        print(f"\n⬇️ Inizio download di {nome_modello} ({repo_id})...")
        try:
            # Scarichiamo il modello ignorando i formati pesanti non necessari a PyTorch
            snapshot_download(
                repo_id=repo_id,
                local_dir=save_path,
                local_dir_use_symlinks=False,
                token=hf_token,
                ignore_patterns=["*.msgpack", "*.h5", "coreml/*", "original/*"]
            )
            print(f"✅ Download completato: {save_path}")
        except Exception as e:
            print(f"❌ Errore scaricando {nome_modello}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scarica i modelli LLM per il benchmark RAG.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help=f"Specifica i modelli da scaricare separati da spazio. Opzioni: all, {', '.join(MODELS_ZOO.keys())}"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Il tuo token Hugging Face (necessario per Llama, Gemma, Ministral)."
    )

    args = parser.parse_args()

    # Se l'utente digita "all", scarichiamo tutti i modelli del dizionario
    models_list = list(MODELS_ZOO.keys()) if "all" in args.models else args.models
    download_models(models_list, args.token)
