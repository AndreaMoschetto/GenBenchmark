from datasets import load_dataset
import json
from tqdm import tqdm


def extract_well_formed_subset(output_file="msmarco_v2.1_wellformed.json"):
    print("Connessione a Hugging Face per MS MARCO v2.1 (Streaming mode)...")

    # Carichiamo il dataset in streaming
    dataset = load_dataset("microsoft/ms_marco", "v2.1", streaming=True)

    subset = []
    # Usiamo il validation set per il benchmark
    iterator = dataset['validation']

    print("Filtro dei record con 'well_formed_answers' in corso...")

    for record in tqdm(iterator, desc="Processing"):
        # Verifichiamo la presenza della risposta ben formata
        wf_answers = record.get('wellFormedAnswers', [])

        if wf_answers and len(wf_answers) > 0:
            # Puliamo il record per tenere solo ciò che serve al RAG
            clean_record = {
                "query_id": record.get("query_id"),
                "query": record.get("query"),
                "passages": record.get("passages"),  # Contiene testi e is_selected
                "answers": record.get("answers"),
                "well_formed_answer": wf_answers[0],  # Prendiamo la prima
                "query_type": record.get("query_type")  # Utile per analisi per categoria
            }
            subset.append(clean_record)

    # Salvataggio locale
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(subset, f, indent=4, ensure_ascii=False)

    print(f"\nEstrazione completata!")
    print(f"Record totali con risposta ben formata: {len(subset)}")
    print(f"File salvato in: {output_file}")


if __name__ == "__main__":
    extract_well_formed_subset()
