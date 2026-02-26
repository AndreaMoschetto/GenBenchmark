import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


class LocalGenerator:
    def __init__(self, model_path: str):
        """
        Inizializza il modello caricandolo fisicamente nella VRAM/RAM.
        model_path deve puntare alla cartella locale (es. './local_models/llama-3.2-1b')
        """
        # Rilevamento automatico dell'hardware
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"GPU NVIDIA rilevata: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"  # Per il tuo Mac M1
            print("GPU Apple Silicon (M1/M2) rilevata: usando Metal Performance Shaders (MPS)")
        else:
            self.device = "cpu"
            print("Nessuna GPU rilevata, usando CPU. Le prestazioni saranno limitate.")

        print(f"Caricamento modello da {model_path} su device: {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,  # float16 su GPU per risparmiare memoria, float32 su CPU per compatibilità
            local_files_only=True
        )

        print("✅ Generatore inizializzato.")

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 150) -> str:
        """
        Genera una risposta usando un formato prompt standard per il RAG.
        """

        system_prompt = "You are an expert assistant. Answer the question strictly based on the provided context. If the context does not contain the answer, say 'I don't know'."
        user_prompt = f"CONTEXT:\n{context}\nQUESTION:\n{query}"
        # chat_template standard per modelli conversazionali
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback se il modello non ha un chat_template definito
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nANSWER:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # ----- GENERAZIONE -----
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Bassa per limitare le allucinazioni nel RAG
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decodifica estraendo solo la parte nuova generata
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return answer.strip()

    def cleanup(self):
        """
        Libera la memoria in modo aggressivo
        """
        print("Liberazione memoria VRAM/RAM...")
        del self.model
        del self.tokenizer
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
