import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


class LocalGenerator:
    def __init__(self, model_path: str):
        """
        Initializes the model by loading it into VRAM/RAM.
        model_path must point to the local directory (e.g., './models_cache/llama-3.2-1b').
        """
        # Hardware detection
        if torch.cuda.is_available():
            self.device = "cuda"
            print(f"NVIDIA GPU detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            print("Apple Silicon (M1/M2) GPU detected: using Metal Performance Shaders (MPS)")
        else:
            self.device = "cpu"
            print("No GPU detected, using CPU. Performance will be limited.")

        print(f"Loading model from {model_path} on device: {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=self.device,
            torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            local_files_only=True
        )

        print("✅ Generator initialized.")

    def generate_answer(self, query: str, context: str, max_new_tokens: int = 150) -> str:
        """
        Generates an answer using a standard prompt format for RAG.
        """
        # System prompt to force MS MARCO style generation

        # =====================================================================
        # PROMPT 0: Standard simple prompt for RAG generation
        # =====================================================================
        # system_prompt = "You are an expert assistant. Answer the question strictly based on the provided context. If the context does not contain the answer, say 'I don't know'."

        # =====================================================================
        # PROMPT 1: For WF_ROUGE-L & WF_BLEU (Well-Formed Answers)
        # =====================================================================
        system_prompt = (
            "You are an AI assistant tasked with answering questions based ONLY on the provided context.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer using a single, well-formed declarative sentence.\n"
            "2. Do not use bullet points, lists, or markdown formatting.\n"
            "3. Do not add extra advice, explanations, or conversational filler.\n"
            "4. Rephrase the context into a full sentence, do not just output the raw entity."
        )

        # =====================================================================
        # PROMPT 2: For Orig_ROUGE-L & Orig_BLEU (Short "Original" Answers)
        # =====================================================================
        # system_prompt = (
        #     "You are an AI assistant specialized in highly extractive question answering.\n"
        #     "CRITICAL INSTRUCTIONS:\n"
        #     "1. Output ONLY the exact entity, number, date, or shortest possible phrase that answers the question.\n"
        #     "2. DO NOT write a full sentence. DO NOT rephrase the context.\n"
        #     "3. DO NOT add any conversational filler, markdown formatting, or punctuation unless part of the entity.\n"
        #     "4. Base your extraction strictly on the provided context."
        # )

        user_prompt = f"CONTEXT:\n{context}\nQUESTION:\n{query}"

        # Standard chat_template for conversational models
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
            # Fallback if the model does not have a defined chat_template
            prompt = f"{system_prompt}\n\n{user_prompt}\n\nANSWER:\n"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generation phase
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.1,  # Low temperature to limit hallucinations in RAG
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode extracting only the newly generated part
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return answer.strip()

    def cleanup(self):
        """
        Aggressively frees memory.
        """
        print("Freeing VRAM/RAM memory...")
        del self.model
        del self.tokenizer
        gc.collect()

        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()
