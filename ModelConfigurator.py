from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelConfigurator:
    MODEL_TYPES = {
        "causal": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM,
        "base": AutoModel,
    }

    def __init__(self, save_dir: str = "./models"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = None
        self.model_name = None

    def configure(self, model_name: str, model_type: str = "causal"):
      
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Invalid model_type: {model_type}. Use: {list(self.MODEL_TYPES.keys())}")

        self.model_name = model_name
        self.model_type = model_type

        try:
            logger.info(f"Loading tokenizer for {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            ModelClass = self.MODEL_TYPES[model_type]

            logger.info(f"Loading {model_type} model: {model_name} on {self.device}")
            self.model = ModelClass.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self.device)

            self.model.eval()

            local_path = os.path.join(self.save_dir, model_name.replace("/", "_"))
            os.makedirs(local_path, exist_ok=True)
            self.model.save_pretrained(local_path)
            self.tokenizer.save_pretrained(local_path)

            logger.info(f"Model saved to: {local_path}")
            return {
                "model": self.model,
                "tokenizer": self.tokenizer,
                "path": local_path,
                "type": model_type,
                "device": str(self.device),
            }
        except Exception as e:
            logger.error(f"Error configuring model: {e}")
            raise

    def chat(self, prompt: str, max_new_tokens: int = 256):
       
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not configured. Call configure() first.")

        try:
            if self.model_type == "causal":
                return self._chat_causal(prompt, max_new_tokens)
            elif self.model_type == "seq2seq":
                return self._chat_seq2seq(prompt, max_new_tokens)
            elif self.model_type == "base":
                return self._chat_base(prompt)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise

    def _chat_causal(self, prompt: str, max_new_tokens: int):
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

    def _chat_seq2seq(self, prompt: str, max_new_tokens: int):
        """برای مدل‌های Seq2Seq مثل T5, BART, Flan-T5"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_beams=4  # seq2seq معمولاً با beam search بهتره
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

    def _chat_base(self, prompt: str):
        """برای AutoModel - فقط embeddings برمی‌گردونه"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # برای base model معمولاً embeddings می‌خوایم
        embeddings = outputs.last_hidden_state.mean(dim=1)  # average pooling
        
        return {
            "embeddings": embeddings.cpu().numpy(),
            "shape": embeddings.shape,
            "message": "Base model returns embeddings, not text generation"
        }
