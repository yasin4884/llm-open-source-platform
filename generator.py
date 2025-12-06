from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QwenCodeGenerator:
    """
    Simple wrapper around Qwen coder model to generate setup code
    for other Hugging Face models.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        self.model_name = model_name
        self.device = None
        self.model = None
        self.tokenizer = None

    def load(self):
        """Load Qwen model + tokenizer once on startup."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Loading Qwen model on {self.device}")

            common_kwargs = {
                "force_download": True,
                "resume_download": False,
                "local_files_only": False,
            }

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                **common_kwargs,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                **common_kwargs,
            ).to(self.device)

            self.model.eval()
            logger.info("Qwen model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading Qwen model: {e}")
            raise

    def generate_setup_code(self, target_model_name: str, model_type: str) -> str | None:
        """
        Generate initialization + chat code for a target HF model.

        Args:
            target_model_name: e.g. "meta-llama/Llama-3-8B-Instruct"
            model_type: "AutoModelForCausalLM" | "AutoModelForSeq2SeqLM" | "AutoModel"
        """
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("Qwen model is not loaded. Call load() first.")

        model_descriptions = {
            "AutoModelForCausalLM": "causal language model (GPT-like, decoder-only)",
            "AutoModelForSeq2SeqLM": "sequence-to-sequence model (T5/BART-like, encoder-decoder)",
            "AutoModel": "base model without task-specific head",
        }
        model_desc = model_descriptions.get(model_type, model_type)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Qwen, a helpful coding assistant. "
                    "Generate clean, working Python code."
                ),
            },
            {
                "role": "user",
                "content": f"""
Generate complete Python code to:
1. Load the Hugging Face model '{target_model_name}' using {model_type} ({model_desc})
2. Set up a simple chat interface with proper chat template (if needed)
3. Include all necessary imports and text-generation parameters
4. Add minimal error handling and device selection (CUDA if available)

Code must be directly runnable in a single file.
""",
            },
        ]

        try:
            # apply chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.8,
                    repetition_penalty=1.2,
                )

            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            return response

        except Exception as e:
            logger.error(f"Error generating setup code with Qwen: {e}")
            return None
