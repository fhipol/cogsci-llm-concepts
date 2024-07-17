import gc
import torch
import csv
import numpy as np


class ModelManager:

    def __init__(self, model_path, max_tokens=60, temperature=0):
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.temperature = temperature

    def load_model(self, model_path):
        from mistral.model import Transformer
        from mistral.tokenizer import Tokenizer

        self.model = Transformer.from_folder(
            model_path,
            max_batch_size=1,
            num_pipeline_ranks=1,
            dtype=torch.float16,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    def clean_model(self):
        """
        Used to clean the model from memory
        """
        if self.model:
            del self.model
            self.model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_tokenizer(self):
        from mistral.tokenizer import Tokenizer
        self.tokenizer = Tokenizer(str(model_path / "tokenizer.model"))

    def prompt(self, prompts):
        """
      Check the generate function from the git repo,
      is an interesting implementation in how to infer from the model
      """

        from main import generate
        max_tokens = self.max_tokens
        temperature = self.temperature

        res, logprobs = generate(
            [prompts],
            self.model,
            self.tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return res, logprobs
