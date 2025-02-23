import numpy as np
import onnxruntime as ort
import os
from typing import List, Tuple, Dict
import json

class DraftModelWrapper:
    def __init__(self, model_dir: str):
        # Load ONNX model
        self.session = ort.InferenceSession(
            os.path.join(model_dir, "model.onnx"),
            providers=['CPUExecutionProvider']
        )
        
        # Load vocab and other metadata from JSON
        with open(os.path.join(model_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
            self.vocab = metadata["vocab"]
            self.pick_positions = metadata["pick_positions"]
            self.pack_map = {int(k): v for k, v in metadata["pack_map"].items()}
            self.expansion_code = metadata["expansion_code"]
            self.max_seq_len = metadata["max_seq_len"]
            
        # Create reverse vocab lookup
        self.index_to_vocab = {i: token for token, i in self.vocab.items()}
    
    def encode(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(token, 0) for token in tokens]
        
    def get_top_k_predictions(self, draft_tokens: List[str], k: int = 3) -> List[Tuple[str, float]]:
        # Convert tokens to numpy array
        token_ids = np.array([self.encode(draft_tokens)], dtype=np.int64)
        
        # Get current pick position
        current_pos = len(draft_tokens)
        
        # Pad sequence if needed
        if current_pos < self.max_seq_len:
            padding = np.zeros((1, self.max_seq_len - current_pos), dtype=np.int64)
            token_ids = np.concatenate([token_ids, padding], axis=1)
            
        # Get predictions from ONNX model
        logits = self.session.run(
            None,  # Get all outputs
            {"input": token_ids}
        )[0]  # First output is logits
            
        # Apply pack mask
        pack_start, pack_end = self.pack_map[current_pos]
        pack_cards = token_ids[:, pack_start:pack_end]
        vocab_indices = np.arange(len(self.vocab))
        valid_mask = (pack_cards[:, :, None] == vocab_indices).any(axis=1)
        logits = np.where(valid_mask, logits, -np.inf)
        
        # Get top k predictions using numpy
        probs = self._softmax(logits[0, current_pos-1])
        indices = np.argpartition(-probs, k)[:k]
        indices = indices[np.argsort(-probs[indices])]
        
        return {self.index_to_vocab[idx].replace("_", " "): float(probs[idx])
                for idx in indices}
    
    @staticmethod
    def _softmax(x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum() 