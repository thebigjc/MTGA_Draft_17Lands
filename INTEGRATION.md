# Integration Guide: Using the Draft Model

This guide explains how to use the trained `model.onnx` and its `metadata.json` file to make predictions in a separate application.

## 1. Overview

The core of the model takes a sequence of token IDs representing the state of a draft and outputs logits for every card in the vocabulary. The process involves:

1.  **Loading** the ONNX model and the metadata.
2.  **Constructing** an input sequence of token IDs based on the current draft state (the cards you've picked, the cards in the current pack, etc.).
3.  **Running** inference with the ONNX runtime.
4.  **Interpreting** the output by masking the logits to only the valid cards in the current pack and finding the most likely pick.

## 2. Prerequisites

Your application will need the following Python libraries:

```bash
pip install onnxruntime numpy
```

## 3. Core Components

You will have two essential files generated from the training pipeline:

*   `model.onnx`: The trained model weights in the Open Neural Network Exchange format.
*   `metadata.json`: A JSON file containing crucial information needed to prepare model inputs and interpret its outputs.

The metadata contains:
*   `vocab`: A dictionary mapping card names and special tokens to their integer IDs.
*   `pick_positions`: A list mapping the logical pick number (0-41) to its actual index in the model's fixed-length sequence.
*   `max_seq_len`: The total length of the input sequence the model expects.

## 4. Step-by-Step Integration

Here is a detailed breakdown of the inference process, complete with a Python code example.

### Step 1: Initialize a Helper Class

First, create a class to manage the model, metadata, and all the associated logic.

```python
import onnxruntime as ort
import numpy as np
import json

class DraftModelHelper:
    def __init__(self, model_path="model.onnx", metadata_path="metadata.json"):
        """
        Initializes the helper by loading the ONNX model and metadata.
        """
        self.session = ort.InferenceSession(model_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
            
        self.vocab = self.metadata['vocab']
        self.ivocab = {v: k for k, v in self.vocab.items()} # Inverse vocab for decoding
        self.max_seq_len = self.metadata['max_seq_len']
        self.pick_positions = self.metadata['pick_positions']
        
        # Get special token IDs
        self.pad_token_id = self.vocab.get('[PAD]', 0)
        self.pack_cards_token_id = self.vocab.get('[PACK_CARDS]', -1)
        self.pack_cards_end_token_id = self.vocab.get('[PACK_CARDS_END]', -1)
        self.pick_cards_token_id = self.vocab.get('[PICK_CARDS]', -1)
        
        print("Draft Model Helper initialized successfully.")

```

### Step 2: Prepare the Model Input

This is the most critical step. The model doesn't understand card names; it understands a fixed-length sequence of token IDs. Your application must construct this sequence.

The sequence represents the history of picks followed by the current pack of cards to choose from.

```python
# In your DraftModelHelper class:
    def _prepare_input_sequence(self, picked_cards, pack_cards):
        """
        Constructs the token ID sequence for the model.
        
        Args:
            picked_cards (list[str]): A list of card names already picked.
            pack_cards (list[str]): A list of card names in the current pack.
            
        Returns:
            np.array: A numpy array of token IDs with shape (1, max_seq_len).
        """
        # 1. Map card names to their token IDs
        picked_ids = [self.vocab.get(c, self.pad_token_id) for c in picked_cards]
        pack_ids = [self.vocab.get(c, self.pad_token_id) for c in pack_cards]
        
        # 2. Construct the sequence with special tokens
        sequence = []
        sequence.append(self.pick_cards_token_id)
        sequence.extend(picked_ids)
        sequence.append(self.pack_cards_token_id)
        sequence.extend(pack_ids)
        sequence.append(self.pack_cards_end_token_id)
        
        # 3. Pad the sequence to the model's required length
        num_padding = self.max_seq_len - len(sequence)
        if num_padding < 0:
            raise ValueError("Input sequence is longer than model's max_seq_len.")
            
        padded_sequence = sequence + [self.pad_token_id] * num_padding
        
        # 4. Reshape for the model (batch size of 1)
        return np.array(padded_sequence, dtype=np.int64).reshape(1, -1)
```

### Step 3: Run Inference and Interpret the Output

Once you have the input sequence, you can run the model. The raw output is a large array of logits. You must mask these logits to only consider the cards that are actually in the current pack, and then convert them to probabilities using a softmax function.

```python
# In your DraftModelHelper class:
    def _softmax(self, x):
        """Computes softmax for a 1D numpy array, handling numerical stability."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict(self, picked_cards, pack_cards, current_pick_num):
        """
        Performs a full prediction for a given draft state.
        
        Args:
            picked_cards (list[str]): Cards already picked.
            pack_cards (list[str]): Cards in the current pack.
            current_pick_num (int): The current pick number (0 for P1P1, 1 for P1P2, etc.)
            
        Returns:
            list[tuple[str, float]]: A list of (card_name, probability) tuples, sorted by probability.
        """
        if current_pick_num >= len(self.pick_positions):
            raise ValueError("Invalid pick number.")

        # 1. Prepare the input tensor
        input_tensor = self._prepare_input_sequence(picked_cards, pack_cards)
        
        # 2. Run the ONNX model
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        result = self.session.run([output_name], {input_name: input_tensor})
        all_logits = result[0]
        
        # 3. Get the logits for the specific pick we are making
        model_pick_index = self.pick_positions[current_pick_num]
        pick_logits = all_logits[0, model_pick_index - 1, :]
        
        # 4. Create a mask for only the cards in the current pack
        valid_card_ids = {self.vocab.get(c) for c in pack_cards if c in self.vocab}
        full_vocab_size = len(self.vocab)
        mask = np.full(full_vocab_size, -np.inf, dtype=np.float32)
        
        for card_id in valid_card_ids:
            mask[card_id] = 0.0
        
        # 5. Apply the mask and calculate softmax probabilities
        masked_logits = pick_logits + mask
        probabilities = self._softmax(masked_logits)
        
        # 6. Collect and sort the results for valid cards
        results = []
        for card_id in valid_card_ids:
            prob = probabilities[card_id]
            card_name = self.ivocab.get(card_id)
            results.append((card_name, prob))
            
        return sorted(results, key=lambda x: x[1], reverse=True)
```

### Step 4: Putting It All Together

Here's how a developer would use this helper class in their application to display the pick ratings.

```python
# --- Example Usage ---
if __name__ == '__main__':
    # Assume model.onnx and metadata.json are in the same directory
    model_helper = DraftModelHelper()
    
    # --- P1P1 ---
    # It's the first pick, so our pool is empty.
    p1p1_picked = []
    p1p1_pack = [
        "Archon of Cruelty", "Damn", "Solitude", "Grief", "Ragavan, Nimble Pilferer",
        "Dragon's Rage Channeler", "Unholy Heat", "Murktide Regent", "Prismatic Ending",
        "Urza's Saga", "Endurance", "Scalding Tarn", "Misty Rainforest", "Marsh Flats"
    ]
    pick_number = 0 # P1P1 is the 0th pick
    
    predicted_probabilities = model_helper.predict(p1p1_picked, p1p1_pack, pick_number)
    
    print(f"--- Predictions for Pick {pick_number + 1} ---")
    for i, (card, prob) in enumerate(predicted_probabilities):
        print(f"{i+1}: {card:<30} (Confidence: {prob:.2%})")

    # For the next step, we'll assume the user takes the top recommendation
    best_pick = predicted_probabilities[0][0]
    
    # --- P1P2 ---
    # We picked the recommended card. Now it's the next pick.
    p1p2_picked = [best_pick]
    p1p2_pack = [
        "Thoughtseize", "Inquisition of Kozilek", "Lightning Bolt", "Counterspell",
        "Brainstone", "Abundant Harvest", "Arid Mesa", "Bloodstained Mire", "Flooded Strand",
        "Polluted Delta", "Verdant Catacombs", "Windswept Heath", "Wooded Foothills"
    ]
    pick_number = 1 # P1P2 is the 1st pick
    
    predicted_probabilities_2 = model_helper.predict(p1p2_picked, p1p2_pack, pick_number)
    
    print(f"\n--- Predictions for Pick {pick_number + 1} ---")
    for i, (card, prob) in enumerate(predicted_probabilities_2):
        print(f"{i+1}: {card:<30} (Confidence: {prob:.2%})")
```

## 5. Final Considerations

*   **State Management:** Your application is responsible for tracking the state of the draft (which cards have been picked, what the current pack is, and the current pick number).
*   **Unknown Cards:** The `vocab` in the metadata is fixed at training time. If a new set is released or an unknown card appears, your application should handle the resulting `KeyError` or use a default "unknown" token if one was included in the vocabulary.
*   **Performance:** For single picks, this process is very fast. If you need to evaluate many draft states at once, you can modify the helper to process a batch of input sequences simultaneously for higher throughput. 