import numpy as np
import json
import onnxruntime as ort
from src.dataset import Dataset
from typing import List, Dict, Tuple, Callable
import os.path
from collections import defaultdict
from src.utils import Result

class History:
    def __init__(self):
        self.pack = []
        self.pick = -1

class BuildModelWrapper:
    def __init__(self, model_path, vocab_path=None):
        """
        Initialize the ONNX deck predictor
        
        Args:
            model_path: Path to the ONNX model
            vocab_path: Path to vocabulary file (optional, can be derived from model path)
        """
        model_path = os.path.join(model_path, "model.onnx")
        # If no vocab path provided, try to derive from model path
        if vocab_path is None:
            # Look for *_vocab.json next to the ONNX model
            vocab_path = model_path.replace('.onnx', '_vocab.json')
            if not os.path.exists(vocab_path):
                # Look for PT vocabulary format as fallback
                pt_vocab_path = model_path.replace('.onnx', '_vocab.pt')
                if os.path.exists(pt_vocab_path):
                    vocab_path = pt_vocab_path
                else:
                    # Try to find vocab file in the same directory
                    model_dir = os.path.dirname(model_path)
                    json_vocab_files = [f for f in os.listdir(model_dir) if f.endswith('_vocab.json')]
                    pt_vocab_files = [f for f in os.listdir(model_dir) if f.endswith('_vocab.pt')]
                    
                    if json_vocab_files:
                        vocab_path = os.path.join(model_dir, json_vocab_files[0])
                    elif pt_vocab_files:
                        vocab_path = os.path.join(model_dir, pt_vocab_files[0])
                    else:
                        raise ValueError("Could not find vocabulary file. Please specify with --vocab_path")
        
        if not os.path.exists(vocab_path):
            raise ValueError(f"Vocabulary file not found: {vocab_path}")
        
        print(f"Loading vocabulary from {vocab_path}")
        # Load the vocabulary
        with open(vocab_path, 'r') as f:
            vocab_data = json.load(f)
            if "card_vocab" in vocab_data:
                self.card_vocab = vocab_data["card_vocab"]
            else:
                self.card_vocab = vocab_data
        
        # Create reverse mapping (index to card name)
        self.index_to_card = {idx: card for card, idx in self.card_vocab.items()}
        
        # Check for metadata file
        metadata_path = model_path.replace('.onnx', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                if "basic_land_indices" in self.metadata:
                    self.basic_land_indices = self.metadata["basic_land_indices"]
                    self.max_copies = self.metadata["max_copies"]
                    self.seq_len = self.metadata["seq_len"]
                    print(f"Loaded basic land indices from metadata: {self.basic_land_indices}")
                    print(f"Loaded max_copies from metadata: {self.max_copies}")
        else:
            self.metadata = None
                
        # Identify basic lands in the vocabulary if not loaded from metadata
        if not hasattr(self, 'basic_land_indices'):
            self.basic_land_names = ["Plains", "Island", "Swamp", "Mountain", "Forest"]
            self.basic_land_indices = []
            
            # Search for basic lands in the vocabulary
            print("Looking for basic lands in vocabulary...")
            for land_name in self.basic_land_names:
                land_idx = -1  # Default to not found
                
                # Try direct name-to-index mapping
                if land_name in self.card_vocab:
                    land_idx = self.card_vocab[land_name]
                    print(f"Found {land_name} at index {land_idx}")
                else:
                    # Try partial matches
                    partial_matches = [card for card in self.card_vocab.keys() if land_name in card]
                    if partial_matches:
                        chosen_match = partial_matches[0]
                        land_idx = self.card_vocab[chosen_match]
                        print(f"Found partial match for {land_name}: {chosen_match} at index {land_idx}")
                
                # Add the index (or -1 if not found)
                self.basic_land_indices.append(land_idx)
                if land_idx == -1:
                    print(f"Warning: Could not find {land_name} in vocabulary")
        
        # Load the ONNX model
        print(f"Loading ONNX model from {model_path}")
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Set max_copies if not set from metadata
        if not hasattr(self, 'max_copies'):
            self.max_copies = 18  # Default value
        
        # Create a deck builder for post-processing
        self.deck_builder = DeckBuilder(max_copies=self.max_copies, basic_land_indices=self.basic_land_indices)
        
        print("ONNX model loaded successfully!")
    
    def _get_card_id_from_name(self, card_name):
        """Get card ID from name"""
        if card_name in self.card_vocab:
            return self.card_vocab[card_name]
        if card_name.replace(" ", "_") in self.card_vocab:
            return self.card_vocab[card_name.replace(" ", "_")]
        return None
            
    def predict(self, taken_cards : List[Dict]):
        """
        Generate deck prediction for the given draft pool
        
        Args:
            draft_pool_tensor: NumPy array of card indices in the draft pool
            original_data: Original JSON data for reference (optional)
            
        Returns:
            Dictionary with prediction results
        """
        card_pool = []
        for card in taken_cards:
            card_pool.append(self._get_card_id_from_name(card["name"]))
        # Pad to max sequence length - we don't get passed basic lands
        while len(card_pool) < self.seq_len:
            card_pool.append(0)
        card_pool_array = np.array([card_pool], dtype=np.int64)        
        # Run inference with ONNX Runtime
        ort_inputs = {self.input_name: card_pool_array}
        ort_outputs = self.session.run(self.output_names, ort_inputs)
        
        # Get copy scores from the output
        copy_scores = ort_outputs[0][0]  # Remove batch dimension
        
        # Convert draft pool to counts
        draft_pool_i = card_pool_array[0]  # Remove batch dimension
        unique_indices, unique_counts = np.unique(draft_pool_i[draft_pool_i > 0], return_counts=True)
        counts = np.zeros(len(self.card_vocab), dtype=np.int64)
        for idx, count in zip(unique_indices, unique_counts):
            counts[idx] = count
        
        # Build the deck using the deck builder
        predicted_deck_counts = self.deck_builder.build_deck(
            copy_scores,
            counts
        )
        
        # Separate basic lands from non-basic cards
        nonbasics = {}
        basics = {name: [] for name in ["Plains", "Island", "Swamp", "Mountain", "Forest"]}
        
        for card_id, scores in predicted_deck_counts.items():
            if card_id in self.basic_land_indices:
                # Get the index in basic_land_indices
                idx = self.basic_land_indices.index(card_id)
                if 0 <= idx < len(basics):
                    land_name = list(basics.keys())[idx]
                    basics[land_name] = scores
            else:
                nonbasics[card_id] = scores
        
    
        # Format the results
        result = {
            'draft_pool': self._format_card_list(draft_pool_i),
            'predicted_deck': {
                'nonbasics': self._format_card_list(nonbasics),
                'basics': basics,
            }
        }
        
        return result
    
    def _format_card_list(self, cards):
        """
        Format a list of card indices as a dictionary of card names to counts
        
        Args:
            cards: Either a numpy array of indices or a dict of {index: count}
            
        Returns:
            Dict of {card_name: count}
        """
        result = {}
        
        if isinstance(cards, dict):
            # Dict of {index: count}
            for idx, count in cards.items():
                if idx in self.index_to_card:
                    name = self.index_to_card[idx]
                    result[name] = count
                else:
                    result[f"Unknown({idx})"] = count
        else:
            # Numpy array of indices
            for idx in cards:
                if idx > 0:  # Skip padding
                    if idx in self.index_to_card:
                        name = self.index_to_card[idx]
                        result[name] = result.get(name, 0) + 1
                    else:
                        result[f"Unknown({idx})"] = result.get(f"Unknown({idx})", 0) + 1
        
        return result
            
    def print_debug_info(self):
        """Print debug information about model structure"""
        print("\n==== ONNX Model Debug Information ====")
        print(f"ONNX session: {self.session}")
        
        print("\nModel Inputs:")
        for i, input_info in enumerate(self.session.get_inputs()):
            print(f"Input #{i}")
            print(f"  Name: {input_info.name}")
            print(f"  Shape: {input_info.shape}")
            print(f"  Type: {input_info.type}")
        
        print("\nModel Outputs:")
        for i, output_info in enumerate(self.session.get_outputs()):
            print(f"Output #{i}")
            print(f"  Name: {output_info.name}")
            print(f"  Shape: {output_info.shape}")
            print(f"  Type: {output_info.type}")
            
        if hasattr(self, 'metadata') and self.metadata:
            print("\nModel Metadata:")
            for key, value in self.metadata.items():
                print(f"  {key}: {value}")
                
        print("\nDeck Builder Info:")
        print(f"  max_copies: {self.deck_builder.max_copies}")
        print(f"  basic_land_indices: {self.deck_builder.basic_land_indices}")
        
        print("\nVocabulary Info:")
        print(f"- Vocabulary size: {len(self.card_vocab)}")
        print(f"- First 5 entries: {list(self.card_vocab.items())[:5]}")
        
        print("\nBasic Land Info:")
        print(f"- Basic land indices: {self.basic_land_indices}")
        for i, idx in enumerate(self.basic_land_indices):
            if i < len(["Plains", "Island", "Swamp", "Mountain", "Forest"]):
                land_name = ["Plains", "Island", "Swamp", "Mountain", "Forest"][i]
                print(f"  {land_name}: {idx} ({self.index_to_card.get(idx, 'Unknown')})")
            
        print("\n===================================")

    def print_prediction(self, prediction):
        """
        Print the prediction in a human-readable format
        
        Args:
            prediction: The prediction dictionary returned by predict method
        """
        print("\n===== Draft to Deck Prediction =====")
        
        # Print input draft pool summary
        print("\n=== Draft Pool ===")
        pool_cards = prediction.get('draft_pool', {})
        # The pool_cards is a dict with card names as keys and counts as values
        print(f"Total cards in pool: {sum(count for count in pool_cards.values())}")
        print(f"Unique cards in pool: {len(pool_cards)}")
        
        # Print predicted deck
        print("\n=== Predicted Deck ===")
        
        # Print non-basic cards
        print("\n--- Spells and Non-basic Lands ---")
        nonbasics = prediction.get('predicted_deck', {}).get('nonbasics', {})
        
        # The nonbasics is a dict with card names as keys and counts as values
        sorted_nonbasics = sorted(
            [(name, scores) for name, scores in nonbasics.items()],
            key=lambda x: (-sum(x[1]), x[0])  # Sort by count (desc) then name (asc)
        )
        
        for name, scores in sorted_nonbasics:
            print(f"  {len(scores)}x {name} {",".join(map(str, sorted(scores, reverse=True)))}")
            
        # Calculate non-basic total
        nonbasic_total = sum(len(scores) for _, scores in sorted_nonbasics)
        
        # Print basic lands
        print("\n--- Basic Lands ---")
        basics = prediction.get('predicted_deck', {}).get('basics', {})
        for land_name, scores in basics.items():
            if len(scores) > 0:
                print(f"  {len(scores)}x {land_name} {",".join(map(str, sorted(scores, reverse=True)))}")
        
        # Calculate basic land total
        basic_total = sum(len(scores) for _, scores in basics.items())
        
        # Print deck summary
        print("\n--- Deck Summary ---")
        print(f"  Non-basic cards: {nonbasic_total}")
        print(f"  Basic lands: {basic_total}")
        print(f"  Total deck size: {nonbasic_total + basic_total}")
                
        # Print land ratio if available
        if 'land_ratio' in prediction.get('predicted_deck', {}):
            land_ratio = prediction['predicted_deck']['land_ratio'] * 100
            print(f"\nLand ratio: {land_ratio:.1f}%")
        
        print("=" * 40)

class DeckBuilder:
    def __init__(self, max_copies=20, basic_land_indices=None):
        self.max_copies = max_copies
        self.basic_land_indices = basic_land_indices

    def build_deck(self, scores_np, drafted_pool_counts):
        """
        Constructs the final 40-card deck by selecting the highest-scoring card copies.

        Arguments:
            scores_np: numpy array of shape [max_copies, total_card_vocab_size] for one example,
                         representing scores for each copy of each card.
            drafted_pool_counts: A list/tensor of available counts for each card type in the drafted pool.

        Returns:
            deck: dict mapping card id to count
        """
        # Calculate desired number of cards based on land ratio
        total_cards = 40  # Standard deck size

        # Create a list of all card copies with their scores
        # Format: (card_id, copy_index, score)
        all_copies = []

        for card_id, available_count in enumerate(drafted_pool_counts):
            # Check if this is a basic land
            is_basic_land = hasattr(self, 'basic_land_indices') and self.basic_land_indices and card_id in self.basic_land_indices

            # For basic lands, always consider up to max_copies regardless of available count
            if is_basic_land:
                # Use max_copies for basic lands (typically unlimited in MTG)
                effective_count = self.max_copies
            else:
                # For non-basic lands, use the available count as before
                effective_count = available_count

            if effective_count > 0:  # Only consider cards that are available
                # Only consider copies up to the effective count
                for copy_idx in range(min(effective_count, self.max_copies)):
                    # Get score for this specific copy of this specific card
                    score = scores_np[copy_idx, card_id]
                    all_copies.append((card_id, copy_idx, score))

        # Sort all copies by their scores (highest first)
        all_copies.sort(key=lambda x: x[2], reverse=True)

        # Select the top N copies with highest scores
        selected_copies = all_copies[:total_cards]

        # Count how many copies of each card were selected
        counts = defaultdict(list)
        for card_id, _, score in selected_copies:
            counts[card_id].append(score)

        # Verify deck size is exactly 40 cards
        total_deck_size = sum((len(scores) for scores in counts.values()))
        assert total_deck_size == 40, f"Deck size is {total_deck_size}, expected 40"

        return counts

class DraftModelWrapper:
    def __init__(self, model_dir: str):
        """
        Initializes the helper by loading the ONNX model and metadata.
        """
        model_path = os.path.join(model_dir, "model.onnx")
        metadata_path = os.path.join(model_dir, "metadata.json")

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

    def _prepare_input_sequence(self, picked_cards: List[str], pack_cards: List[str]) -> np.ndarray:
        """
        Constructs the token ID sequence for the model.
        
        Args:
            picked_cards (list[str]): A list of card names already picked (with underscores).
            pack_cards (list[str]): A list of card names in the current pack (with underscores).
            
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
            raise ValueError(f"Input sequence is longer than model's max_seq_len. Sequence length: {len(sequence)}, max: {self.max_seq_len}")
            
        padded_sequence = sequence + [self.pad_token_id] * num_padding
        
        # 4. Reshape for the model (batch size of 1)
        return np.array(padded_sequence, dtype=np.int64).reshape(1, -1)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Computes softmax for a 1D numpy array, handling numerical stability."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def predict(self, picked_cards: List[str], pack_cards: List[str], current_pick_num: int) -> Dict[str, float]:
        """
        Performs a full prediction for a given draft state.
        
        Args:
            picked_cards (list[str]): Cards already picked.
            pack_cards (list[str]): Cards in the current pack.
            current_pick_num (int): The current pick number (0 for P1P1, 1 for P1P2, etc.)
            
        Returns:
            dict[str, float]: A dict of {card_name: probability}.
        """
        if current_pick_num >= len(self.pick_positions):
            print(f"Warning: Invalid pick number {current_pick_num}. No recommendations will be provided.")
            return {}

        # 1. Prepare the input tensor
        picked_card_names_for_input = [c.replace(" ", "_") for c in picked_cards]
        pack_card_names_for_input = [c.replace(" ", "_") for c in pack_cards]

        input_tensor = self._prepare_input_sequence(picked_card_names_for_input, pack_card_names_for_input)
        
        # 2. Run the ONNX model
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        
        result = self.session.run([output_name], {input_name: input_tensor})
        all_logits = result[0]
        
        # 3. Get the logits for the specific pick we are making
        model_pick_index = self.pick_positions[current_pick_num]
        pick_logits = all_logits[0, model_pick_index - 1, :]
        
        # 4. Create a mask for only the cards in the current pack
        valid_card_ids = {self.vocab.get(c) for c in pack_card_names_for_input if c in self.vocab}
        full_vocab_size = len(self.vocab)
        mask = np.full(full_vocab_size, -np.inf, dtype=np.float32)
        
        for card_id in valid_card_ids:
            if card_id is not None:
                mask[card_id] = 0.0
        
        # 5. Apply the mask and calculate softmax probabilities
        masked_logits = pick_logits + mask
        probabilities = self._softmax(masked_logits)
        
        # 6. Collect and sort the results for valid cards
        results = {}
        for card_id in valid_card_ids:
            if card_id is not None:
                prob = probabilities[card_id]
                card_name = self.ivocab.get(card_id).replace("_", " ")
                results[card_name] = float(prob)
            
        return results

class MLRecommender:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.draft_model = DraftModelWrapper(os.path.join(model_path, "draft"))
        self.build_model = BuildModelWrapper(os.path.join(model_path, "build"))

        self.history = {}
    def reset(self):
        self.tokens = []
            
    def add_pack_history(self, pack_cards: List[int], pack: int, pick: int):
        p_p = (pack, pick)
        if p_p not in self.history:
            h = History()
            h.pack = pack_cards
            self.history[p_p] = h
        else:
            h = self.history[p_p]
            if h.pack != pack_cards:
                print(f"Updating pack history for {p_p}")
                h.pack = pack_cards
                
    def add_pick_history(self, card: int, pack: int, pick: int):
        p_p = (pack, pick)
        if p_p not in self.history:
            h = History()
            h.pick = card
            self.history[p_p] = h
        else:
            h = self.history[p_p]
            if h.pick != card:
                print(f"Updating card history for {p_p}")
                h.pick = card

    def pick_tokens(self, set_data: Dataset):
        tokens = self._token_prefix()

        for p_p in sorted(self.history.keys()):
            h = self.history[p_p]
            if not h.pack:
                print(f"No pack history for {p_p}")
                continue

            tokens.append("[PACK_CARDS]")
            pack_cards = [name.replace(" ", "_") for name in set_data.get_names_by_id(h.pack)]
            tokens.extend(pack_cards)
            tokens.append("[PACK_CARDS_END]")

            tokens.append("[PICK]")
            
            if h.pick != -1:
                pick_name = [name.replace(" ", "_") for name in set_data.get_names_by_id([h.pick])]
                tokens.extend(pick_name)
                tokens.append("[PICK_END]")

        return tokens

    def _token_prefix(self):
        tokens = [
            "[DRAFT_START]",
            "[EXPANSION]", 
            self.draft_model.expansion_code, # TODO: Get expansion code from current draft
            "[EXPANSION_END]",
            "[EVENT_TYPE]", "PremierDraft", "[EVENT_TYPE_END]",
            "[RANK]", "mythic", "7", "[RANK_END]"
        ]
        
        return tokens
    
    def _get_current_pack_cards(self, card_list: List[Dict]) -> List[str]:
        return [card["name"].replace(" ", "_") for card in card_list]
    
    def compare_tokens(self, card_list: List[Dict]) -> List[str]:
        tokens = self._token_prefix()
        tokens.append("[PACK_CARDS]")
        tokens.extend(card_list)
        # Append basic lands until we reach the first pick
        # -2 because we have the [PACK_CARDS_END] and [PICK] tokens
        while len(tokens) < self.draft_model.pick_positions[0] -2:
            tokens.append("Plains")

        tokens.append("[PACK_CARDS_END]")
        tokens.append("[PICK]")

        return tokens
    
    def recommend_deck(self, card_list: List[Dict], basic_lands: Dict[str, Dict], make_land_card: Callable[[str, Dict[str, Dict], int], Dict]) -> List[Dict]:
        # Create a lookup that handles both spaced and underscore variants of card names
        card_map = {}
        for card in card_list:
            name_spaced = card["name"]
            name_underscore = name_spaced.replace(" ", "_")
            card_map[name_spaced] = card
            card_map[name_underscore] = card
        deck = self.build_model.predict(card_list)
        recommended_deck = []
        lands = []
        for card, scores in deck["predicted_deck"]["nonbasics"].items():
            recommended_deck.extend([card_map[card]] * len(scores))
        for card, scores in deck["predicted_deck"]["basics"].items():
            if scores:
                basic_land = make_land_card(basic_lands, card, len(scores))
                lands.append(basic_land)
        return recommended_deck, lands
    
    def get_recommendations(self, card_list: List[Dict], set_data: Dataset) -> List[Dict]:
        # If the current pack is empty, return an empty list
        if not card_list:
            return []

        # Construct picked_cards list from history
        picked_cards = []
        if self.history:
            sorted_picks = sorted(self.history.keys())
            for p_p in sorted_picks:
                h = self.history[p_p]
                if h.pick != -1:
                    picked_card_name = set_data.get_names_by_id([h.pick])[0]
                    picked_cards.append(picked_card_name)

        current_pick_num = len(picked_cards)
        
        # The card_list is a list of dicts, where each dict is a card.
        # We need to extract the name from each card.
        pack_cards = [card["name"] for card in card_list]

        # Get model predictions
        predictions = self.draft_model.predict(
            picked_cards=picked_cards,
            pack_cards=pack_cards,
            current_pick_num=current_pick_num
        )
                
        return predictions 

if __name__ == '__main__':
    import json
    # --- Example Usage ---
    print("Running Draft Model Test...")

    # Initialize the recommender
    recommender = MLRecommender(model_path="model")
    print("Recommender initialized.")
    
    set_data = Dataset()
    # Using a test dataset file
    result = set_data.open_file("tests/data/FIN_PremierDraft_Data_Test.json")
    if result != Result.VALID:
        print(f"Failed to load dataset. Result: {result}")
        exit()

    # Load test draft data from FIN_PremierDraft_Test.json
    try:
        with open("tests/data/FIN_PremierDraft_Test.json", "r") as f:
            draft_data = json.load(f)
        picks_data = draft_data.get("picks", [])
        if not picks_data:
            print("No picks found in the test data file.")
            exit()
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading test data file: {e}")
        exit()

    # --- P1P1 ---
    print("\n--- Testing Pick 1, Pack 1 ---")
    p1p1_data = picks_data[0]
    p1p1_pack_cards = p1p1_data.get("available", [])
    p1p1_actual_pick = p1p1_data.get("pick", {}).get("name")
    
    print(f"Actual user pick: {p1p1_actual_pick}")

    # Get recommendations for P1P1
    recommendations_p1p1 = recommender.get_recommendations(p1p1_pack_cards, set_data)
    
    print("Recommendations for P1P1:")
    if recommendations_p1p1:
        sorted_recs = sorted(recommendations_p1p1.items(), key=lambda item: item[1], reverse=True)
        for i, (card, prob) in enumerate(sorted_recs):
            print(f"{i+1}: {card:<40} (Confidence: {prob:.2%})")
        
        # Simulate picking the card the user actually picked
        print(f"\nSimulating pick: {p1p1_actual_pick}")
        
        # To add to history, we need the card ID. We'll look it up from the test data.
        best_pick_id = set_data.get_ids_by_name([p1p1_actual_pick])[0]
        recommender.add_pick_history(card=int(best_pick_id), pack=1, pick=1)
    else:
        print("No recommendations were returned for P1P1.")
        exit()

    # --- P1P2 ---
    print("\n--- Testing Pick 1, Pack 2 ---")
    if len(picks_data) < 2:
        print("Not enough pick data in test file for P1P2.")
        exit()

    p1p2_data = picks_data[1]
    p1p2_pack_cards = p1p2_data.get("available", [])
    p1p2_actual_pick = p1p2_data.get("pick", {}).get("name")

    print(f"Actual user pick: {p1p2_actual_pick}")
    
    # Get recommendations for P1P2
    recommendations_p1p2 = recommender.get_recommendations(p1p2_pack_cards, set_data)

    print("Recommendations for P1P2:")
    if recommendations_p1p2:
        sorted_recs_p2 = sorted(recommendations_p1p2.items(), key=lambda item: item[1], reverse=True)
        for i, (card, prob) in enumerate(sorted_recs_p2):
            print(f"{i+1}: {card:<40} (Confidence: {prob:.2%})")
    else:
        print("No recommendations were returned for P1P2.")

    print("\nDraft Model Test Finished.")

    # --- Test Last Pick (P3P14) ---
    print("\n--- Testing Last Pick of Draft (Pick 42) ---")
    
    # Re-initialize recommender for a clean test
    recommender = MLRecommender(model_path="model")
    print("Recommender re-initialized for last-pick test.")

    if len(picks_data) < 42:
        print("Not enough pick data in test file for a full draft test (requires 42 picks). Skipping last-pick test.")
    else:
        # Simulate the first 41 picks to set up the test for the last pick
        print("Simulating first 41 picks...")
        for i in range(41):
            pick_data = picks_data[i]
            actual_pick_name = pick_data.get("pick", {}).get("name")
            if not actual_pick_name:
                print(f"Warning: Missing pick name for pick {i + 1}. Skipping this history entry.")
                continue
            
            pick_id_list = set_data.get_ids_by_name([actual_pick_name])
            if not pick_id_list:
                print(f"Warning: Could not find ID for card '{actual_pick_name}' at pick {i + 1}. Skipping.")
                continue
                
            pick_id = pick_id_list[0]
            pack_num = (i // 14) + 1
            pick_num_in_pack = (i % 14) + 1
            recommender.add_pick_history(card=int(pick_id), pack=pack_num, pick=pick_num_in_pack)
        
        print("Simulated 41 picks successfully.")

        # --- Test the 42nd pick (index 41) ---
        print("\n--- Evaluating recommendations for the final pick (P3P14) ---")
        p3p14_data = picks_data[41]
        p3p14_pack_cards = p3p14_data.get("available", [])
        p3p14_actual_pick = p3p14_data.get("pick", {}).get("name")
        
        print(f"Cards in pack for last pick: {[card['name'] for card in p3p14_pack_cards]}")
        print(f"User's actual pick: {p3p14_actual_pick}")

        # Get recommendations for the final pick
        last_pick_recommendations = recommender.get_recommendations(p3p14_pack_cards, set_data)

        print("\nRecommendations for Final Pick:")
        if last_pick_recommendations:
            sorted_recs = sorted(last_pick_recommendations.items(), key=lambda item: item[1], reverse=True)
            for i, (card, prob) in enumerate(sorted_recs):
                print(f"{i+1}: {card:<40} (Confidence: {prob:.2%})")
            
            # Add the final pick to history
            last_pick_id = set_data.get_ids_by_name([p3p14_actual_pick])[0]
            recommender.add_pick_history(card=int(last_pick_id), pack=3, pick=14)

            # --- Test behavior after the draft is complete ---
            print("\n--- Testing for recommendations after draft completion ---")
            # At this point, 42 cards have been picked, and current_pick_num will be 42.
            # This should trigger the "Invalid pick number" warning and return no recommendations.
            after_draft_recommendations = recommender.get_recommendations(p3p14_pack_cards, set_data)
            
            print(f"Recommendations returned: {after_draft_recommendations}")
            if not after_draft_recommendations:
                print("SUCCESS: As expected, no recommendations were returned after the draft was complete.")
            else:
                print("FAILURE: Recommendations were returned for a completed draft, which is incorrect.")
        else:
            print("FAILURE: No recommendations were returned for the last pick. This could indicate an off-by-one error.")
    
    print("\nLast Pick Test Finished.") 

    # --- Deck Building Test ---
    print("\n--- Testing Deck Building from Full Draft Pool ---")

    # Collect the names of all picks (limit to first 42 or less if file shorter)
    draft_pool_cards = []
    for i, pick_data in enumerate(picks_data[:42]):
        pick_name = pick_data.get("pick", {}).get("name")
        if not pick_name:
            print(f"Warning: Missing pick name for pick {i + 1}, skipping.")
            continue
        draft_pool_cards.append({"name": pick_name})

    if len(draft_pool_cards) < 40:
        print("Not enough cards in draft pool to build a 40-card deck. Skipping deck build test.")
    else:
        deck_prediction = recommender.build_model.predict(draft_pool_cards)

        # Print the deck prediction in a readable format
        recommender.build_model.print_prediction(deck_prediction)

        # Verify total deck size is 40 cards
        nb_total = sum(len(scores) for scores in deck_prediction["predicted_deck"]["nonbasics"].values())
        b_total = sum(len(scores) for scores in deck_prediction["predicted_deck"]["basics"].values())
        total_size = nb_total + b_total

        if total_size == 40:
            print("SUCCESS: Deck builder produced a 40-card deck as expected.")
        else:
            print(f"FAILURE: Deck builder produced {total_size} cards (expected 40).")

    print("\nDeck Building Test Finished.") 
