from src.draft_inference import DraftModelWrapper
from src.dataset import Dataset
from typing import List, Dict

class History:
    def __init__(self):
        self.pack = []
        self.pick = -1

class MLRecommender:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = DraftModelWrapper(model_path)
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
            if h.pack_cards != pack_cards:
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
            self.model.expansion_code, # TODO: Get expansion code from current draft
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
        while len(tokens) < self.model.pick_positions[0] -2:
            tokens.append("Plains")

        tokens.append("[PACK_CARDS_END]")
        tokens.append("[PICK]")

        return tokens
    
    def get_recommendations(self, card_list: List[Dict], set_data: Dataset) -> List[Dict]:
        current_pack_cards = self._get_current_pack_cards(card_list)

        # If the current pack is empty, return an empty list
        if len(current_pack_cards) == 0:
            return []
        
        if len(self.history) == 0:
            tokens = self.compare_tokens(current_pack_cards)
        else:            
            tokens = self.pick_tokens(set_data)
            if tokens[-1] != "[PICK]":
                return {}

        # Get model predictions
        predictions = self.model.get_top_k_predictions(tokens, k=len(current_pack_cards))
                
        return predictions 