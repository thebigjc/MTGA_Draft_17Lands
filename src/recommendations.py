from src.draft_inference import DraftModelWrapper
from src.dataset import Dataset

from typing import List, Dict
class MLRecommender:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = DraftModelWrapper(model_path)
        self.tokens = []
    
    def reset(self):
        self.tokens = []
            
    def add_pack_history(self, pack_cards: List[int], set_data: Dataset):
        if len(self.tokens) == 0:
            self.tokens.extend([
            "[DRAFT_START]",
            "[EXPANSION]", 
            self.model.expansion_code, # TODO: Get expansion code from current draft
            "[EXPANSION_END]",
            "[EVENT_TYPE]", "PremierDraft", "[EVENT_TYPE_END]",
            "[RANK]", "mythic", "7", "[RANK_END]"
        ])
        
        self.tokens.append("[PACK_CARDS]")
        pack_cards = [name.replace(" ", "_") for name in set_data.get_names_by_id(pack_cards)]
        self.tokens.extend(pack_cards)
        self.tokens.append("[PACK_CARDS_END]")

    def add_pick_history(self, pick: int, set_data: Dataset):
        self.tokens.append("[PICK]")
        pick_name = [name.replace(" ", "_") for name in set_data.get_names_by_id([pick])]
        self.tokens.extend(pick_name)
        self.tokens.append("[PICK_END]")
    
    def _get_current_pack_cards(self, card_list: List[Dict]) -> List[str]:
        return [card["name"].replace(" ", "_") for card in card_list]
    
    def get_recommendations(self, card_list: List[Dict]):
        current_pack_cards = self._get_current_pack_cards(card_list)

        tokens = self.tokens.copy()
        tokens.append("[PICK]")

        # If the current pack is empty, return an empty list
        if len(current_pack_cards) == 0:
            return []
        
        # Get model predictions
        predictions = self.model.get_top_k_predictions(tokens, k=len(current_pack_cards))
                
        return predictions 