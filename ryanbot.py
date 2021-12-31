# 1. pack: self.t x n dimensional vector such that the ith element is 1 if the ith card in the set is in the pack. You have a dictionary mapping these ids to card names.
# 2. shifted picks: self.t length vector where each element corresponds to the PRIOR pick. So the first element is always n + 1 (a card id that doesnt exist), and the second element is the card ID for what the human took P1P1
# 3. Positional information: literally just the list [0,1,2,...,41]
import numpy as np
import tensorflow as tf
import os
import pickle

MODEL_LOCATION = "Models/vow_emb_deep"

class RyanBot:
    def __init__(self):
        self.model = self.load_model_and_attrs(MODEL_LOCATION)
        self.clear_callback()
    def get_idx(self, pack, pick):
        idx = ((pack - 1) * self.pack_size) + pick - 1
        self.last_idx = idx
        return idx
    def pack_callback(self, pack, pick, pack_card_names):
        print('pack_callback!', pack, pick, pack_card_names)
        idx = self.get_idx(pack, pick)
        # run predictions without considering pack context
        self.run_prediction(pack, pick, full_set=True)
        # when we get to a pack we ensure all cards are empty
        # we do this because we initialize all packs to be every card
        self.pack[idx, :] = 0.0
        print(idx)
        for name in pack_card_names:
            card_idx = self.get_card_idx(name)
            self.pack[idx, card_idx] = 1
        self.run_prediction(pack, pick)
    def pick_callback(self, pack, pick, card_name):
        print('pick_callback!', pack, pick, card_name)
        idx = self.get_idx(pack, pick)
        print(idx)
        if idx + 1 < self.t:
            card_idx = self.get_card_idx(card_name)
            self.shifted_picks[idx + 1] = card_idx
    def clear_callback(self):
        print('clear_callback!')
        self.pack = np.ones((self.t, self.n_cards), dtype=np.float32)
        self.shifted_picks = np.ones(self.t, dtype=np.int32) * self.n_cards
        self.predictions = np.zeros((self.t, self.n_cards), dtype=np.float32)
        self.predictions_out_of_pack = np.zeros((self.t, self.n_cards), dtype=np.float32)
        self.positions = np.arange(self.t, dtype=np.int32)
    def run_prediction(self, pack, pick, full_set=False):
        model_input = (
            np.expand_dims(self.pack, 0),
            np.expand_dims(self.shifted_picks, 0),
            np.expand_dims(self.positions, 0),
        )
        model_output, _ = self.model(model_input, training=False, return_attention=True)
        idx = self.get_idx(pack, pick)
        prediction = np.squeeze(model_output)[idx]
        #self.rating(idx, self.idx_to_name[prediction])
        if full_set:
            self.predictions_out_of_pack[idx] = prediction
        else:
            self.predictions[idx] = prediction

    def rating(self, idx, card_name, full_set=False):
#        print(self.predictions)
        card_idx = self.get_card_idx(card_name)
        if full_set:
            card_rating = self.predictions_out_of_pack[idx, card_idx]
        else:
            card_rating = self.predictions[idx, card_idx]
        print('rating', card_name, card_rating)
        return np.round(card_rating, 3)
#        return self.predictions[pick_number][lookup_card(card_name)]
    def get_card_idx(self, card_name):
        name = card_name.lower().split("//")[0].strip()
        return self.name_to_idx[name]
    def load_model_and_attrs(self, location):
        model_loc = os.path.join(location, "model")
        data_loc = os.path.join(location, "attrs.pkl")
        model = tf.saved_model.load(model_loc)
        with open(data_loc, "rb") as f:
            attrs = pickle.load(f)
        self.t = attrs['t']
        self.pack_size = self.t//3
        self.n_cards = attrs['n_cards']
        self.idx_to_name = attrs['idx_to_name']
        self.name_to_idx = {v:k for k,v in self.idx_to_name.items()}
        return model

        