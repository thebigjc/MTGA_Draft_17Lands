import json
import requests
from typing import List

PACK_PARSER_URL = "https://us-central1-mtgalimited.cloudfunctions.net/pack_parser"
REQUEST_TIMEOUT_SEC = 5.0

class OCR:
    def __init__(self, url: str = PACK_PARSER_URL):
        self.url = url

    def get_pack(self, card_names: List[str], screenshot: str, timeout: float = REQUEST_TIMEOUT_SEC) -> List[str]:
        """
        Calls an OCR endpoint with a screenshot and a list of names,
        retrieves the OCR results for the names detected in the screenshot.
        
        Args:
            screenshot (base64str): The screenshot image data.
            names (list of str): A list of names to search for in the screenshot.
        
        Returns:
            list of str: A list of names detected in the screenshot through OCR.
        """
        basic_lands = ["plains", "island", "swamp", "mountain", "forest"]
        card_names_with_basics = card_names + basic_lands

        data = {
            "card_names": card_names_with_basics,
            "image": screenshot
        }

        headers = {'Content-Type': 'application/json'}
        response = requests.post(self.url, headers=headers, data=json.dumps(data), timeout=timeout)
        received_names = json.loads(response.text)

        return received_names
