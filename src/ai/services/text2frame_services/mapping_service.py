"""
File contains function and class related to similarity sentence
Aim to convert from a paragraph of raw text to a list of existing words from database
"""

import re
# import time
import json
from typing import List, Dict
from threading import Thread
from queue import Queue
# import orjson
import torch
import numpy as np
from pyvi import ViTokenizer
from loguru import logger
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from src.ai.services.text2frame_services.elastic_service import ESEngine
from src.ai.services.utils.decorator import processing_time


def remove_accents(old: str):
    """
    Removes common accent characters, lower form.
    Uses: regex.
    """
    new = old.lower()
    new = re.sub(r'[àáảãạằắẳẵặăầấẩẫậâ]', 'a', new)
    new = re.sub(r'[èéẻẽẹềếểễệê]', 'e', new)
    new = re.sub(r'[ìíỉĩị]', 'i', new)
    new = re.sub(r'[òóỏõọồốổỗộôờớởỡợơ]', 'o', new)
    new = re.sub(r'[ừứửữựưùúủũụ]', 'u', new)
    return new


class SimilaritySentence():
    """Class for Elastic Search"""
    _instance = None

    def __new__(cls, *args, **kwargs) -> None:
        if cls._instance is None:
            cls._instance = super(SimilaritySentence, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, default_dict_path: str = "character_dict.json"):
        self.es: ESEngine = ESEngine()
        self.default_frame: Dict = {}
        self.word_list: Queue = Queue()
        self.frame_list: Queue = Queue()
        try:
            with open(default_dict_path, 'r') as f:
                self.default_frame = json.load(f)

        except FileNotFoundError as e:
            raise e

    def clean_text(self, text: str) -> str:
        """Clean raw text"""
        text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", text)
        return text

    def push_word(self, words: str) -> None:
        self.word_list.put(words)

    def get_frame(self) -> List[np.ndarray]:
        if not self.word_list.empty():
            current_word = self.word_list.get()
            searched_result = self.es.search(word=current_word)
            # searched_word = searched_result[0]["_source"]["word"]
            frames = self.es.decode_frame(searched_result[0]["_source"]["frame"])
            return frames
        else:
            return self.default_frame["default"]

# ss = SimilaritySentence()
