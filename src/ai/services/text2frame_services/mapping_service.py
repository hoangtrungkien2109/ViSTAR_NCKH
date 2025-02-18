"""
File contains function and class related to similarity sentence
Aim to convert from a paragraph of raw text to a list of existing words from database
"""

import re
# import time
import json
from typing import List, Dict
# import orjson
import torch
import numpy as np
from loguru import logger
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
# from sentence_transformers import SentenceTransformer
# from sentence_transformers import SentenceTransformer
from src.ai.services.text2frame_services.elastic_service import ESEngine
from src.ai.services.utils.decorator import processing_time
from src.ai.services.text2frame_services.models.custom_list import WordList, Word
from src.ai.services.text2frame_services.models.custom_list import WordList, Word


class SimilaritySentence():
    """Class for Elastic Search"""
    # _instance = None

    # def __new__(cls, *args, **kwargs) -> None:
    #     if cls._instance is None:
    #         cls._instance = super(SimilaritySentence, cls).__new__(cls, *args, **kwargs)
    #     return cls._instance

    def __init__(self,
            default_dict_path: str = "D:/NCKH/Text_to_Sign/ViSTAR/src/ai/services/text2frame_services/data/character_dict.rar",
            ner_model_name: str = "NlpHUST/ner-vietnamese-electra-base",
            ner_min_length: int = 5,
            ner_max_length: int = 20,
        ):
        self.es: ESEngine = ESEngine()
        self.word_queue: WordList = WordList()
        self.ner_list: WordList = WordList()
        self.ner_min_length = ner_min_length
        self.ner_max_length = ner_max_length

        tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
        self.ner = pipeline("ner", model=ner_model, tokenizer=tokenizer,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
        try:
            with open(default_dict_path, 'r') as f:
                self.default_frame = json.load(f)
                logger.info(f"Default character: {self.default_frame.keys()}")
        except FileNotFoundError as e:
            raise e

    def clean_text(self, text: str) -> str:
        """Clean raw text"""
        text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", text)
        return text

    def push_word(self, word: str) -> None:
        self.ner_list.put(Word(word=word))
        logger.debug(f"Len of word list: {self.ner_list.get_len()}")
        logger.debug(f"Word list: {self.ner_list.get_raw_sentence()}")
        if self.ner_list.get_len() > self.ner_min_length:
            self._detect_name()
            if self.ner_list.get_len() > self.ner_max_length:
                self.ner_list.pop()
                self.ner_list.pointer -= 1
            self.word_queue.put(self.ner_list.get(self.ner_list.pointer))
            self.ner_list.pointer += 1

    @processing_time
    def get_frame(self) -> List[np.ndarray]:
        if self.word_queue.get_len() > 0:
            current_word = self.word_queue.pop()
            if current_word.is_name == True:
                logger.success("Sent a name (list of character) to streaming")
                return [self.default_frame[character.lower()] for character in current_word.sub_words]
            else:
                searched_result = self.es.search(word=current_word.word)
                logger.debug(f"CURRENT: {current_word.word}")
                if len(searched_result) > 0:
                    logger.success("Sent frame to streaming")
                    frames = self.es.decode_frame(searched_result[0]["_source"]["frame"])
                    return [frames]
                else:
                    logger.error("Word is not contained in DB")
                    return [self.default_frame["default"]]
        else:
            logger.error("Default")
            return [self.default_frame["default"]]

    @processing_time
    def _detect_name(self) -> None:
        """Detect all entity in sentence"""
        entities = self.ner(self.ner_list.get_raw_sentence())
        for entity in reversed(entities):
            if "PERSON" in entity['entity']:
                self.ner_list.get(entity['index'] - 1).is_name = True
                self.ner_list.get(entity['index'] - 1).sub_words = [
                    character for character in self.remove_accents(self.ner_list.get(entity['index'] - 1).word).upper()
                ]
                logger.success(f"Name: {self.ner_list.get(entity['index'] - 1).word}, Convert: {self.ner_list.get(entity['index'] - 1).sub_words}")

    def remove_accents(self, old: str):
        """
        Removes common accent characters, lower form.
        Uses: regex.
        """
        new = old.lower()
        new = re.sub(r'[àáảãạ]', 'a', new)
        new = re.sub(r'[ằắẳẵặă]', 'ă', new)
        new = re.sub(r'[ầấẩẫậâ]', 'â', new)
        new = re.sub(r'[èéẻẽẹ]', 'e', new)
        new = re.sub(r'[ềếểễệê]', 'ê', new)
        new = re.sub(r'[ìíỉĩị]', 'i', new)
        new = re.sub(r'[òóỏõọ]', 'o', new)
        new = re.sub(r'[ồốổỗộô]', 'o', new)
        new = re.sub(r'[ồốổỗộô]', 'o', new)
        new = re.sub(r'[ùúủũụ]', 'u', new)
        new = re.sub(r'[ừứửữựư]', 'ư', new)
        new = re.sub(r'[ỳýỷỹỵy]', 'y', new)
        return new

# ss = SimilaritySentence()
