"""
File contains function and class related to similarity sentence
Aim to convert from a paragraph of raw text to a list of existing words from database
"""

import re
# import time
import json
# import orjson
import torch
import numpy as np
from pyvi import ViTokenizer
from loguru import logger
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from src.ai.services.text2frame_services.elastic_service import es
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

    def __init__(self, embed_model_name: str = "dangvantuan/vietnamese-embedding",
                 ner_model_name: str = "NlpHUST/ner-vietnamese-electra-base",
                 character_dict_path: str = "character_dict.json"):
        self.embed_model = SentenceTransformer(embed_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(ner_model_name)
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ner_model_name)
        self.ner = pipeline("ner", model=self.ner_model, tokenizer=self.tokenizer,
                       device='cuda' if torch.cuda.is_available() else 'cpu')
        self.es = es
        self.character_dict = {}
        try:
            with open(character_dict_path, 'r') as f:
                self.character_dict = json.load(f)

        except FileNotFoundError as e:
            logger.error(e)

    def clean_text(self, text: str) -> str:
        """Clean raw text"""
        text = re.sub(r"[-()\"#/@;:<>{}=~|.?,]", "", text)
        return text

    def _tokenize_text(self, text: str) -> str:
        """Tokenize text"""
        segment = ViTokenizer.tokenize(text)
        return segment

    @processing_time
    def convert_sentence_to_words(self, sentence: str) -> list[str]:
        """Convert raw sentence into words from database"""
        SPECIAL_TOKEN = "SPECIAL_TOKEN"
        map_word_to_frame = {}
        named_sentence = sentence.split(" ")
        _, name_indices = self._detect_name(sentence)
        names = []
        for idx, word in enumerate(named_sentence):
            if name_indices[idx] == 1:
                names.append(remove_accents(word).upper())
                named_sentence[idx] = SPECIAL_TOKEN
        pre_embedding = self.embed_model.encode([sentence])
        words_to_search = []
        segment_sentence = self._tokenize_text(self.clean_text(" ".join(named_sentence)))
        logger.info(f"Segment: {segment_sentence}")
        word_list = segment_sentence.split(" ")
        cnt = 0
        for idx, word in enumerate(word_list):
            if word == SPECIAL_TOKEN:
                words_to_search.append(names[cnt])
                cnt+=1
            else:
                words_to_search.append(" ".join(word.split("_")) if "_" in word else word)
        logger.info(f"Word Search: {words_to_search}")
        result_sentence = []
        existing_words = []
        scores = []
        max_similarity_score = 0
        for word in words_to_search:
            if word in names:
                scores.append(20)
                existing_words.append(word)
                map_word_to_frame[word] = self._search_name(word)
                logger.info("Chars: ",len(map_word_to_frame[word]))
                continue
            searching_result = es.search(word)
            if len(searching_result) > 0:
                scores.append(searching_result[0]["_score"])
                existing_words.append(searching_result[0]["_source"]["word"])
                map_word_to_frame[existing_words[-1]] = self.es.decode_frame(searching_result[0]["_source"]["frame"])
        logger.warning(f"map keys: {map_word_to_frame.keys()}")
        for base_score in range(5, int(max(scores))):
            current_words = []
            for idx, score in enumerate(scores):
                if score > base_score:
                    current_words.append(existing_words[idx])
            current_sentence = " ".join(current_words)
            post_embedding = self.embed_model.encode([current_sentence])
            similarity = self.embed_model.similarity(pre_embedding, post_embedding)
            if similarity > max_similarity_score:
                max_similarity_score = similarity
                result_sentence = current_words.copy()

        # for idx, word in enumerate(result_sentence):
        #     if word in names:
        #         result_sentence.pop(idx)
        #         for char in reversed(self._process_name(word)):
        #             result_sentence.insert(idx, char)
        result_frames = []
        # logger.info(len(map_word_to_frame["chúng tôi"]))
        # logger.info(len(map_word_to_frame["HOANG"]))
        for word in result_sentence:
            if word in names:
                result_frames.extend(map_word_to_frame[word])  # FIX
            else:
                result_frames.append(map_word_to_frame[word])
                
        # logger.warning(f"{len(result_frames)}, {len(result_frames[0])}, {len(result_frames[1])},   {len(result_frames[2])}")
        return result_frames

    def _detect_name(self, sentence: str) -> dict:
        """Detect all entity in sentence"""
        masked_index = [0] * len(self.clean_text(sentence).split(" "))
        entities = self.ner(sentence)
        sentence = list(sentence)
        logger.info(entities)
        for entity in reversed(entities):
            if "PERSON" in entity['entity']:
                masked_index[entity['index'] - 1] = 1
                logger.info(entity['word'])
                for idx in range(entity['end'] - 1, entity['start'], -1):
                    sentence[idx] = remove_accents(sentence[idx]).upper()
                    sentence.insert(idx, ' ')
        result = ''.join(sentence)
        logger.info(f"Detected: {result}")
        return result, masked_index

    def _process_name(self, name: str) -> str:
        """Split an uppercased name into a list of character"""
        return [char for char in list(name)]

    def _search_name(self, name: str) -> list:
        """Convert from name to list of frames of characters"""
        result = []
        for char in self._process_name(name):
            # logger.warning(f"char: {char}")
            if char.lower() in self.character_dict:
                result.append(
                    self.character_dict[char.lower()]
                    )
        
        return result

ss = SimilaritySentence()
