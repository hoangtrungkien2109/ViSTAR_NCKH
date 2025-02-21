from pydantic import BaseModel
from pyvi import ViTokenizer
from typing import List


class Word(BaseModel):
    word: str = ""
    sub_words: List[str] = []
    is_name: bool = False

class WordList(BaseModel):
    word_list: List[Word] = []

    def get_len(self):
        return len(self.word_list)

    def get_raw_sentence(self):
        return " ".join([temp_word.word for temp_word in self.word_list])

    def put(self, word: str):
        self.word_list.append(word)

    def get(self, index: int = 0):
        return self.word_list[index]

    def pop(self, index: int = 0):
        return self.word_list.pop(index)

    def tokenize_text(self):
        """Tokenize text"""
        segments = ViTokenizer.tokenize(self.get_raw_sentence())
        