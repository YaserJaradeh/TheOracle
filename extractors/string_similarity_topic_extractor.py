from typing import List
from extractors.topic_extractor import TopicExtractor
from paper_chunk import PaperChunk
import pickle
from Levenshtein.StringMatcher import StringMatcher
import itertools
from time import time
from fuzzywuzzy import fuzz


class StringSimilarityTopicExtractor(TopicExtractor):

    def __init__(self, min_similarity=0.9):
        super().__init__()
        self.min_similarity = min_similarity
        with open(self.fields_path, 'rb') as infile:
            self.fields = pickle.load(infile)

    def get_topics_old(self, paper: PaperChunk, k: int = 10) -> List[str]:
        possibilities = list(itertools.product(set(paper.everygram), set(self.fields)))
        print(len(possibilities))
        similarities = [(p[0], p[1], self.get_levenshtein_similarity(p[0], p[1])) for p in possibilities]
        return [t[1] for t in sorted(similarities, key=lambda x: x[2], reverse=True) if t[2] >= self.min_similarity][:k]

    def get_topics(self, paper: PaperChunk, k: int = 10) -> List[str]:
        possibilities = [tup for tup in itertools.product(set(paper.everygram), set(self.fields)) if len(tup[0].split()) == len(tup[1].split())]
        similarities = [(p[0], p[1], self.get_levenshtein_similarity(p[0], p[1])) for p in possibilities]
        return [t[1] for t in sorted(similarities, key=lambda x: x[2], reverse=True) if t[2] >= self.min_similarity][:k]

    @staticmethod
    def get_levenshtein_similarity(str1: str, str2: str):
        return StringMatcher(None, str1, str2).ratio()

    @staticmethod
    def get_similarity(strings):
        str1, str2 = strings
        return str1, str2, StringSimilarityTopicExtractor.get_levenshtein_similarity(str1, str2)

    @staticmethod
    def get_fuzzy_similarity(str1: str, str2: str):
        return fuzz.ratio(str1, str2) / 100.0


if __name__ == '__main__':
    abstract = """
    Despite improved digital access to scholarly knowledge in recent decades, scholarly communication remains exclusively document-based. In this form, scholarly knowledge is hard to process automatically. We present the first steps towards a knowledge graph based infrastructure that acquires scholarly knowledge in machine actionable form thus enabling new possibilities for scholarly knowledge curation, publication and processing. The primary contribution is to present, evaluate and discuss multi-modal scholarly knowledge acquisition, combining crowdsourced and automated techniques. We present the results of the first user evaluation of the infrastructure with the participants of a recent international conference. Results suggest that users were intrigued by the novelty of the proposed infrastructure and by the possibilities for innovative scholarly knowledge processing it could enable.
    """
    test = StringSimilarityTopicExtractor()
    start = time()
    print(test.get_topics(PaperChunk(abstract=abstract)))
    end = time()
    print(f"time: {end - start}")
