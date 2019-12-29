from typing import List
from paper_chunk import PaperChunk


class TopicExtractor:

    def __init__(self, mag_fields: str = '../data/fields.pkl'):
        self.fields_path = mag_fields

    def get_topics(self, paper: PaperChunk, k: int = 10) -> List[str]:
        pass
