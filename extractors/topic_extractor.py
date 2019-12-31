from typing import List
from paper_chunk import PaperChunk


class TopicExtractor:

    def __init__(self, data_path: str):
        self.data_path = data_path

    def get_topics(self, paper: PaperChunk, k: int = 10) -> List[str]:
        pass
