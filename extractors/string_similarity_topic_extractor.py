from extractors.topic_extractor import TopicExtractor
from paper_chunk import PaperChunk


class StringSimilarityTopicExtractor(TopicExtractor):

    def __init__(self, paper: PaperChunk):
        super().__init__(paper)

    def get_topics(self, k: int = 10):
        pass
