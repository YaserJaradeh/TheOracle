from typing import List
from paper_chunk import PaperChunk
from extractors.topic_extractor import TopicExtractor
from gensim.models import Word2Vec
import nltk


class WordEmbeddingTopicExtractor(TopicExtractor):

    def __init__(self, embedding_model_path: str = '../data/w2v_model.bin'):
        super().__init__(embedding_model_path)
        self.model = Word2Vec.load(self.data_path)

    def get_topics(self, paper: PaperChunk, k: int = 10) -> List[str]:
        chunks = paper.get_grammar_chunks()
        for chunk in chunks:
            for ngram in nltk.everygrams(chunk, min_len=1, max_len=3):
                gram = '_'.join(ngram)
                if gram in self.model:
                    sim_words = self.model[gram]
                else:
                    embeddings = []
                    for word in ngram:
                        if word in self.model:
                            embeddings.append(self.model.wv[word])


if __name__ == '__main__':
    abstract = """
    Despite improved digital access to scholarly knowledge in recent decades, scholarly communication remains exclusively document-based. In this form, scholarly knowledge is hard to process automatically. We present the first steps towards a knowledge graph based infrastructure that acquires scholarly knowledge in machine actionable form thus enabling new possibilities for scholarly knowledge curation, publication and processing. The primary contribution is to present, evaluate and discuss multi-modal scholarly knowledge acquisition, combining crowdsourced and automated techniques. We present the results of the first user evaluation of the infrastructure with the participants of a recent international conference. Results suggest that users were intrigued by the novelty of the proposed infrastructure and by the possibilities for innovative scholarly knowledge processing it could enable.
    """
    extractor = WordEmbeddingTopicExtractor()
    extractor.get_topics(paper=PaperChunk(abstract=abstract))
