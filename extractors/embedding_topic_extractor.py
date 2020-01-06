from typing import List
from paper_chunk import PaperChunk
from extractors.topic_extractor import TopicExtractor
import numpy as np
from bert_embedding import BertEmbedding
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from time import time


class EmbeddingTopicExtractor(TopicExtractor):

    def __init__(self, field_embeddings_path='../data/fields_embedding.pkl', min_similarity=0.8):
        super().__init__(field_embeddings_path)
        self.min_similarity = min_similarity
        self.model = BertEmbedding()
        with open(self.data_path, 'rb') as infile:
            self.fields_embeddings: dict = pickle.load(infile)

    def get_embeddings(self, string):
        return np.mean(np.array(self.model([string])[0][1]), axis=0)

    def get_topics(self, paper: PaperChunk, k: int = 10) -> List[str]:
        chunks = set(paper.get_grammar_chunks())
        chunks_embeddings = [self.get_embeddings(chunk) for chunk in chunks]
        fields = set(self.fields_embeddings.keys())
        fields_embeddings = [self.fields_embeddings[field] for field in fields]
        sim_matrix = pd.DataFrame(data=cosine_similarity(chunks_embeddings, fields_embeddings), index=chunks, columns=fields)
        values = sim_matrix.max(axis=1).sort_values(ascending=False)
        matching = values[values >= self.min_similarity]
        idx = sim_matrix.idxmax(axis=1)
        return idx[matching.index.tolist()].values.tolist()[:k]


if __name__ == '__main__':
    abstract = """
    Despite improved digital access to scholarly knowledge in recent decades, scholarly communication remains exclusively document-based. In this form, scholarly knowledge is hard to process automatically. We present the first steps towards a knowledge graph based infrastructure that acquires scholarly knowledge in machine actionable form thus enabling new possibilities for scholarly knowledge curation, publication and processing. The primary contribution is to present, evaluate and discuss multi-modal scholarly knowledge acquisition, combining crowdsourced and automated techniques. We present the results of the first user evaluation of the infrastructure with the participants of a recent international conference. Results suggest that users were intrigued by the novelty of the proposed infrastructure and by the possibilities for innovative scholarly knowledge processing it could enable.
    """
    extractor = EmbeddingTopicExtractor()
    start = time()
    print(extractor.get_topics(paper=PaperChunk(abstract=abstract)))
    end = time()
    print(f'Time: {end-start}')
