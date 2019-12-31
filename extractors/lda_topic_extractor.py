from extractors.topic_extractor import TopicExtractor
from paper_chunk import PaperChunk
from typing import List
import tomotopy as tp
import numpy as np
from time import time


class LDATopicExtractor(TopicExtractor):

    def __init__(self, model_path: str = '../data/lda_model.bin'):
        super().__init__(model_path)
        self.mdl = tp.LDAModel.load(self.data_path)

    def get_topics(self, paper: PaperChunk, k: int = 10) -> List[str]:
        document = self.mdl.make_doc(words=paper.clean_text)  # paper.everygram can be used! still to be tested
        result = self.mdl.infer(document)
        index = np.argmax(result[0])
        return [w[0] for w in self.mdl.get_topic_words(index, top_n=k)]


if __name__ == '__main__':
    text = """
    Despite improved digital access to scholarly knowledge in recent decades, scholarly communication remains exclusively document-based. In this form, scholarly knowledge is hard to process automatically. We present the first steps towards a knowledge graph based infrastructure that acquires scholarly knowledge in machine actionable form thus enabling new possibilities for scholarly knowledge curation, publication and processing. The primary contribution is to present, evaluate and discuss multi-modal scholarly knowledge acquisition, combining crowdsourced and automated techniques. We present the results of the first user evaluation of the infrastructure with the participants of a recent international conference. Results suggest that users were intrigued by the novelty of the proposed infrastructure and by the possibilities for innovative scholarly knowledge processing it could enable.
    """
    mdl = LDATopicExtractor()
    start = time()
    print(mdl.get_topics(PaperChunk(abstract=text)))
    end = time()
    print(f'time: {end-start}')