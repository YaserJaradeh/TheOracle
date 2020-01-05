from lxml import etree
import pickle
import tomotopy as tp
from paper_chunk import PaperChunk
from bert_embedding.bert import BertEmbedding
import numpy as np


def serialize_mag_topics():
    topics = set()
    context = etree.iterparse('./data/data.xml', huge_tree=True, remove_blank_text=True, recover=True)
    for event, element in context:
        if element.tag == '{http://www.w3.org/2005/sparql-results#}result':
            for child in element.iterchildren():
                if child.get('name') == 'topics':
                    for topic in child[0].text.strip().split("|"):
                        topics.add(topic)
    final_topics = list(topics)
    with open('./data/fields.pkl', 'wb') as f:
        pickle.dump(final_topics, f)


def create_lda_model():
    mdl = tp.LDAModel(k=150)
    context = etree.iterparse('./data/data.xml', huge_tree=True, remove_blank_text=True, recover=True)
    for event, element in context:
        if element.tag == '{http://www.w3.org/2005/sparql-results#}result':
            for child in element.iterchildren():
                if child.get('name') == 'abstract':
                    if child[0].text is None:
                        print(child[0])
                        print("element skipped because of empty text!!")
                        continue
                    paper = PaperChunk(abstract=child[0].text)
                    mdl.add_doc(paper.everygram)
    print("done adding")
    for i in range(0, 300, 10):
        mdl.train(10)
        print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))
    print("done training")
    for k in range(mdl.k):
        print('Top 10 words of topic #{}'.format(k))
        print(mdl.get_topic_words(k, top_n=10))
    mdl.save('./data/lda_mag_abstracts_k150.bin')


def create_bert_embeddings():
    with open('./data/fields.pkl', 'rb') as infile:
        fields = pickle.load(infile)
        print('fields loaded!')
        bert = BertEmbedding()
        print('Bert Embedding initialized!')
        embeddings = {field: np.mean(np.array(bert([field])[0][1]), axis=0) for field in fields}
        print('Embeddings created for all fields!!')
        with open('./data/fields_embedding.pkl', 'wb') as outfile:
            fields = pickle.dump(embeddings, outfile)


if __name__ == '__main__':
    create_bert_embeddings()
