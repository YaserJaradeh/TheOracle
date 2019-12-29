from lxml import etree
import pickle


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


if __name__ == '__main__':
    serialize_mag_topics()
