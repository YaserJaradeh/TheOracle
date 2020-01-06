from nltk.util import everygrams
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize as tokenize
import spacy
from nltk import RegexpParser, tree
import re
import html

PATTERN = r'\w+'
GRAMMAR = "CONCEPT: {<JJ.*>*<HYPH>*<JJ.*>*<HYPH>*<NN.*>*<HYPH>*<NN.*>+}"


class PaperChunk:

    def __init__(self, url=None, title=None, abstract=None):
        self.text = abstract
        self.url = url
        self.title = title
        self._stopwords = set(stopwords.words('english'))
        self.tagger = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

    @property
    def clean_text(self):
        word_tokens = tokenize(self.text, PATTERN)
        return [t for t in word_tokens if t.lower() not in self._stopwords]

    @property
    def everygram(self):
        word_tokens = tokenize(self.text, PATTERN)
        ngrams = [' '.join(html.unescape(w)) for w in everygrams(word_tokens, max_len=2)]
        return [w for w in ngrams if len(set(tokenize(w.lower(), PATTERN)) & self._stopwords) == 0]

    def __part_of_speech_tagger(self):
        doc = self.tagger(self.text)
        for token in doc:
            if token.tag_:
                yield token.text, token.tag_

    def get_grammar_chunks(self, grammar=GRAMMAR):
        pos_tags = self.__part_of_speech_tagger()
        grammar_parser = RegexpParser(grammar)
        chunks = list()
        pos_tags_with_grammar = grammar_parser.parse(list(pos_tags))
        for node in pos_tags_with_grammar:
            if isinstance(node, tree.Tree) and node.label() == 'CONCEPT':
                chunk = ''
                for leaf in node.leaves():
                    concept_chunk = leaf[0]
                    concept_chunk = re.sub('[\=\,\…\’\'\+\-\–\“\”\"\/\‘\[\]\®\™\%]', ' ', concept_chunk)
                    concept_chunk = re.sub('\.$|^\.', '', concept_chunk)
                    concept_chunk = concept_chunk.lower().strip()
                    chunk += ' ' + concept_chunk
                chunk = re.sub('\.+', '.', chunk)
                chunk = re.sub('\s+', ' ', chunk)
                chunks.append(chunk)
        return chunks

    def get_ngrams(self, n=3):
        return [' '.join(w) for w in everygrams(self.clean_text, max_len=n)]
