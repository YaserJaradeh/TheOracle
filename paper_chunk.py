from nltk.util import everygrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class PaperChunk:

    def __init__(self, url=None, title=None, abstract=None):
        self.text = abstract
        self.url = url
        self.title = title
        self._stopwords = set(stopwords.words('english'))

    @property
    def clean_text(self):
        word_tokens = word_tokenize(self.text)
        return [t for t in word_tokens if t.lower() not in self._stopwords]

    @property
    def everygram(self):
        word_tokens = word_tokenize(self.text)
        ngrams = [' '.join(w) for w in everygrams(word_tokens, max_len=2)]
        return [w for w in ngrams if len(set(word_tokenize(w.lower())) & self._stopwords) == 0]
