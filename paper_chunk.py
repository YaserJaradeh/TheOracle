from nltk.util import everygrams
from nltk.corpus import stopwords
from nltk.tokenize import regexp_tokenize as tokenize
import html

PATTERN = r'\w+'


class PaperChunk:

    def __init__(self, url=None, title=None, abstract=None):
        self.text = abstract
        self.url = url
        self.title = title
        self._stopwords = set(stopwords.words('english'))

    @property
    def clean_text(self):
        word_tokens = tokenize(self.text, PATTERN)
        return [t for t in word_tokens if t.lower() not in self._stopwords]

    @property
    def everygram(self):
        word_tokens = tokenize(self.text, PATTERN)
        ngrams = [' '.join(html.unescape(w)) for w in everygrams(word_tokens, max_len=2)]
        return [w for w in ngrams if len(set(tokenize(w.lower(), PATTERN)) & self._stopwords) == 0]

    def get_ngrams(self, n=3):
        return [' '.join(w) for w in everygrams(self.clean_text, max_len=n)]
