from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import reuters
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk.data
import math
import re

SUMMARY_LENGTH = 3  # number of sentences in final summary
ideal_sent_length = 15.0
stemmer = SnowballStemmer("english")


class Summarizer():

    def __init__(self, articles):
        self._articles = articles

    def tokenize_and_stem(self, text):
        tokens = [word for sent in nltk.sent_tokenize(
            text) for word in nltk.word_tokenize(sent)]
        filtered = []

        # filter out numeric tokens, raw punctuation, etc.
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered.append(token)
        stems = [stemmer.stem(t) for t in filtered]
        return stems

    def score(self, article):
        sentences = self.split_into_sentences(article)
        frequency_scores = self.frequency_scores(article)

        for i, s in enumerate(sentences):
            wordsInSentence = len(s.split())
            if wordsInSentence >= 10:
                length_score = self.length_score(self.split_into_words(s))
                frequency_score = frequency_scores[i] * 2
                score = frequency_score + length_score
                self._scores[s] = score

    def generate_summaries(self):
        """ If article is shorter than the desired summary, just return the original articles."""

        # Rare edge case (when total num sentences across all articles is smaller than desired summary length)
        total_num_sentences = 0
        for article in self._articles:
            total_num_sentences += len(self.split_into_sentences(article))

        if total_num_sentences <= SUMMARY_LENGTH:
            return [x for x in self._articles]

        self.build_TFIDF_model()  # only needs to be done once

        self._scores = Counter()
        for article in self._articles:
            self.score(article)

        highest_scoring = self._scores.most_common(SUMMARY_LENGTH)

        # Appends highest scoring "representative" sentences, returns as a single summary paragraph.
        return ' '.join([sent[0] for sent in highest_scoring])

    # ----- STRING PROCESSING HELPER FUNCTIONS -----

    def split_into_words(self, text):
        """ Split a sentence string into an array of words """
        try:
            text = re.sub(r'[^\w ]', '', text)  # remove non-words
            return [w.strip('.').lower() for w in text.split()]
        except TypeError:
            return None

    def remove_smart_quotes(self, text):
        """ Only concerned about smart double quotes right now. """
        return text.replace(u"\u201c", "").replace(u"\u201d", "")

    def split_into_sentences(self, text):
        tok = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tok.tokenize(self.remove_smart_quotes(text))
        sentences = [sent.replace('\n', '')
                     for sent in sentences if len(sent) > 10]
        return sentences

    # ----- CALCULATING WEIGHTS FOR EACH FEATURE -----

    def length_score(self, sentence):
        """ Gives sentence score between (0,1) based on how close sentence's length is to the ideal length."""
        len_diff = math.fabs(ideal_sent_length - len(sentence))
        return len_diff / ideal_sent_length

    def build_TFIDF_model(self):
        self._tfidf = TfidfVectorizer(stop_words='english')
        self._tfidf.fit_transform(self._articles)

    def frequency_scores(self, article_text):
        """ Individual (stemmed) word weights are then calculated for each
            word in the given article. Sentences are scored as the sum of their TF-IDF word frequencies.
        """

        # Add our document into the model so we can retrieve scores
        response = self._tfidf.transform([article_text])
        feature_names = self._tfidf.get_feature_names()  # these are just stemmed words

        word_prob = {}  # TF-IDF individual word probabilities
        for col in response.nonzero()[1]:
            word_prob[feature_names[col]] = response[0, col]

        sent_scores = []
        for sentence in self.split_into_sentences(article_text):
            score = 0
            sent_tokens = self.tokenize_and_stem(sentence)
            for token in (t for t in sent_tokens if t in word_prob):
                score += word_prob[token]

            # Normalize score by length of sentence, since we later factor in sentence length as a feature
            sent_scores.append(score / len(sent_tokens))

        return sent_scores
