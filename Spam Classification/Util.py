import os
import nltk
import unicodedata
from bs4 import BeautifulSoup
import numpy as np

class HTMLNormalizer:

    FILENAME_OF_DICT_LEMMA = "generate.txt"

    def __init__(self, language='spanish', decode='utf-8'):
        self.language = language
        self.decode = decode
        self.dict_lemmas = self.get_dict_from_file()

    def normalize_html(self, original, is_html=True, remove_spec_char=True, remove_stopwords=True):
        html_raw_text = original
        if is_html:
            html_raw_text = self.remove_tags(original)
        
        tokens = self.get_tokens(html_raw_text)

        if remove_spec_char:
            tokens = self.remove_special_chars(tokens)

        # tokens = self.supress_accents(tokens)
        tokens = self.lower_tokens(tokens)
        if remove_stopwords:
            tokens = self.remove_stop_words(tokens)
            
        lemmatized_tokens = self.lemmatize(tokens)
        return lemmatized_tokens

    def remove_tags(self, html):
        return BeautifulSoup(html, 'html.parser').get_text()
    
    def get_tokens(self, raw_text):
        return nltk.word_tokenize(raw_text, language=self.language)

    def remove_special_chars(self, tokens):
        return [w for w in tokens if w.isalpha()]

    def supress_accents(self, tokens):
        tokens_norm = []

        for word in tokens:
            word_in_nfkd = unicodedata.normalize('NFKD', word)
            ascii_word = word_in_nfkd.encode('ascii', 'ignore')
            tokens_norm.append(ascii_word.decode('utf-8', 'ignore'))
        return tokens_norm

    def lower_tokens(self, tokens):
        return [w.lower() for w in tokens]

    def remove_stop_words(self, tokens):
        stopwords = nltk.corpus.stopwords.words(self.language)
        
        tokens_filtered = list(filter(lambda word: word not in stopwords, tokens))
        return tokens_filtered

    def get_dict_from_file(self):
        file_generator = open( self.FILENAME_OF_DICT_LEMMA, mode='r', encoding='latin-1')

        dict_lemmas = {}
        for line in file_generator:
            tokens = line.replace('#','').split()

            if len(tokens) == 0:
                continue
            
            word = tokens[0]
            lemma = tokens[-1]
            dict_lemmas[word] = lemma
            # dict_lemmas[(word, tag Regex)] = (lemma, tag)

        return dict_lemmas

    def lemmatize(self, tokens):
        lemmatized_tokens = []

        for token in tokens:
            lemma = self.dict_lemmas.get(token, token)
            lemmatized_tokens.append(lemma)

        return lemmatized_tokens


class HTMLNormalizerSents(HTMLNormalizer):

    def normalize_html(self, original_html):
        html_raw_text = self.remove_tags(original_html)
        sents_tokenized = self.get_sents_tokenized(html_raw_text)
        sents_tokenized = self.remove_special_chars(sents_tokenized)
        
        
        sents_tokenized = self.lower_sents(sents_tokenized)

        sents_tokenized = self.remove_stop_words(sents_tokenized)

        lemmatized_tokens = self.lemmatize_sents(sents_tokenized)
        return lemmatized_tokens


    def get_sents_tokenized(self, html_raw_text):
        sents = nltk.sent_tokenize(html_raw_text)
        sents_tokenized = [super(HTMLNormalizerSents, self).get_tokens(sent) for sent in sents]
        return sents_tokenized

    def remove_special_chars(self, sents_tokenized):
        return [ super(HTMLNormalizerSents, self).remove_special_chars(sent) for sent in sents_tokenized ]
        
    def lower_sents(self, sents_tokenized):
        return [ super(HTMLNormalizerSents, self).lower_tokens(sent) for sent in sents_tokenized ]

    def remove_stop_words(self, sents_tokenized):
        return [ super(HTMLNormalizerSents, self).remove_stop_words(sent) for sent in sents_tokenized]

    def lemmatize_sents(self, sents_tokenized):
        return [ super(HTMLNormalizerSents, self).lemmatize(sent) for sent in sents_tokenized]


class HTMLNormalizerTagged(HTMLNormalizer):

    TAG_NON_EXIST = 'unknown'
    FILENAME_OF_DICT_LEMMA = "generate.txt"

    def __init__(self, language='spanish', decode='utf-8'):
        self.language = language
        self.decode = decode
        self.dict_lemmas = self.get_dict_from_file()
        self.tagger = self.get_trained_tagger()
        

    def normalize_html(self, original_html):
        html_raw_text = self.remove_tags(original_html)
        tokens = self.get_tokens(html_raw_text)
        tokens = self.remove_special_chars(tokens)
        tokens = self.lower_tokens(tokens)
        tokens = self.remove_stop_words(tokens)
        tokens_tagged = self.tag_words_with_tagger(tokens)

        lemmatized_tokens = self.lemmatize(tokens_tagged)
        return lemmatized_tokens

    def get_pattern_regex_tagger(self):
        patterns = [
            (r'^.*ez$', 'n'), # Ibanez, Chavez, Gutierrez, Gonzalez
            (r'^.*ismo$', 'a'), # gigantismo, activismo
            (r'^.*(ar|er|ir)$', 'v') # correr, activar, 
        ]

        return patterns

    def get_trained_tagger(self):
        train_data = nltk.corpus.cess_esp.tagged_sents()

        patterns = self.get_pattern_regex_tagger()
        backoff_tagger = nltk.RegexpTagger(patterns, nltk.DefaultTagger(self.TAG_NON_EXIST))

        return nltk.UnigramTagger(train_data, None, backoff_tagger)

    def tag_words_with_tagger(self, tokens):
        tokens_tagged = self.tagger.tag(tokens)
        tokens_tagged_norm = []
        
        for token, tag in tokens_tagged:
            if tag != self.TAG_NON_EXIST:
                tokens_tagged_norm.append((token, tag[0].lower()))
            else:
                tokens_tagged_norm.append((token, tag))

        return tokens_tagged_norm


    def get_dict_from_file(self):
        file_generator = open( self.FILENAME_OF_DICT_LEMMA, mode='r', encoding='latin-1')

        dict_lemmas = {}
        for line in file_generator:
            tokens = line.replace('#','').split()

            if len(tokens) == 0:
                continue
            
            word = tokens[0]
            lemma = tokens[-1]    
            tag = tokens[1][0].lower()


            dict_lemmas[(word, tag)] = (lemma, tag)
            # dict_lemmas[(word, tag Regex)] = (lemma, tag)

        return dict_lemmas





def getSimilars(vocabulary, contexts, word_target):
    vectors = {}

    idx_of_word_in_vector = {word: idx for idx, word in enumerate(vocabulary)}

    for target, context in contexts.items():
        vector = np.zeros(len(vocabulary))

        for word_in_context in context:
            idx_word = idx_of_word_in_vector[word_in_context]

            vector[idx_word] += 1

        vectors[target] = vector

    vector_target = vectors[word_target]

    similarities = []

    np.seterr('raise')
    for word in vocabulary:
        vector_candidate = vectors[word]
        
        den = np.linalg.norm(vector_candidate) * np.linalg.norm(vector_target)
        num = np.dot(vector_target, vector_candidate)

        try:
            cosine = num / den
            similarities.append((word, cosine))
        except Exception:
            pass
        

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities


def openFiles(encoding='utf-8'):

    NAME_FOLDER = 'EXCELSIOR_100_files'
    htmls = []
    for filename in os.listdir(NAME_FOLDER):
        dir_file = os.path.join(NAME_FOLDER, filename)

        with open(dir_file, mode='r', encoding=encoding) as f:
            html = f.read()
            htmls.append(html)

    return htmls

