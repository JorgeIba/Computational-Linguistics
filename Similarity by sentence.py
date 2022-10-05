import numpy as np
from Util import openFiles, HTMLNormalizer, getSimilars
import nltk

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


def getAllContextsBySent(sents_tokenized):
    word_to_context = {}

    for sent in sents_tokenized:

        for idx, word in enumerate(sent):

            if word not in word_to_context:
                word_to_context[word] = []

            word_to_context[word] += (sent[:idx] + sent[(idx+1):])
            
    return word_to_context
    

all_htmls = openFiles()
normalizer = HTMLNormalizerSents()


sents_normalized = []
for html in all_htmls:
    sents_normalized += normalizer.normalize_html(html)

contexts = getAllContextsBySent(sents_normalized)

vocabulary = sorted(set( [word for sent in sents_normalized for word in sent] ))
similars = getSimilars(vocabulary, contexts, 'banco')


for i in range(20):
    print(similars[i])
