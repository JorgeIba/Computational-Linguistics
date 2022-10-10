from Util import HTMLNormalizerTagged, HTMLNormalizerSents, openFiles
import nltk
import math






def get_term_frequencies(sents, vocabulary):
    freq_dist = nltk.FreqDist()

    for word in vocabulary:
        for sent in sents:
            if word in sent:
                freq_dist[word] += 1

    return freq_dist

def get_document_freq(target, docs):
    freq = 0
    for doc in docs:
        if target in doc:
            freq += 1

    return freq

def get_idfs_per_voc(vocabulary, htmls):
    idfs = dict()
    N = len(htmls)

    for word in vocabulary:
        df = get_document_freq(word, htmls)
        idfs[word] = 1 + math.log10( (N + 1) / (1 + df) )

    return idfs


def get_all_tf_idf(html_sents_target, vocabulary_target, htmls):
    
    tfs = get_term_frequencies(html_sents_target, vocabulary_target)
    idfs = get_idfs_per_voc(vocabulary_target, htmls)
    
    tf_idf = [ (word, tfs[word]*idfs[word] / len(vocabulary_target) ) for word in vocabulary_target]

    tf_idf.sort(key=lambda pair: pair[1], reverse=True)


    return tf_idf
        





htmls = openFiles()
normalizer_tags = HTMLNormalizerTagged()
normalizer_sents = HTMLNormalizerSents()

htmls_normalized_sents = [normalizer_sents.normalize_html(html) for html in htmls]

htmls_normalized_sents_tagged = [ [normalizer_tags.tag_words_with_tagger(sent) for sent in html] for html in htmls_normalized_sents ]

html_target = htmls_normalized_sents_tagged[0]
vocab = sorted(set(word for sent in html_target for word in sent))
all_htmls_tagged = [[word for sent in html for word in sent] for html in htmls_normalized_sents_tagged]



tf_idfs = get_all_tf_idf(html_target, vocab, all_htmls_tagged)

for i in range(10):
    print(tf_idfs[i])








