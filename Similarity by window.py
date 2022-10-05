import numpy as np
from Util import openFiles, HTMLNormalizer, getSimilars

def getAllContextsByWindow(text_tokenized, window = 3):
    word_to_context = {}

    for idx, word in enumerate(text_tokenized):

        if word not in word_to_context:
            word_to_context[word] = []
        
        # Get context by window
        idx_start = max(0, idx - window)
        idx_end = min(len(text_tokenized) - 1, idx + window)

        left_context = text_tokenized[idx_start : idx]
        right_context = text_tokenized[(idx+1) : (idx_end+1)]

        word_to_context[word] += (left_context + right_context)

    return word_to_context
    



all_htmls = openFiles()
normalizer = HTMLNormalizer()

# print(all_htmls[0][-350:])

normalized = []
for html in all_htmls:
    normalized += normalizer.normalize_html(html)


contexts = getAllContextsByWindow(normalized, 4)

vocabulary = sorted(set(normalized))
similars = getSimilars(vocabulary, contexts, 'banco')


for i in range(20):
    print(similars[i])
