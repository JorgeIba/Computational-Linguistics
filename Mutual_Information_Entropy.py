from Util import openFiles, HTMLNormalizerSents
import math


def sents_with_wordW(wordW, sents_tokenized):
    cntSents = 0
    for sent in sents_tokenized:
        if wordW in sent:
            cntSents += 1
    return cntSents

def sents_with_wordW_and_wordV(wordW, wordV, sents_tokenized):
    cntSents = 0
    for sent in sents_tokenized:
        if wordW in sent and wordV in sent:
            cntSents += 1
    return cntSents

def get_mutual_information(sents_tokenized, vocabulary, wordW):
    totalCntSents = len(sents_tokenized)
    p_w1 = (sents_with_wordW(wordW, sents_tokenized) + 0.5) / ( totalCntSents + 1 )
    p_w0 = 1 - p_w1

    words_and_mutual_information = []
    for wordV in vocabulary:
        p_v1 = (sents_with_wordW(wordV, sents_tokenized) + 0.5) / ( totalCntSents + 1 )
        p_v0 = 1 - p_v1

        p_w1_v1 = (sents_with_wordW_and_wordV(wordW, wordV, sents_tokenized) + 0.25) / (totalCntSents + 1)
        p_w1_v0 = p_w1 - p_w1_v1

        p_w0_v1 = p_v1 - p_w1_v1
        p_w0_v0 = p_w0 - p_w0_v1

        first_term =  p_w0_v0 * math.log2( p_w0_v0 / ( p_w0 * p_v0) )
        second_term = p_w0_v1 * math.log2( p_w0_v1 / ( p_w0 * p_v1 ) )
        third_term =  p_w1_v0 * math.log2( p_w1_v0 / ( p_w1 * p_v0) ) 
        fourth_term = p_w1_v1 * math.log2( p_w1_v1 / ( p_w1 * p_v1) ) 

        mutual_info = first_term + second_term + third_term + fourth_term
        
        words_and_mutual_information.append( (wordV, mutual_info) )
    

    words_and_mutual_information.sort(key=lambda tupl: tupl[1], reverse=True)

    return words_and_mutual_information







all_htmls = openFiles()
normalizer = HTMLNormalizerSents()


sents_tokenized = []
#for html in all_htmls:
for i in range(1):
    html = all_htmls[i]
    sents_tokenized += normalizer.normalize_html(html)


vocabulary = sorted(set(word for sent in sents_tokenized for word in sent))

mutual_information = get_mutual_information(sents_tokenized, vocabulary, "méxico")

print(mutual_information[:10])