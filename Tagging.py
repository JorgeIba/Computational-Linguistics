from Util import openFiles, HTMLNormalizer, getSimilars
import nltk



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
normalizer = HTMLNormalizerTagged()

# print(all_htmls[0][-350:])

normalized = []
for html in all_htmls:
    normalized += normalizer.normalize_html(html)

context = getAllContextsByWindow(normalized, 4)
vocab = sorted(set(normalized))

similars = getSimilars(vocab, context, ('méxico', 'n'))


for i in range(20):
    print(similars[i])



# nltk.UnigramTagger()

# nltk.download('cess_esp')
# train_sents = nltk.corpus.cess_esp

# print(train_sents.tagged_sents()[:20])
