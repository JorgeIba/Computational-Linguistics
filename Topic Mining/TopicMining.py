from Util import HTMLNormalizer
import gensim
import os

def openArticles(encoding='utf-8'):

    NAME_FOLDER = 'Articles'
    articles = []
    for filename in os.listdir(NAME_FOLDER):
        dir_file = os.path.join(NAME_FOLDER, filename)

        with open(dir_file, mode='r', encoding=encoding) as f:
            article = f.read()
            articles.append(article)

    return articles


articles = openArticles()
normalizer = HTMLNormalizer()

texts = []
for article in articles:
    texts.append(normalizer.normalize_html(article, is_html=False))



dictionary = gensim.corpora.Dictionary(texts)

doc_term_matrix = [dictionary.doc2bow(t) for t in texts]

Lda = gensim.models.ldamodel.LdaModel


NUM_TOPICS = 4


ldaModel = Lda(doc_term_matrix, num_topics=NUM_TOPICS, id2word= dictionary, passes=50)

topics = ldaModel.print_topics(num_topics=NUM_TOPICS, num_words=10)

for i in range(4):
    print(topics[i])




