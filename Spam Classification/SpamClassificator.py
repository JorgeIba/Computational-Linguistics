from Util import HTMLNormalizer
import numpy as np

def open_samples(encoding='utf-8'):
    NAME = 'SMS_Spam_Corpus_big.txt'
    
    samples = []
    with open(NAME, mode='r+') as f:
        for line in f.readlines():
            samples.append(line)
    return samples


def classify_samples(samples):
    samples_simplified = []

    for sample in samples:
        splitted = sample.strip().split(',')

        category = splitted[-1]
        text = " ".join(splitted[:-1])

        samples_simplified.append((text, category))

    return samples_simplified


def get_matrix_X(text_samples, vocabulary):
    word2id = { word: idx+1 for idx, word in enumerate(vocabulary) }

    row_vectors = []

    for text in text_samples:
        vector = np.zeros(len(vocabulary)+1)
        
        for word in text:
            vector[ word2id[word] ] += 1

        vector[0] = 1

        row_vectors.append(vector)

    return np.matrix(row_vectors)


# Cost of Logistic Regresion
def J_THETA(THETA, Y):
    y_1 = np.dot( Y, np.log(THETA) )

    y_2 = np.dot( (1-Y), np.log(  1 - THETA  )  )

    return (-1 / len(Y)) * (y_1 + y_2)

def getMatchs(H, Y):
    PRED = np.round(H)

    matchs = 0
    for i in range(M):
        if PRED[i] == Y[i]:
            matchs += 1

    return matchs



# Open samples and classify them
samples = open_samples()
classified = classify_samples(samples)

normalizer = HTMLNormalizer(language='english')


# Normalize text and categories
classified_normalized = []
for text, category in classified:
    normalized_text = normalizer.normalize_html(text, is_html=False)
    is_spam = 1 if category == 'spam' else 0

    classified_normalized.append((normalized_text, is_spam))


vocabulary = sorted(set( word for text, _ in classified_normalized for word in text ))

M, N = len(samples), len(vocabulary)


text_values = [text for text, _ in classified_normalized]

Y = np.matrix([y for _, y in classified_normalized]).T
X = get_matrix_X(text_values, vocabulary)



sigmoid = lambda x: 1 / (1 + np.exp(-x))


# INIT
THETA = np.random.random((N+1, 1))
Z = X.dot(THETA)
H = sigmoid(Z)


print("INITIAL: ")
print(J_THETA( np.ravel(H), np.ravel(Y) ))


print(f"Matchs: {getMatchs(H, Y)} / {M} \n")


LEARNING_RATE = 5
for it in range(100):

    S = (LEARNING_RATE / M) * ( np.dot((H-Y).T, X) )
    THETA = THETA - S.T
    Z = X.dot(THETA)
    H = sigmoid(Z)

    if it % 20 == 0:
        print(J_THETA( np.ravel(H), np.ravel(Y) ))
    



print("\nFINAL: ")
print(J_THETA( np.ravel(H), np.ravel(Y) ))


print(f"Matchs: {getMatchs(H, Y)} / {M} \n")







