from Util import HTMLNormalizer
import numpy as np
import random

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
    for i in range(len(H)):
        if PRED[i] == Y[i]:
            matchs += 1

    return matchs

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_model(X, Y, N, it = 500):
    M= len(X)

    # INIT
    THETA = np.random.random((N+1, 1))
    Z = X.dot(THETA)
    H = sigmoid(Z)


    print("INITIAL: ")
    print(J_THETA( np.ravel(H), np.ravel(Y) ))
    print(f"Matchs: {getMatchs(H, Y)} / {M} \n")


    LEARNING_RATE = 5
    for it in range(it):

        S = (LEARNING_RATE / M) * ( np.dot((H-Y).T, X) )
        THETA = THETA - S.T
        Z = X.dot(THETA)
        H = sigmoid(Z)

        if it % 20 == 0:
            print(J_THETA( np.ravel(H), np.ravel(Y) ))

    print("\nCOST AFTER TRAINING: ")
    print(J_THETA( np.ravel(H), np.ravel(Y) ))


    print(f"Matchs: {getMatchs(H, Y)} / {M} \n")

    return THETA        


def get_confusion_matrix(y_real, y_pred):
    conf_mat = np.zeros((2,2))

    for val_real, val_pred in zip(y_real, y_pred):
        conf_mat[val_real][val_pred] += 1

    return conf_mat



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




text_values = [text for text, _ in classified_normalized]



samples_train = 800
M, N = samples_train, len(vocabulary)


random.shuffle(classified_normalized)

Y_TRAIN = np.matrix([y for _, y in classified_normalized[:samples_train]]).T
Y_TEST  = np.matrix([y for _, y in classified_normalized[samples_train:]]).T
X_TRAIN = get_matrix_X([text for text, _ in classified_normalized[:samples_train]], vocabulary)
X_TEST  = get_matrix_X([text for text, _ in classified_normalized[samples_train:]], vocabulary)



THETA = train_model(X_TRAIN, Y_TRAIN, N, 500)


Z = X_TEST.dot(THETA)
Y_PRED = np.round(sigmoid(Z)).astype(int)


CONF_MAT = get_confusion_matrix(np.ravel(Y_TEST), np.ravel(Y_PRED))

print("TOTAL SAMPLES TEST: ", len(Y_TEST), "\n")
print("Confusion Matrix: ")
print(CONF_MAT, "\n")



# 0 is ham
# 1 is spam
TN, FN = CONF_MAT[0][0], CONF_MAT[0][1]
FP, TP = CONF_MAT[1][0], CONF_MAT[1][1]

specifity = TN / (TN + FP)
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
f_measure = precision * recall / (precision + recall)
fallout   = FP / (FP + TN)
accuracy  = (TP + TN) / (TP + TN + FP + FN)


print("Specifity: ", specifity)
print("Precision: ", precision)
print("Recall: "   , recall)
print("F Measure: ", f_measure)
print("Fallout: "  , fallout)
print("Accuracy: " , accuracy)




