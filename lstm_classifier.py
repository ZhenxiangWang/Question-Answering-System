from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
import pandas as pd
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import numpy as np

eng_stopwords = set(stopwords.words("english"))
lem = WordNetLemmatizer()
tokenizer=TweetTokenizer()
labels = ["EVENT", "GPE", "LANGUAGE", "LOC", "MONEY", "NUMBER", "ORG", "OTHER", "PERCENT", "PERSON", "TIME"]
def clean(question):
    question = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", question)
    question=question.lower()
    question = re.sub(r",", " , ", question)
    question = re.sub(r"!", " ! ", question)
    question = re.sub(r"\(", " \( ", question)
    question = re.sub(r"\)", " \) ", question)
    question = re.sub(r"\?", " \? ", question)
    question = re.sub(r"\s{2,}", " ", question)
    question = re.sub(r"\'s", " \'s", question)
    question = re.sub(r"\'ve", " \'ve", question)
    question = re.sub(r"n\'t", " n\'t", question)
    question = re.sub(r"\'re", " \'re", question)
    question = re.sub(r"\'d", " \'d", question)
    question = re.sub(r"\'ll", " \'ll", question)
    words = tokenizer.tokenize(question)
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]
    clean_question=" ".join(words)
    clean_question=re.sub("\W+"," ",clean_question)
    clean_question=re.sub("  "," ",clean_question)
    return (clean_question)

# def train():

EVENT = list(open("EVENT.txt", "r").readlines())
EVENT = [s.strip() for s in EVENT]

GPE = list(open("GPE.txt", "r").readlines())
GPE = [s.strip() for s in GPE]

LANGUAGE = list(open("LANGUAGE.txt", "r").readlines())
LANGUAGE = [s.strip() for s in LANGUAGE]

LOC = list(open("LOC.txt", "r").readlines())
LOC = [s.strip() for s in LOC]

MONEY = list(open("MONEY.txt", "r").readlines())
MONEY = [s.strip() for s in MONEY]

NUMBER = list(open("NUMBER.txt", "r").readlines())
NUMBER = [s.strip() for s in NUMBER]

ORG = list(open("ORG.txt", "r").readlines())
ORG = [s.strip() for s in ORG]

OTHER = list(open("OTHER.txt", "r").readlines())
OTHER = [s.strip() for s in OTHER]

PERCENT = list(open("PERCENT.txt", "r").readlines())
PERCENT = [s.strip() for s in PERCENT]

PERSON = list(open("PERSON.txt", "r").readlines())
PERSON = [s.strip() for s in PERSON]

TIME = list(open("TIME.txt", "r").readlines())
TIME = [s.strip() for s in TIME]

examples = [EVENT, GPE, LANGUAGE, LOC, MONEY, NUMBER, ORG, OTHER, PERCENT, PERSON, TIME]
x_text = []
for example in examples:
    x_text += example
x_text = [clean(sent) for sent in x_text]
x_text = [s.split(" ") for s in x_text]

EVENT_labels = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in EVENT]  # 0
GPE_labels = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in GPE]  # 1
LANGUAGE_labels = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0 ] for _ in LANGUAGE]  # 2
LOC_labels = [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 ] for _ in LOC]  # 3
MONEY_labels = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 ] for _ in MONEY]  # 4
NUMBER_labels = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 ] for _ in NUMBER]  # 5
ORG_labels = [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 ] for _ in ORG]  # 6
OTHER_labels = [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 ] for _ in OTHER]  # 7
PERCENT_labels = [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 ] for _ in PERCENT]  # 8
PERSON_labels = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 ] for _ in PERSON]  # 9
TIME_labels = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] for _ in TIME]  # 10

y = np.concatenate([EVENT_labels, GPE_labels, LANGUAGE_labels, LOC_labels, MONEY_labels, NUMBER_labels, ORG_labels, OTHER_labels,
     PERCENT_labels, PERSON_labels, TIME_labels], 0)

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(x_text))
list_tokenized_train = tokenizer.texts_to_sequences(x_text)

maxlen = 100
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)

inp = Input(shape=(maxlen, ))

embed_size = 128
x = Embedding(max_features, embed_size)(inp)
x = LSTM(60, return_sequences=True,name='lstm_layer')(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(11, activation="softmax")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='softmax_crossentropy',optimizer='adam',metrics=['accuracy'])

batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

test = pd.read_csv('test.csv', encoding='latin-1')

def predict(questionList):
    list_sentences_test = []
    for example in questionList:
        list_sentences_test += example
    list_sentences_test = [clean(sent) for sent in list_sentences_test]
    list_sentences_test = [s.split(" ") for s in list_sentences_test]
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_test = pad_sequences(list_tokenized_test, maxlen=maxlen)
    y_pred = model.predict(X_test, batch_size=1024)
    pre_labels=[]
    for y in y_pred:
        pre_labels.append(labels[np.random.choice(range(y.shape[1]),p=y.ravel())])
    return pre_labels







