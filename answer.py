import json
import numpy as np
import nltk
import re
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import linear_kernel
from nltk.corpus import stopwords
import spacy
from lstm_classifier import predict

nlp = spacy.load('en_core_web_sm')

stemmer = nltk.stem.PorterStemmer()

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
stopwords = set(nltk.corpus.stopwords.words('english'))
test=False

def load_data(file):
    with open(file, 'r') as f: data = json.load(f)
    return data

document_ner_json=load_data('sentence_level_document_ner.json')
document_json=load_data("documents.json")
train_json=load_data("training.json")
dev_json=load_data("devel.json")
test_json=load_data("testing.json")

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
    return lemma

def get_terms(sentence):
    sentence = re.sub(r'[^a-zA-Z0-9\-%]+', ' ', sentence)
    return [lemmatize(stemmer.stem(token.lower())) for token in word_tokenize(sentence.strip())]

def get_doc_sent_word_dict(documents):
    doc_sent_word_dict = {}
    doc_sent_dict = {}
    for document in documents:
        docid=document['docid']
        sentence_list=[]
        for para in document['text']:
            sentence_list+=sent_tokenize(para)
            doc_sent_dict[docid] = sentence_list
        doc_sent_word_dict[docid] = [get_terms(sentence) for sentence in sentence_list ]
    return doc_sent_word_dict,doc_sent_dict

doc_sent_word_dict,doc_sent_dict=get_doc_sent_word_dict(document_json)
# print(doc_sent_dict)
# doc_sent_word_dict is a dictionary of list of list {0:[['first','recogn','in','1900'


# def find_topk_sent(query, docid, doc_sent_word_dict,k):
#     results=[]
#     query_set=set(query)
#     sentences=doc_sent_word_dict[docid]
#     for sentid in range(len(sentences)):
#         sentence=sentences[sentid]
#         sentence_set=set(sentence)
#         jaccard=len(query_set&sentence_set)*1.0#/len(query_set|sentence_set)
#         results.append((sentid,jaccard))
#     results=sorted(results, key=lambda x: x[1], reverse=True)
#     return results[:k]

def find_topk_sent(query, docid, doc_sent_dict,k):
    sent_list=doc_sent_dict[docid]
    vectorizer = CountVectorizer(stop_words=stopwords)
    transformer = TfidfTransformer()
    train = vectorizer.fit_transform(sent_list).toarray()
    query = vectorizer.transform(query).toarray()
    transformer.fit(train)
    train_tfidf = transformer.transform(train)
    transformer.fit(query)
    query_tfidf = transformer.transform(query)
    cosine_similarities = linear_kernel(query_tfidf, train_tfidf).flatten()
    related_sents_indices = cosine_similarities.argsort()[:-k-1:-1]
    return related_sents_indices

questionsList = []
for train_data in test_json:
    question = train_data['question']
    questionsList.append(question)
pre_labels=predict(questionsList)

# pre_labels = []
# for train_data in test_json:
#     question = train_data['question']
#     pre_labels.append(predict(question))

csv_file = open('test_answer.csv', 'w',encoding='utf-8',newline='')
writer = csv.writer(csv_file)
writer.writerow(['id', 'answer'])

for i in range(len(test_json)):
    print(i)
    answer=""
    train_data = test_json[i]
    query = ''
    for token in get_terms(train_data['question']):
        query += token
        query += ' '
    query = query[:-1]
    # query = [term for term in get_terms(question['question']) if term not in stopwords]
    docid = train_data['docid']
    # results = rsearch(query, docid, inverted_index)
    topk_sent=find_topk_sent([query], docid, doc_sent_dict,3)
    find = False
    for sentid in topk_sent:
        if len(document_ner_json[str(docid)][str(sentid)])==0:
            continue
        for word, ner in document_ner_json[str(docid)][str(sentid)] :
            if ner==pre_labels[i]:# and preTags[i]!="OTHER":
                answer=word
                if ner=="TIME":
                    answer = re.sub(r',', ' ,', answer)
                if ner=="PERCENT":
                    answer = re.sub(r'%', ' %', answer)
                # answer = re.sub(r'"', '', answer)
                find=True
                break
        if find==True:
            writer.writerow([i, answer.lower()])
            break
        elif sentid==topk_sent[-1]:
            doc=nlp(doc_sent_dict[docid][topk_sent[0]])
            answer=""
            for np in doc.noun_chunks:
                answer+=str(np)
                answer+=" "
            writer.writerow([i, answer[:-1]])
            find=True
            break
    if find==False:
        writer.writerow([i, ""])

csv_file.close()









