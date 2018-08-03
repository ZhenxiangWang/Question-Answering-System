import json
import spacy
from nltk.tokenize import sent_tokenize

nlp = spacy.load('en_core_web_sm')
def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

documents_json=load_json('documents.json')

document_dict={}
for docid in range(len(documents_json)):
    print(docid)
    sentence_dict = {}
    sentid = 0
    for paragraph in documents_json[docid]["text"]:
        # print(paragraph)
        sentences=sent_tokenize(paragraph)
        for sent in sentences:
            # print(sentences[sentid])
            doc=nlp(sent)
            # print(doc)
            sent_list=[]
            for entity in doc.ents:
                word=entity.text
                ner=entity.label_
                if ner == "" or  ner == "NORP" or ner == "ORDINAL" or ner == "PRODUCT" or ner == "LAW":
                    ner = "OTHER"
                elif ner == "CARDINAL" or ner == "QUANTITY":
                    ner = "NUMBER"
                elif ner == "FAC" or ner == "LOC":
                    ner = "LOC"
                elif ner == "TIME" or ner == "DATE":
                    ner = "TIME"
                # print([word,ner])
                sent_list.append([word,ner])
            sentence_dict[sentid]=sent_list
            sentid += 1
    document_dict[docid]=sentence_dict

with open('sentence_level_document_ner.json', 'w') as f:
    json.dump(document_dict, f)

