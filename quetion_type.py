import spacy
import json
import csv
import re

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

nlp = spacy.load('en_core_web_sm')
devel_json = load_json('devel.json')
training_json=load_json('training.json')

documents_json=load_json('documents.json')

csv_file = open('question_type.csv', 'w',encoding='utf-8',newline='')
writer = csv.writer(csv_file)
writer.writerow(["Question", "Ner"])

def find_start_end_index_of_answer(lower_relative_paragraph):
    start_index=[]
    for index,paragraph_char in list(enumerate(lower_relative_paragraph)):
        if paragraph_char==answer[0]:
            start_index.append(index)
    for index in start_index:
        start=index
        for i in range(len(answer)):
            end=start+i
            # print(answer[i])
            try:
                if lower_relative_paragraph[end]!=answer[i]:
                    start=None
                    end=None
                    break
                if i==len(answer)-1:
                    return start,end
            except:
                return None,None
    return None, None

j=0
for question in devel_json:
    j+=1
    print(j)
    question_content=question['question']
    answer=question['text']
    # print("answer",answer)
    p = re.compile('[\s\t]+,')
    answer = re.sub(p, ',', answer)
    q=re.compile('[\s\t]+%')
    answer = re.sub(q, '%', answer)
    # print("replaced_answer",answer)
    answer_paragraph=question['answer_paragraph']
    docid=question['docid']
    relative_document=documents_json[docid]["text"]
    relative_paragraph=relative_document[answer_paragraph]
    lower_relative_paragraph = relative_paragraph.lower()

    start,end=find_start_end_index_of_answer(lower_relative_paragraph)
    if start!=None and end!=None:
        true_answer=relative_paragraph[start:end+1]
        # print("true_answer", true_answer)
        doc = nlp(true_answer)
        answer_ner = "OTHER"
        for i in range(len(doc)):
            if len(doc[i].ent_type_)!=0:
                answer_ner=doc[i].ent_type_
                print(answer_ner)
                break
        writer.writerow([question_content, answer_ner])

for question in training_json:
    j += 1
    print(j)
    question_content=question['question']
    answer=question['text']
    # print("answer",answer)
    p = re.compile('[\s\t]+,')
    answer = re.sub(p, ',', answer)
    q=re.compile('[\s\t]+%')
    answer = re.sub(q, '%', answer)
    # print("replaced_answer",answer)
    answer_paragraph=question['answer_paragraph']
    docid=question['docid']
    relative_document=documents_json[docid]["text"]
    relative_paragraph=relative_document[answer_paragraph]
    lower_relative_paragraph = relative_paragraph.lower()

    start,end=find_start_end_index_of_answer(lower_relative_paragraph)
    if start!=None and end!=None:
        true_answer=relative_paragraph[start:end+1]
        # print("true_answer", true_answer)
        doc = nlp(true_answer)
        answer_ner = "OTHER"
        for i in range(len(doc)):
            if len(doc[i].ent_type_)!=0:
                answer_ner=doc[i].ent_type_
                print(answer_ner)
                break
        writer.writerow([question_content, answer_ner])

csv_file.close()













