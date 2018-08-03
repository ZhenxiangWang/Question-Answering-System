1. document_ner.py will create a document with each word attached by a NER tag.
2. question_type.py will read the devel.json and training.json and then label each question with its answer's NER tag. 
3. combine_question_type.py will combine different question types into 11 types.
4. separate_question.py will separate question_type file into 11 subset files.
5. lstm_classifier.py is the question classifier model.
6. answer.py will generate the answer of the testing data.