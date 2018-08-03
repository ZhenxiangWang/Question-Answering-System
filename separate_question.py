import csv
input_file='question_type2.csv'

EVENT = open('EVENT.TXT', 'w',encoding='utf-8')
GPE= open('GPE.TXT', 'w',encoding='utf-8')
LANGUAGE=open('LANGUAGE.TXT', 'w',encoding='utf-8')
LOC=open('LOC.TXT', 'w',encoding='utf-8')
MONEY=open('MONEY.TXT', 'w',encoding='utf-8')
NUMBER=open('NUMBER.TXT', 'w',encoding='utf-8')
ORG=open('ORG.TXT', 'w',encoding='utf-8')
OTHER=open('OTHER.TXT', 'w',encoding='utf-8')
PERCENT=open('PERCENT.TXT', 'w',encoding='utf-8')
PERSON=open('PERSON.TXT', 'w',encoding='utf-8')
TIME=open('TIME.TXT', 'w',encoding='utf-8')
# WORK_OF_ART=open('WORK_OF_ART.TXT', 'w',encoding='utf-8')

with open(input_file,encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    for row in reader:
        question_content=row[0]
        question_type=row[1]
        if question_type=="EVENT":
            EVENT.write(question_content+'\n')
        elif question_type=="GPE":
            GPE.write(question_content+'\n')
        elif question_type=="LANGUAGE":
            LANGUAGE.write(question_content+'\n')
        elif question_type=="LOC":
            LOC.write(question_content+'\n')
        elif question_type=="MONEY":
            MONEY.write(question_content+'\n')
        elif question_type=="NUMBER":
            NUMBER.write(question_content+'\n')
        elif question_type=="ORG":
            ORG.write(question_content+'\n')
        elif question_type=="OTHER":
            OTHER.write(question_content+'\n')
        elif question_type=="PERCENT":
            PERCENT.write(question_content+'\n')
        elif question_type=="PERSON":
            PERSON.write(question_content+'\n')
        elif question_type=="TIME":
            TIME.write(question_content+'\n')
        # elif question_type=="WORK_OF_ART":
        #     WORK_OF_ART.write(question_content+'\n')
EVENT.close()
GPE.close()
LANGUAGE.close()
LOC.close()
MONEY.close()
NUMBER.close()
ORG.close()
OTHER.close()
PERCENT.close()
PERSON.close()
TIME.close()
# WORK_OF_ART.close()





