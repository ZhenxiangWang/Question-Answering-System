import csv
filename='question_type.csv'

csv_file = open('question_type2.csv', 'w',encoding='utf-8',newline='')
writer = csv.writer(csv_file)
writer.writerow(["Question", "Ner"])

with open(filename,encoding='ISO-8859-1') as f:
    reader = csv.reader(f)
    for row in reader:
        question_content=row[0]
        question_type=row[1]
        if question_type=="ORG" and "who" in question_content.lower():
            # print(question_content, question_type)
            print(question_content,"PERSON")
            question_type="PERSON"
        elif question_type=="ORG" and "where" in question_content.lower():
            question_type = "LOC"
            print(question_content, "LOC")
            # print(question_content, question_type)
        elif question_type=="CARDINAL" or question_type=="QUANTITY":
            question_type = "NUMBER"
        elif question_type == "FAC"or question_type=="LOC":
            question_type = "LOC"
        elif question_type == "TIME"or question_type=="DATE":
            question_type = "TIME"
        elif question_type=="NORP" or question_type=="WORK_OF_ART":
            question_type = "OTHER"
        elif question_type=="ORDINAL" or question_type=="PRODUCT" or question_type=="LAW":
            continue
        writer.writerow([question_content, question_type])

csv_file.close()




