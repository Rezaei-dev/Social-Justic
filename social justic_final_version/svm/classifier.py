import pickle
import os
from flask import Flask, jsonify, make_response, request, redirect
import csv
import pandas as pd

vectorizer = pickle.load(open('models/vectorizer.sav', 'rb'))
classifier = pickle.load(open('models/classifier.sav', 'rb'))
Data = pd.read_csv("data_.csv")
rr=[]
total=[]

with open('data_.csv',encoding="utf8") as csvfile:
        
        all_data=[]
        #writer_ = csv.writer(newfile,dialect='excel')
        all_data=[]
        readCSV = csv.reader(csvfile, delimiter=',')
        #writer.writeheader()
        count=0
        for row in readCSV:
            text=row[7]
            text_vector = vectorizer.transform([text])
            result = classifier.predict(text_vector)
            row.append(result[0])
            total.append(row)

count=0
with open('output.csv', 'w', newline='',encoding="utf8") as file:
    writer = csv.writer(file)
    writer.writerow(["id","peer_username","peer_participants_count","date","fwd_id","type","tag","views","text"])
    for row in total:
        #print(len(row))
        writer.writerow([row[0],row[1],row[2],row[3],row[4],row[5],row[8],row[6],row[7]])
        count+=1
