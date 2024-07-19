
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import svm
from sklearn.metrics import classification_report

import pandas as pd
import pickle

# train Data
trainData = pd.read_csv("data_train.csv")

# test Data
testData = pd.read_csv("data_test.csv")

# Create feature vectors
vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)

train_vectors = vectorizer.fit_transform(trainData['text'])
test_vectors = vectorizer.transform(testData['text'])

# Perform classification with SVM, kernel=linear
classifier_linear = svm.SVC(kernel='linear')
#print(train_vectors,trainData['tag'])
classifier_linear.fit(train_vectors, trainData['tag'])
prediction_linear = classifier_linear.predict(test_vectors)
pickle.dump(vectorizer, open('models/vectorizer.sav', 'wb'))
pickle.dump(classifier_linear, open('models/classifier.sav', 'wb'))

#print('Both vectorizer and classifier has been pickled. Check "classifier_flask" to load and use in flask app')

# results
print("Results for SVM(kernel=linear)")
report = classification_report(testData['tag'], prediction_linear, output_dict=True)
print("ehsas e biedalati :( : \n ",report['1'])
print("ehsas e edalat :):\n ",report['-1'])
print("neutral:\n ",report['0'])
print("weighted avg: \n ",report['weighted avg'])