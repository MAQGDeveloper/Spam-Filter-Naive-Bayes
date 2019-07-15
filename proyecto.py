import glob
import os
emails,labels = [],[]
#SPAM
file_path = 'enron1/spam/'
for filename in glob.glob(os.path.join(file_path, '*.txt')):
    with open(filename,'r',encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)
#NO SPAM
file_path = 'enron1/ham/'
for filename in glob.glob(os.path.join(file_path,'*.txt')):
    with open(filename,'r', encoding = "ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)
print(len(emails))
print(len(labels))

#import nltk
#nltk.download('names')
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
def letters_only(astr):
    return astr.isalpha()
all_names = set(names.words())    
lemmatizer = WordNetLemmatizer()

def clean_text(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(
             ' '.join([lemmatizer.lemmatize(word.lower())
                for word in doc.split()
                    if letters_only(word)
                        and word not in all_names]))
    return cleaned_docs

cleaned_emails = clean_text(emails)
print(cleaned_emails[0])

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(stop_words="english",max_features=500)
term_docs = cv.fit_transform(cleaned_emails)
print(term_docs [0])

feature_names = cv.get_feature_names()
print(feature_names[481])
#prints: u'web'
print(feature_names[357])
#prints: u'receive'
print(feature_names[125])
#prints: u'error'

feature_mapping = cv.vocabulary_

def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index
label_index = get_label_index(labels)

def get_prior(label_index):
     prior = {label: len(index) for label, index in label_index.items()}
     total_count = sum(prior.values())
     for label in prior:
        prior[label] /= float(total_count)
     return prior
prior = get_prior(label_index)
print(prior)
import numpy as np
def get_likelihood(term_document_matrix,label_index,smoothing = 0):
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_document_matrix[index,:].sum(axis=0)+smoothing
        likelihood[label] = np.asarray(likelihood[label])[0]
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label]/float(total_count)
    return likelihood
smoothing = 1
likelihood = get_likelihood(term_docs,label_index,smoothing)
print(likelihood[0])
print(likelihood[0][:5])
print(likelihood[1][:5])
print(feature_names[:5])

def get_posterior(term_document_matrix,prior,likelihood):
    num_docs = term_document_matrix.shape[0]
    posteriors = []
    for i in range(num_docs):
        posterior = {key: np.log(prior_label)
            for key,prior_label in prior.items()}
        for label,likelihood_label in likelihood.items():
            term_document_vector = term_document_matrix.getrow(i)
            counts = term_document_vector.data
            indices = term_document_vector.indices
            for count, index in zip(counts,indices):
                posterior[label] += np.log(likelihood_label[index]) * count
        min_log_posterior = min(posterior.values())
        for label in posterior:
            try:
                posterior[label]=np.exp(posterior[label]-min_log_posterior)
            except:
                posterior[label] = float('inf')
        sum_posterior = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= sum_posterior
        posteriors.append(posterior.copy())
    return posteriors

emails_test = ['''Subject: flat screens
hello ,
please call or contact regarding the other flat screens
requested . 
trisha tlapek - eb 3132 b
michael sergeev - eb 3132 a
also the sun blocker that was taken away from eb 3131 a .
trisha should two monitors also michael .
thanks
kevin moore''',
'''Subject: having problems in bed ? we can help !
cialis allows men to enjoy a fully normal sex life without
having to plan the sexual act .
if we let things terrify us, life will not be worth living
brevity is the soul of lingerie .
suspicion always haunts the guilty mind .''',]

cleaned_test = clean_text(emails_test)
term_docs_test = cv.transform(cleaned_test)
posterior = get_posterior(term_docs_test,prior,likelihood)
print(posterior)
#Prints: [{0: 0.99546887544929274, 1: 0.0045311245507072767}, {0: 0.00036156051848121361, 1: 0.99963843948151876}] 

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails,labels,test_size=0.33, random_state=42)

print(len(X_train),len(Y_train))
print(len(X_test),len(Y_test))

term_docs_train = cv.fit_transform(X_train)
label_index = get_label_index(Y_train)
prior = get_prior(label_index)
likelihood = get_likelihood(term_docs_train,label_index,smoothing)

term_docs_test = cv.transform(X_test)
posterior = get_posterior(term_docs_test,prior,likelihood)
print(posterior)
correct = 0.0
for pred, actual in zip(posterior,Y_test):
    if actual == 1:
        if pred[1] >= 0.5:
            correct +=1
    elif pred[0] > 0.5:
        correct += 1
print('La exactitud en {0} muestras de prueba es: {1:.1f}%'.format(len(Y_test),correct/len(Y_test)*100))
#prints: The accuracy on 1707 testing samples is: 92.0% 

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB(alpha=1.0,fit_prior=True)
clf.fit(term_docs_train,Y_train)
prediction_prob = clf.predict_proba(term_docs_test)
#prediction_prob[0:10]
print(prediction_prob[0:10])
#prints: array([[  1.00000000e+00,   2.12716600e-10], [  1.00000000e+00,  2.72887131e-75], [  6.34671963e-01,   3.65328037e-01], [  1.00000000e+00,  1.67181666e-12], [  1.00000000e+00,   4.15341124e-12], [  1.37860327e-04,  9.99862140e-01], [  0.00000000e+00,   1.00000000e+00], [  1.00000000e+00,  1.07066506e-18], [  1.00000000e+00,   2.02235745e-13], [  3.03193335e-01,  6.96806665e-01]])
prediction = clf.predict(term_docs_test)
#prediction[:10]
print(prediction[:10])
#prints: array([0, 0, 0, 0, 0, 1, 1, 0, 0, 1])

accuracy = clf.score(term_docs_test,Y_test)
print('La precisión con MultinomialNB es: {0:.1f}%'.format(accuracy*100))
#prints: The accuracy using MultinomialNB is: 92.0%

from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test,prediction,labels=[0,1])
#prints: array([[1098,   93],       [  43,  473]])

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(Y_test,prediction,pos_label=1)
#prints: 0.83568904593639581
recall_score(Y_test,prediction,pos_label=1)
#prints: 0.91666666666666663
f1_score(Y_test,prediction,pos_label=1)
#prints: 0.87430683918669128
f1_score(Y_test,prediction,pos_label=0)
#prints: 0.94168096054888506

from sklearn.metrics import classification_report
report = classification_report(Y_test,prediction)
print(report)

pos_prob = prediction_prob[:, 1]
thresholds = np.arange(0.0, 1.2, 0.1)
true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
for pred, y in zip(pos_prob, Y_test):
    for i, threshold in enumerate(thresholds):
        if pred >= threshold:
            if y == 1:
                true_pos[i] += 1
            else:
                false_pos[i] +=1
        else:
            break
true_pos_rate = [tp / 516.0 for tp in true_pos]
false_pos_rate = [fp / 1191.0 for fp in false_pos]

import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(false_pos_rate,true_pos_rate,color='darkorange',lw=lw)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('Tasa positiva falsa')
plt.ylabel('Tasa positiva verdadera')
plt.title('Característica de funcionamiento del receptor')
plt.legend(loc="lower right")
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(Y_test, pos_prob)
#prints: 0.95828777198497783

from sklearn.model_selection import StratifiedKFold
k = 10
k_fold = StratifiedKFold(n_splits=k)
cleaned_emails_np = np.array(cleaned_emails)
labels_np = np.array(labels)

max_features_option = [2000,4000,8000]
smoothing_factor_option = [0.5,1.0,1.5,2.0]
fit_prior_option = [True,False]
auc_record = {}

for train_indices, test_indices in k_fold.split(cleaned_emails,labels):
    X_train,X_test = cleaned_emails_np[train_indices],cleaned_emails_np[test_indices]
    Y_train,Y_test = labels_np[train_indices],labels_np[test_indices]
    for max_features in max_features_option:
        if max_features not in auc_record:
            auc_record[max_features] = {}
        cv = CountVectorizer(stop_words="english",max_features=max_features)
        term_docs_train = cv.fit_transform(X_train)
        term_docs_test = cv.transform(X_test)
        for smoothing in smoothing_factor_option:
            if smoothing not in auc_record[max_features]:
                auc_record[max_features][smoothing] = {}
            for fit_prior in fit_prior_option:
                clf = MultinomialNB(alpha=smoothing,fit_prior=fit_prior)
                clf.fit(term_docs_train,Y_train)
                prediction_prob=clf.predict_proba(term_docs_test)
                pos_prob = prediction_prob[:,1]
                auc = roc_auc_score(Y_test,pos_prob)
                auc_record[max_features][smoothing][fit_prior] = auc+auc_record[max_features][smoothing].get(fit_prior,0.0)

print('Maximas funciones de suavizado de ajuste prior auc'.format(max_features, smoothing, fit_prior, auc/k))
for max_features,max_features_record in auc_record.items():
    for smoothing,smoothing_record in max_features_record.items():
        for fit_prior, auc in smoothing_record.items():
            print('       {0}      {1}      {2}    {3:.4f}'.format(max_features, smoothing, fit_prior, auc/k))