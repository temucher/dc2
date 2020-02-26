import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


data = pd.read_csv('spam 2.csv', header=None, delimiter=',,,', engine='python') # names=['text', 'label'],
data = data.drop([0])
data[0] = data[0].replace('spam.', 'spam,')
data[0] = data[0].replace('ham.', 'ham,')

labels = []
texts = []
def splitter(x):
    spl_str = x.split(',')

    label = spl_str[0]
    labels.append(label)

    new_text = "".join(spl_str[1:len(spl_str)])
    texts.append(new_text)


df = [splitter(x) for x in data[0]]

result = pd.DataFrame(texts, columns=['text'])
label_df = pd.DataFrame(labels, columns=['label'])
result = result.join(label_df)

print(result.head())

tfidf_vectorizer = TfidfVectorizer(stop_words='english')
bow_vectorizer = CountVectorizer(stop_words='english') # unigrams
bigram_vectorizer = CountVectorizer(ngram_range=(2,2), stop_words='english') # bigrams
model = MultinomialNB()

X = result['text']
y = result['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# feature extraction
tfidf_train_feats = tfidf_vectorizer.fit_transform(X_train)
tfidf_test_feats = tfidf_vectorizer.transform(X_test)

bow_train_feats = bow_vectorizer.fit_transform(X_train)
bow_test_feats = bow_vectorizer.transform(X_test)

bigram_train_feats = bigram_vectorizer.fit_transform(X_train)
bigram_test_feats = bigram_vectorizer.transform(X_test)

# testing tfidf
print('TESTING TFIDF\n')
print('======================')
model.fit(tfidf_train_feats, y_train)
tfidf_pred = model.predict(tfidf_test_feats)
print('Accuracy score: ', accuracy_score(y_test, tfidf_pred))
print('Precision score: ', precision_score(y_test, tfidf_pred, pos_label='spam'))
print('Recall score: ', recall_score(y_test, tfidf_pred, pos_label='spam'))
print('F-1 score: ', f1_score(y_test, tfidf_pred, pos_label='spam'))
print(classification_report(y_test, tfidf_pred))

# testing bag of words
print('TESTING BAG OF WORDS\n')
print('======================')
model.fit(bow_train_feats, y_train)
bow_pred = model.predict(bow_test_feats)
print('Accuracy score: ', accuracy_score(y_test, bow_pred))
print('Precision score: ', precision_score(y_test, bow_pred, pos_label='spam'))
print('Recall score: ', recall_score(y_test, bow_pred, pos_label='spam'))
print('F-1 score: ', f1_score(y_test, bow_pred, pos_label='spam'))
print(classification_report(y_test, bow_pred))

# testing bigrams
print('TESTING BIGRAMS\n')
print('======================')
model.fit(bigram_train_feats, y_train)
bigram_pred = model.predict(bigram_test_feats)
print('Accuracy score: ', accuracy_score(y_test, bigram_pred))
print('Precision score: ', precision_score(y_test, bigram_pred, pos_label='spam'))
print('Recall score: ', recall_score(y_test, bigram_pred, pos_label='spam'))
print('F-1 score: ', f1_score(y_test, bigram_pred, pos_label='spam'))
print(classification_report(y_test, bigram_pred))
