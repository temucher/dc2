import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import string

lemmatizer = WordNetLemmatizer()
def nltk2wn_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None
def lemmatize_sentence(sentence):
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    wn_tagged = map(lambda x: (x[0], nltk2wn_tag(x[1])), nltk_tagged)
    res_words = []
    for word, tag in wn_tagged:
        if tag is None:            
            res_words.append(word)
        else:
            res_words.append(lemmatizer.lemmatize(word, tag))
    return " ".join(res_words)

#remaining stopwords after cutting off last 50~ words of "from nltk.corpus import stopwords" since they had sentiment value (don't, should've, etc)

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
    #remove punctuation
    nopunc = [char for char in new_text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    #lemmatize
    new_text = lemmatize_sentence(nopunc)
        
    #remove stop words
    wordList = new_text.split()
    goodWords = []
    for word in wordList:
        if word.lower() not in stopwords.words('english'):
            goodWords.append(word)
    texts.append(' '.join(goodWords))

#generate list of 'spam' or 'ham' labels and list of messages, used to create dataframe representing csv file
for x in data[0]:
    splitter(x) 

result = pd.DataFrame(texts, columns=['text'])
label_df = pd.DataFrame(labels, columns=['label'])
result = result.join(label_df)

print(result.head())

bow_vectorizer = CountVectorizer() # unigrams
bigram_vectorizer = CountVectorizer(ngram_range=(2,2)) # bigrams
model = MultinomialNB()

X = result['text']
y = result['label']
X_train, X_test, y_train, y_test = train_test_split(X, y)

# feature extraction
bow_train_feats = bow_vectorizer.fit_transform(X_train)
bow_test_feats = bow_vectorizer.transform(X_test)

bigram_train_feats = bigram_vectorizer.fit_transform(X_train)
bigram_test_feats = bigram_vectorizer.transform(X_test)

model.fit(bow_train_feats, y_train)
bow_pred = model.predict(bow_test_feats)

# testing backoff
print('TESTING BACKOFF\n')
print('======================')
model.fit(bigram_train_feats, y_train)
bigram_pred = model.predict(bigram_test_feats)
#find messages classified as 'ham', run BoW on them to see if they're actually 'spam'
ham_indices = [i for i, value in enumerate(bigram_pred) if value == 'ham']
#aka use the output in the BoW at those indices
for index in ham_indices:
    bigram_pred[index] = bow_pred[index]

print(classification_report(y_test, bigram_pred))
