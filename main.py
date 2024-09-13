import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from keras.preprocessing.text import text_to_word_sequence
import collections
from sklearn.model_selection import train_test_split


spam_ham = pd.DataFrame( columns=['S/H', 'filename','subject','body'])


for filename in os.listdir('../input/spam-2/email archive'):
    f = open("../input/spam-2/email archive/" + filename, "r")
    message= f.read()
    lst = message.split("\n\n")
    if filename[0:2] == 'sp':
        SH = 1
    else:
        SH = 0
    
    
    clean_text = lst[1].casefold()
    clean_text = " ".join([w for w in clean_text.split() if w.isalpha()]) # Side effect: removes extra spaces
    stopwords = ["is", "a"]

    tokens = clean_text.split()
    clean_tokens = [t for t in tokens if not t in stopwords]
    clean_text = " ".join(clean_tokens)
    tokens = text_to_word_sequence(clean_text)
    
    spam_ham = spam_ham.append({'S/H': SH,'filename':filename,'subject':lst[0],'body':clean_text},ignore_index=True)
    

spam_ham.to_csv('spam_ham.csv')
data = pd.read_csv('spam_ham.csv')
data1 = data.to_numpy()
data = data.drop('Unnamed: 0', axis = 1)


ham_words2={}
ham_words3={}
spam_words2={}
spam_words3={}
data2 = data1

spam_words =""
ham_words =""


for x in data2:
    if x[1] ==0:
        for i in text_to_word_sequence(x[4]):
            ham_words = ham_words + i
        for w in text_to_word_sequence(x[4]):
            wfreq=x[4].count(w)
            ham_words2[w]= wfreq
        for x in ham_words2:
            try:
                ham_words3[x] = ham_words3[x] + ham_words2[x] 
            except:
                ham_words3[x] =ham_words2[x]
    else:
        for i in text_to_word_sequence(x[4]):
            spam_words = spam_words + i
        for w in text_to_word_sequence(x[4]):
            wfreq=x[4].count(w)
            spam_words2[w]= wfreq
        for x in spam_words2:
            try:
                spam_words3[x] = spam_words3[x] + spam_words2[x] 
            except:
                spam_words3[x] =spam_words2[x]



from wordcloud import WordCloud
import matplotlib.pyplot as plt

#spam_words graph
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(spam_words)
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()




#Ham words graph
word_cloud = WordCloud(collocations = False, background_color = 'white').generate(ham_words)


plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


from sklearn.feature_extraction.text import CountVectorizer


cv = CountVectorizer()

body2 =[]
spam_ham =[]
for x in data2:
    body2.append(x[4])
    spam_ham.append(x[1])

#vector = cv.fit_transform(data['body'])
vector = cv.fit_transform(body2)

vector_df = pd.DataFrame(vector.toarray())

vector_df.columns = cv.get_feature_names()

vector_df["S/H"] = spam_ham
vector_df






from keras.models import Sequential
from keras.layers import Dense, Activation, Input
import numpy as np

model = Sequential()
model.add(Input(len(vector_df.columns)-1))
model.add(Dense(100, activation='relu'))# relu proceess the data but reduces training time of this large netowork
model.add(Dense(1, activation='sigmoid')) #accuratley figures out the importance of the word.

# Compile the model and calculate its accuracy:
model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']) 

# Print a summary of the Keras model:
model.summary()










train, test = train_test_split(vector_df, test_size=0.3)

X_train = train.iloc[:,:-1].values
y_train = train.iloc[:, -1:].values

X_test = test.iloc[:, :-1].values
y_test = test.iloc[:, -1:].values






Nueral_network = model.fit(X_train, y_train, batch_size = 100, epochs =10, verbose =1) 



model.evaluate(X_test, y_test)


print(model.predict(X_test, batch_size=128, verbose=0))


def spam_ham(filenamepath):
    f = open(filenamepath, "r")
    message= f.read()
    lst = message.split("\n\n")
    
    clean_text = lst[1].casefold()
    clean_text = " ".join([w for w in clean_text.split() if w.isalpha()]) # Side effect: removes extra spaces
    stopwords = ["is", "a"]

    tokens = clean_text.split()
    clean_tokens = [t for t in tokens if not t in stopwords]
    clean_text = " ".join(clean_tokens)
    tokens = text_to_word_sequence(clean_text)
    file1 =[]
    for col in vector_df.columns[:-1]:
        if col in clean_text:
            file1.append(1)
        else:
            file1.append(0)
    return file1

Test = spam_ham("/kaggle/input/spam-2/email archive/3-1msg1.txt")
print(model.predict([Test], batch_size=128, verbose=0))
