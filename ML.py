import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


df = pd.read_csv(r'C:\Users\sanji\DataspellProjects\ML-Learning\Twitter-conversation-sentiments.csv', encoding='latin1')
df = df.rename(columns={'0': 'Sentiments'})
df = df.rename(columns={(df.columns[1]): 'ID'})
df = df.rename(columns={(df.columns[2]): 'Date'})
df = df.rename(columns={(df.columns[3]): 'Flag'})
df = df.rename(columns={(df.columns[4]): 'User'})
df = df.rename(columns={(df.columns[5]): 'Message'})

data = df[['Sentiments', 'Message']]
X = data['Message']
y = data['Sentiments']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

tfidf = TfidfVectorizer(stop_words='english')
tfidf.fit(X_train)
X_train_tfidf = tfidf.transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf, y_train)

pipe = Pipeline([('tfidf', TfidfVectorizer()),
                 ('log_model', LogisticRegression())])
pipe.fit(X, y)

end_sequence = ''
print("To break the loop, type: end.")
while end_sequence != 'end':
    user_input = input('Please enter your string to be evaluated by the model: ')
    end_sequence = user_input
    result = pipe.predict([user_input])
    if result == 0:
        print("Negative")
    elif result == 4:
        print("Positive")
