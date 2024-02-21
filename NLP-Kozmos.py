
# Importing necessary libraries

# pip install nltk
# pip install textblob
# pip install wordcloud
# !pip install catboost
# !pip install lightgbm
# !pip install xgboost

from warnings import filterwarnings
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from textblob import Word
from wordcloud import WordCloud
import re

# Suppressing warnings
filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda x: "%2f" % x)

#++++++++++++++++++++++++ TEXT PREPROCESSING ++++++++++++++++++++++++++

df = pd.read_excel(r"C:\\datasets\\amazon.xlsx") # Reading the dataset

def clean_review(Review):
    Review = Review.str.lower() # convert to lowercase
    Review = Review.astype(str)
    Review = Review.apply(lambda x: re.sub(r"[^\w\s]", " ", x)) # Remove punctuation marks
    Review = Review.apply(lambda x: re.sub(r"[\d]", " ", x)) # Remove digits
    return Review
df["Review"] = clean_review(df["Review"])

def remove_stopword(Review): # Remove words that do not convey meaning in spoken language
    sw = stopwords.words("english")
    Review = Review.apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
    return Review
df["Review"] = remove_stopword(df["Review"])

def drop_words(Review): # Removing rare words
    temp_df = pd.Series(" ".join(df["Review"]).split()).value_counts()
    drops = temp_df[temp_df <= 100]
    Review = Review.apply(lambda x: " ".join(x for x in x.split() if x not in drops))
    return Review
df["Review"] = drop_words(df["Review"])

def lemmatization(Review):
    Review = Review.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    return Review
df["Review"] = lemmatization(df["Review"])

#+++++++++++++++++++++++ TEXT VISUALIZATION ++++++++++++++++++++++++++
def text_visulization(Review, Barplot=False, Wordcloud=False):
    tf = Review.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()
    tf.columns = ["words", "tf"]

    #Barplot
    if Barplot:
        tf[tf["tf"] > 500].plot.bar(x="words", y="tf") # Select words with frequency greater than 500
        plt.show()

    #Wordcloud
    if Wordcloud:
        text = " ".join(x for x in df.Review) # Merge texts into one text
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    return Review
text_visulization(df["Review"],Barplot=False, Wordcloud=True)


#++++++++++++++++ SENTIMENT ANALYSIS ++++++++++++++++++
def variable_creation(dataframe):
    sia = SentimentIntensityAnalyzer()
    df["polarity_score"] = df["Review"].apply(lambda x: sia.polarity_scores(x)["compound"]) # Creating a new variable indicating the sentiment of the texts (a positive value of the variable implies a positive comment)
    df["sentiment_label"] = df["Review"].apply(lambda x: 1 if sia.polarity_scores(x)["compound"] > 0 else 0)

    y = df["sentiment_label"] # Dependent variable
    X = df["Review"] # Independent variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
    return X_train, X_test, y_train, y_test
# Creating train and test datasets by calling the variable creation function
X_train, X_test, y_train, y_test = variable_creation(df["Review"])

def vectorization(dataframe):
    tf_idf_word = TfidfVectorizer()
    X_train_tf_idf_word = tf_idf_word.fit_transform(X_train)
    X_test_tf_idf_word = tf_idf_word.fit_transform(X_test)
    return X_train_tf_idf_word,  X_test_tf_idf_word
X_train_tf_idf_word ,X_test_tf_idf_word = vectorization(df["Review"])

def base_models(X, y, cv=5, scoring="accuracy"):
    print("base models")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=5, scoring=scoring, n_jobs=-1)
        for metric in scoring:
            score_key = f'test_{metric}'  # Create score key
            mean_score = round(cv_results[score_key].mean(), 4) # Calculate mean score
            print(f"{metric}: {mean_score} ({name}) ")
base_models(X_train_tf_idf_word, y_train, cv=5, scoring=["accuracy","roc_auc","f1"])

"""
base models
accuracy: 0.984 (LR) 
roc_auc: 0.9988 (LR) 
f1: 0.9901 (LR) 
accuracy: 0.7772 (KNN) 
roc_auc: 0.8572 (KNN) 
f1: 0.8492 (KNN) 
accuracy: 0.994 (CART) 
roc_auc: 0.9908 (CART) 
f1: 0.9962 (CART) 
accuracy: 0.9886 (RF) 
roc_auc: 0.9986 (RF) 
f1: 0.993 (RF) 
accuracy: 0.988 (GBM) 
roc_auc: 0.9962 (GBM) 
f1: 0.9925 (GBM) 
accuracy: 0.9951 (XGBoost) 
roc_auc: 0.9987 (XGBoost) 
f1: 0.9969 (XGBoost) 
accuracy: 0.996 (LightGBM) 
roc_auc: 0.9992 (LightGBM) 
f1: 0.9975 (LightGBM) 
accuracy: 0.9929 (CatBoost)
roc_auc: 0.9997 (CatBoost) 
f1: 0.9955 (CatBoost) 
"""
def model_validation_with_the_test_dataset(dataframe):
    cat_model = CatBoostClassifier().fit(X_train_tf_idf_word, y_train) # Building the model
    cv = cross_validate(cat_model, X_test_tf_idf_word, y_test, cv=5, scoring=["accuracy", "roc_auc", "f1"], n_jobs=-1)

    # Taking the mean of the results
    mean_accuracy = np.mean(cv["test_accuracy"]) #0.98
    mean_roc_auc = np.mean(cv["test_roc_auc"]) #0.99
    mean_f1 =np.mean(cv["test_f1"]) #0.98
    print(mean_accuracy, mean_roc_auc, mean_f1)
    return mean_accuracy, mean_roc_auc, mean_f1

mean_accuracy, mean_roc_auc, mean_f1 = model_validation_with_the_test_dataset(df)

# ++++++++++++++++++++ HYPERPARAMETER OPTIMIZATION +++++++++++++++++++
def hyperparameter_optimization(dataframe):
    cb_model = CatBoostClassifier(random_state=17) # Building an empty model object
    cb_params = {"iterations": [200, 300],
             "learning_rate": [0.01, 0.1],
             "depth": [3, 6]}

    cb_best_grid = GridSearchCV(cb_model, cb_params, cv=5, n_jobs=-1, verbose=1).fit(X_train_tf_idf_word, y_train)
    cb_best_grid.best_params_ # Best parameter values
    cb_final = cb_model.set_params(**cb_best_grid.best_params_, random_state=17).fit(X_train_tf_idf_word, y_train) # Building the final model with the best parameter values
    cvf = cross_validate(cb_final, X_test_tf_idf_word, y_test, cv=5, scoring=["accuracy", "roc_auc", "f1"], n_jobs=-1)

    # Taking the mean of the results
    mean_accuracy_f = np.mean(cvf["test_accuracy"]) #0.98
    mean_roc_auc_f = np.mean(cvf["test_roc_auc"]) #0.99
    mean_f1_f =np.mean(cvf["test_f1"]) #0.98
    print(mean_accuracy_f, mean_roc_auc_f, mean_f1_f)
    return mean_accuracy_f, mean_roc_auc_f, mean_f1_f

mean_accuracy_f, mean_roc_auc_f, mean_f1_f = hyperparameter_optimization(df)


