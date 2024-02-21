# Kozmos Sentiment Analysis and Classification Project
This project aims to analyze the reviews received by Kozmos, a brand selling home textiles and daily clothing on Amazon, and to create a classification model to increase sales.
## Project Story
Kozmos, a brand selling home textiles and daily clothing on Amazon, aims to increase sales by analyzing the reviews of its products and improving their features based on the complaints received. In line with this goal, sentiment analysis will be performed on the reviews to label them, and a classification model will be created using the labeled data.
## Dataset
The dataset consists of reviews for a specific product group, including the review title, star rating, and the number of people who found the review helpful. It comprises 4 variables and 5611 observations.
- Star: Star rating given to the product
- Helpful: Number of people who found the review helpful
- Title: Title given to the review, a brief comment
- Review: Review made for the product
## Steps
#### 1. Data Reading:
The Amazon dataset is read from a file named amazon.xlsx.
#### 2. Text Preprocessing:
Reviews are prepared with text preprocessing steps.
- Texts are converted to lowercase.
- Punctuation marks and numbers are removed.
- Stopwords (unnecessary words) are removed.
- Rare words are removed.
- Words are lemmatized.
#### 3. Text Visualization:
Reviews are visualized to create a vocabulary and word cloud.
#### 4. Sentiment Analysis:
Sentiment analysis is performed on the reviews and labeled according to emotions.
#### 5. Modeling:
Text data is vectorized and various classification models are tested.
- Logistic Regression, KNN, Decision Trees, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
#### 6. Model Evaluation:
The created models are evaluated with test data, and their performances are compared.
#### 7. Hyperparameter Optimization:
Since the dependent variable is a categorical variable, the hyperparameters of the catboost model are optimized with GridSearchCV.
#### 8. Results: 
The performance and hyperparameters of the best model are reported.

## Requirements
To run the project, the following libraries need to be installed:
- nltk
- textblob
- wordcloud
- catboost
- lightgbm
- xgboost

