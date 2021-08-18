# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### Real Estate rent price prediction
# Our topic is real estate price prediction in Dubai
# What our model will do is predict the real estate prices based on the data used for training.
#
# We scraped the data from Propert Finder.
# Each row in our data resembles an ad on the website.
# After collecting the data, we started cleaning and preprocessing the data.

# %%
from joblib import dump
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold, RepeatedKFold
from sklearn.model_selection import RandomizedSearchCV
import scipy as sp

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso, Ridge

from sklearn.preprocessing import StandardScaler


# %%
df = pd.read_csv("apartment_data.csv")


# %%
df


# %%
df.columns

# %% [markdown]
# # Removing duplicate data

# %%
df.duplicated().sum()

# %% [markdown]
# We notice 3k rows of duplicate data

# %%
df = df.drop_duplicates()
df

# %% [markdown]
# So now we are left with 12k rows
# %% [markdown]
# ### Removing trailing spaces around the text

# %%
df.loc[:, 'listing_type'] = df.loc[:, 'listing_type'].map(lambda x: x.strip())
df.loc[:, 'bedrooms'] = df.loc[:, 'bedrooms'].map(lambda x: x.strip())
df.loc[:, 'bathrooms'] = df.loc[:, 'bathrooms'].map(lambda x: x.strip())
df.loc[:, 'area'] = df.loc[:, 'area'].map(lambda x: x.strip())
df.loc[:, 'price'] = df.loc[:, 'price'].map(lambda x: x.strip())
df.loc[:, 'location'] = df.loc[:, 'location'].map(lambda x: x.strip())
df.loc[:, 'description'] = df.loc[:, 'description'].map(lambda x: x.strip())

# %% [markdown]
# # Checking None or missing values
# %% [markdown]
# Our data has 'None' for missing values, which seem to not be caught by pandas

# %%
df.isnull().sum()

# %% [markdown]
# #### Manually checking for 'None' Values

# %%
for label, content in df.items():
    print(
        f"label: {label} has {content.str.contains('None').sum()} None values ")

# %% [markdown]
# ### Switching None values to NaN

# %%
df.loc[:, 'bedrooms'] = df['bedrooms'].replace("None", np.nan, regex=True)
df.loc[:, 'bathrooms'] = df['bathrooms'].replace("None", np.nan, regex=True)
df.loc[:, 'area'] = df['area'].replace("None", np.nan, regex=True)
df.loc[:, 'price'] = df['price'].replace("None", np.nan, regex=True)


# %%
df.isnull().sum()

# %% [markdown]
# #### We can observe "bathrooms" having the highest number of missing values
# %% [markdown]
# ### Drop missing values
# missing values only account to 1.5% of our original data, so we decided to drop them

# %%
df = df.dropna()
df

# %% [markdown]
# # Cleaning and preprocessing data
# %% [markdown]
# ## Price Column

# %%
df

# %% [markdown]
# We need to remove AED/year and rows that have 'ask for price'
# %% [markdown]
# ### Remove AED/year from price AND 'ask for price' entries
# %% [markdown]
# Ask for price entries were removed because the price is our target attribute and so
#
# using techniques to fill 'ask for price' with an estimated number would take away from the accuracy of the model

# %%
# df['price'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')

# removing AED/year
df.loc[:, 'price'].replace(regex=True, inplace=True,
                           to_replace=r'(AED\/year)+', value=r'')

# removing commas
df.loc[:, 'price'].replace(regex=True, inplace=True,
                           to_replace=r'(,)+', value=r'')

# dropping rows that contain 'ask for price'
df = df[~(df['price'].str.contains("price"))]
df

# %% [markdown]
# ### Changing data type from object to float

# %%
df['price'] = df['price'].astype(float)


# %%


# %% [markdown]
# ## Area Column
# %% [markdown]
# ### Remove sqft

# %%
df.loc[:, 'area'].replace(regex=True, inplace=True,
                          to_replace=r'\D', value=r'')
df

# %% [markdown]
# ### Changing data type from object to float

# %%
df['area'] = df['area'].astype(float)

# %% [markdown]
# ## Checking Outliers for area and price

# %%
df['price'].astype(float).describe().apply(lambda x: format(x, 'f'))


# %%
df['area'].astype(float).describe().apply(lambda x: format(x, 'f'))

# %% [markdown]
# ### Price and area zscore

# %%
df['price_zscore'] = (df.price - df.price.mean()) / df.price.std()
df['area_zscore'] = (df.area - df.area.mean()) / df.area.std()


# %%
df['price_zscore']

# %% [markdown]
# #### filtering price outliers according to zscore

# %%
df = df[~((df.price_zscore < -3) | (df.price_zscore > 3))]
df

# %% [markdown]
# ### Area zscore

# %%
df['area_zscore']


# %%
df = df[~((df.area_zscore < -3) | (df.area_zscore > 3))]
df

# %% [markdown]
# ### Removing zscore columns

# %%
df = df.drop(['price_zscore', 'area_zscore'], axis=1)
df

# %% [markdown]
# ## listing_type column

# %%
df['listing_type'].value_counts()

# %% [markdown]
# #### **Since the last 5 entries barely have any data entries, we'll drop them**

# %%
df = df[~(df['listing_type'].isin(
    ['Bulk Rent Unit', 'Whole Building', "Full Floor", 'Bungalow', 'Compound']))]
df

# %% [markdown]
# ##### Checking that they were dropped

# %%
df['listing_type'].value_counts()

# %% [markdown]
# ### Export listing type to txt file

# %%


def export_unique_column_values_to_txt(column_name):
    with open(f'{column_name}.txt', 'w') as file:

        unique_values = df[column_name].unique()
        for idx, item in enumerate(unique_values):
            if (idx + 1 == len(unique_values)):
                file.write(item)
            else:
                file.write(item+"\n")


# %%


# %%
export_unique_column_values_to_txt('listing_type')


# %%


# %% [markdown]
# ### One hot encoding

# %%
def encode_and_bind(df, feature_to_encode):
    dummies = pd.get_dummies(df[[feature_to_encode]])
    res = pd.concat([df, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)
    return res


# %%
df = encode_and_bind(df, 'listing_type')
df


# %%


# %%


# %% [markdown]
# ## Location column
# %% [markdown]
# ### Checking unique values of location

# %%
len(df['location'].unique())

# %% [markdown]
# **1824 unique values in location**

# %%
pd.set_option('display.max_colwidth', None)
df['location']


# %%
pd.reset_option("^display")

# %% [markdown]
# ### **Observations:**
#
# 1872 unique locations in dubai only, we can also see that location here is very detailed to the point of including building names!
# - if one hot encoding were to be applied to those 1824 values, we would end up with too many columns.
# - Building names are not too important and neighborhood names are enough
#
# ### **Ways to solve the problem:**
#
# 1. manually gather major neighborhood names in dubai and convert all values that contain the name of a certain neighborhood to just that neighborhood name
# for ex: convert all values that contain "al satwa" to just "al satwa" instead of including extra detail about where in al satwa
#
# However, we noticed location follows a certain format where it starts with the precise location then the broader ones. Where the neighborhood name is always before the last dash.
#
# 2. So we can keep the right most words around the last dash.
# %% [markdown]
# ### Clean location column: extracting the neighborhood name only

# %%
df['location'] = df['location'].apply(lambda row: row.split('-')[-2])
df['location']


# %%


# %% [markdown]
# #### Removing trailing spaces

# %%
df.loc[:, 'location'] = df.loc[:, 'location'].map(lambda x: x.strip())

# %% [markdown]
# ### Count of unique location values

# %%
len(df['location'].unique())

# %% [markdown]
# Unique values reduced to 107 just by extracting the neighborhood name
# %% [markdown]
# ### Checking number of entries for each neighborhood

# %%
df['location'].value_counts()

# %% [markdown]
# **If certain entries have less than 10 occurrences, remove them to reduce number of distinct values we have for when one hot encoding is done**
# %% [markdown]
# ### Filtering the data frame by value counts of each location value

# %%
df = df.groupby("location").filter(lambda x: len(x) > 10)

# %% [markdown]
# #### Count of unique location values after filtering

# %%
len(df['location'].unique())

# %% [markdown]
# Unique value counts reduced to 83
# %% [markdown]
# #### Checking that counts less than 10 have been removed

# %%
df['location'].value_counts()


# %%
# export to txt file
export_unique_column_values_to_txt('location')

# %% [markdown]
# ### Applying one hot encoding

# %%
df = encode_and_bind(df, 'location')
df

# %% [markdown]
# #### Recap of what has changed in location column:
#
# 1. extracted and kept the neighborhood name only in location text
# 2. removed entries that have less than 10 occurrences to reduce number of columns for one hot encoding
# 3. applied one hot encoding on location

# %%
df

# %% [markdown]
# ## Bedrooms Column

# %%
df.loc[:, 'bedrooms'].value_counts()

# %% [markdown]
# **We notice that:**
#
# We have to have either all values as textual or numerical data but cannot due to "7+" and "studio"
#
# And we have two choices:
#
# 1. convert them all to textual data since they can fit into categories like: "one bedroom", "two bedrooms", "studio", etc...
#
# 2. convert them all to numerical by making studio to 0 and getting rid of "7+" values since they are only 2 entries.
#
# We decided to go with the second option to preserve the ordering of the data and since we are predicting a continuous number (price) numerical data would have a better effect than one hot encoded data of 1s and 0s

# %%
df['bedrooms'] = df['bedrooms'].str.lower()

# %% [markdown]
# #### Removing "7+" rows

# %%
df = df[df['bedrooms'] != '7+']
df.loc[:, 'bedrooms'].value_counts()

# %% [markdown]
# #### Removing row with value of 7

# %%
df = df[df['bedrooms'] != '7']
df.loc[:, 'bedrooms'].value_counts()

# %% [markdown]
# row values 7+ and 7 removed
# %% [markdown]
# #### Replacing studio with 0

# %%
df['bedrooms'] = df['bedrooms'].replace('studio', 0)


# %%
df.loc[:, 'bedrooms'].value_counts()

# %% [markdown]
# Studio values successfully replaced with 0

# %%
df['bedrooms'] = df['bedrooms'].astype(int)

# %% [markdown]
# ## Bathrooms Column

# %%
df['bathrooms'].value_counts()

# %% [markdown]
# We decided to remove the rows that have the value of "7+" due to their limited number

# %%


# %% [markdown]
# ### Removing rows with value "7+"

# %%
df = df[df['bathrooms'] != '7+']
df['bathrooms'].value_counts()


# %%
df['bathrooms'] = df['bathrooms'].astype(int)


# %%
# df.to_csv('raw_data.csv', index=False)

# %% [markdown]
# ## Seperating attributes columns and target column

# %%
Y = df['price']
Y


# %%
X = df.loc[:, df.columns != 'price']

# %% [markdown]
# # Converting description to numerical data
# This stage involves two steps:
#
#     1. Text cleaning:
#
#         - removing punctuation
#         - removing stop words
#         - stemming the text (transform word to the original form)
#
#     2. convert text to numerical form
# %% [markdown]
# Inspecting the first 2 rows

# %%
for row in df[0:2]['description']:
    print(row)
    print()
    print()

# %% [markdown]
# ### 1. clean the text

# %%
nltk.download('stopwords')

stop_words = stopwords.words('english')


# %%
def text_cleaning(text):

    text = text.lower()

    # removing \t
    text = re.sub(r'(\\t)', " ", text)

    # removing symbols
    text = re.sub(r'[*-/■%|●]+', " ", text)

    # removing numbers and phone numbers
    text = re.sub(r'(\+?(971)?[0-9]+)+', " ", text)

    text = remove_stop_words(text)

    text = stem_text(text)

    return text


# %%
def remove_stop_words(text):

    text = " ".join([word for word in text.split() if word not in stop_words])

    return text


# %%
def stem_text(text):
    # split into words
    from nltk.tokenize import word_tokenize
    tokens = word_tokenize(text)

    # stemming of words
    from nltk.stem.porter import PorterStemmer
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in tokens]
    stemmed = " ".join(stemmed)
    return stemmed

# %% [markdown]
# Manual Cleaning of the text generally results in worse results, so we will skip the step of applying cleaning on the text data

# %%
# df['description'].apply(text_cleaning)

# %% [markdown]
# ### 2. convert text to numerical values
# %% [markdown]
# There are two methods for transforming text into numerical values
#
# 1. CountVectorizer
#     - CountVectorizer adopts a bag of words approach where it takes all documents (rows of text), tokenizes the words then build a table where the columns are all the words that appear in the corpus and rows are each document (row) and the count of each word that appears in the document.
# 2. TF-IDF Vectorizer
#     - TF-IDF Vectorizer expands upon the bag of words model by including two calculations: Term frequency (tf) and inverse document frequency (idf)
#         - Term frequency (TF) is uses a bag of words matrix which is called a term frequency matrix to compute the term frequency. Term frequency is for a word in a document is computed as:
#             - the number of times a word appears in a document / total number of words in a document.
#         - Inverse document frequency calculates how important a word is. It is calculated by the following:
#             - log(number of documents /number of documents containing the word)
#             - the more commmon a word is, the lower the score. And the more unique the word is, the higher the score.
#     - **final formula: tf-idf (of a certain word) = tf * idf**
# %% [markdown]
# We decided to go with TF-IDF since it is more sensitive to words that appear frequently and would produce a better estimation of the importance of certain words
# %% [markdown]
# ## Cannot transform all text data (training and testing)
# This would not be a problem if we just fit our training data on the selected model, but since we want to use cross validation to best select a model (training data will keep changing so we cant use data with TfIdf preapplied),
# we will create a pipeline which includes all transformations needed to be applied on the training data.
#
# #### Different types of transformers used in the section below:
# - Column Transformer
# - Function Transformer
#
# Both were used in **pipelines**
#
# Example of a pipeline:
#
# ```py
# pipeline_example = Pipeline([
#     ("tfidf vectorizer", TfidfVectorizer()),
#     ("linear regression", LinearRegression(fit_intercept=False, normalize=True))
# ])
# ```
# %% [markdown]
# ### Creating function transformers
# %% [markdown]
# Column transformers are used in pipelines to transform certain columns only instead of the whole dataset

# %%
# converts data to sparse format


def to_sparse(data):
    return sp.sparse.csr_matrix(data)


def to_dense(data):
    return sp.sparse.csr_matrix.todense(data)


to_dense = FunctionTransformer(to_dense)
to_sparse_transformer = FunctionTransformer(to_sparse)


# %%
# all column names except description
# X_train_cols = X_train.loc[:, X_train.columns != 'description']
# X_train_cols = X_train_cols.columns
# X_train_cols = X_train_cols.tolist()

# %% [markdown]
# ### Cross validation
# %% [markdown]
# #### Creating models and pipeline for cross val
# %% [markdown]
# Here we created a list of the models that we would like to test.
#
# We also created a function that takes in a columns transformer and a list of models, and runs cross validation on those models by creating a pipeline and then printing the score

# %%
models = [
    LinearRegression(fit_intercept=False, normalize=True),
    RandomForestRegressor(max_depth=15),
    DecisionTreeRegressor(),
    Lasso(),
    Ridge()

]


def run_crossval_with_models(models, columns_transformer, X, Y, scoring='r2'):
    cross_val_scores = []
    for model in models:
        # creating the pipeline with column transformer and estimator of choice
        pipe = make_pipeline(
            columns_transformer,
            model
        )

        scores = cross_val_score(pipe, X, Y, n_jobs=-1, scoring=scoring)
        print(type(model).__name__, "with score of: ", scores.mean())

        cross_val_scores.append((type(model).__name__, scores))

    return cross_val_scores


# %%


# %% [markdown]
# ### Testing different approaches in the pipeline
# %% [markdown]
# We are trying different combinations for what a Column Transformer could include, such as:
# Bag of words instead of TF-idf, with/without normalization and so on

# %%
def testing_approaches():

    # Bag of words
    print("Bag of words: ")

    columns_transformer = ColumnTransformer([
        ('tf-idf', CountVectorizer(), 'description'),
        #     ('std scalar', StandardScaler(with_mean=False), X_train_cols),
        ('tosparse', to_sparse_transformer, X_train_cols)
    ], remainder='passthrough')

    run_crossval_with_models(models, columns_transformer, X, Y)

    print("---------------")

    # tf-idf
    print("TF-IDF: ")
    columns_transformer = ColumnTransformer([
        ('tf-idf', TfidfVectorizer(), 'description'),
        #     ('std scalar', StandardScaler(with_mean=False), X_train_cols),
        ('tosparse', to_sparse_transformer, X_train_cols)
    ], remainder='passthrough')

    run_crossval_with_models(models, columns_transformer, X, Y)
    print("---------------")

    #  tfidf with normalization
    print("tfidf - with normalization: ")
    columns_transformer = ColumnTransformer([
        ('tf-idf', TfidfVectorizer(), 'description'),
        ('std scalar', StandardScaler(with_mean=False), X_train_cols),
        ('tosparse', to_sparse_transformer, X_train_cols)
    ], remainder='passthrough')

    run_crossval_with_models(models, columns_transformer, X, Y)
    print("---------------")

    print("description dropped, no tfidf needed: ")
    columns_transformer = ColumnTransformer([
        #     ('tf-idf',TfidfVectorizer(), 'description'),
        ('std scalar', StandardScaler(with_mean=False), X_train_cols),
        #     ('tosparse',to_sparse_transformer , X_train_cols)
    ], remainder='passthrough')

    X_description_dropped = X.drop('description', axis=1)

    run_crossval_with_models(
        models, columns_transformer, X_description_dropped, Y)


# %%
testing_approaches()

# %% [markdown]
# **As per the results shown, we picked the pipeline of TfIdf with normalization**
# %% [markdown]
# ### Testing different values of ngrams in TfIdf

# %%
columns_transformer = ColumnTransformer([
    ('tf-idf', TfidfVectorizer(ngram_range=(1, 2)), 'description'),
    ('std scalar', StandardScaler(with_mean=False), X_train_cols),
    ('tosparse', to_sparse_transformer, X_train_cols)
], remainder='passthrough')


# %%
def testing_ngrams(ngram_range=(1, 1), scoring='r2'):
    print("tfidf - with normalization: ")
    columns_transformer = ColumnTransformer([
        ('tf-idf', TfidfVectorizer(ngram_range=ngram_range), 'description'),
        ('std scalar', StandardScaler(with_mean=False), X_train_cols),
        ('tosparse', to_sparse_transformer, X_train_cols)
    ], remainder='passthrough')

    run_crossval_with_models(
        models, columns_transformer, X, Y, scoring=scoring)
    print("---------------")


# %%
testing_ngrams((1, 2))


# %%
testing_ngrams((2, 2))


# %%
testing_ngrams((2, 3))

# %% [markdown]
# negative mean squared error scores:

# %%
print("neg mean squrare error of TF-IDF - with normalization: ")
columns_transformer = ColumnTransformer([
    ('tf-idf', TfidfVectorizer(ngram_range=(1, 2)), 'description'),
    ('std scalar', StandardScaler(with_mean=False), X_train_cols),
    ('tosparse', to_sparse_transformer, X_train_cols)
], remainder='passthrough')

neg_mean_cross_val_scores = run_crossval_with_models(
    models, columns_transformer, X, Y, scoring='neg_mean_squared_error')

# %% [markdown]
# **Trying different hyper params of Lasso and Ridge**

# %%
models = [
    Lasso(alpha=0.5),
    Ridge(alpha=0.5)
]


# %%
run_crossval_with_models(models, columns_transformer, X, Y)


# %%
models = [
    Lasso(alpha=5),
    Ridge(alpha=5)
]


# %%
run_crossval_with_models(models, columns_transformer, X, Y)


# %%
models = [
    Lasso(alpha=0.05),
    Ridge(alpha=0.05)
]


# %%
run_crossval_with_models(models, columns_transformer, X, Y)

# %% [markdown]
# **No major improvements noticed**
# %% [markdown]
# ## Exploring the effects of categorizing area

# %%
df['area'].hist(bins=14)


# %%
df['area'].max()

# %% [markdown]
# ### dividing into categories based on histogram
# %% [markdown]
# We chose an interval of 500 based on the results we got from the histogram

# %%
intervals = pd.IntervalIndex.from_tuples([(0, 500), (500, 1000), (
    1000, 1500), (1500, 2000), (2000, 2500), (2500, 3000), (3000, float('inf'))])
intervals


# %%
labels = ['0-500', '500-1000', '1000-1500',
          '1500-2000', '2000-2500', '2500-3000', '3000+']
seven_areas = pd.cut(df['area'].tolist(), bins=intervals)
seven_areas.categories = labels
df['area'] = seven_area
df['area']


# %%
df = encode_and_bind(df, 'area')

X = df.loc[:, df.columns != 'price']
y = df['price']

# all column names except description
X_cols = X.loc[:, X.columns != 'description']
X_cols = X_cols.columns
X_cols = X_cols.tolist()

# selected pipeline
columns_transformer = ColumnTransformer([
    ('tf-idf', TfidfVectorizer(ngram_range=(2, 2)), 'description'),
    ('std scalar', StandardScaler(with_mean=False), X_cols),
    ('tosparse', to_sparse_transformer, X_cols)
], remainder='passthrough')


run_crossval_with_models(models, columns_transformer, X, y)

# %% [markdown]
# -------------------------------------------------
# %% [markdown]
# # Summary
# %% [markdown]
# - Removing duplicate data
# - Removing trailing spaces
# - Dropping rows with missing values
# - Price column:
#     - removed AED/year from the price
#     - removed outliers in the data using zscore
#     - dropped rows that contain "Ask for price"
#
# - Area column:
#     - removed sqft from the area
#     - removed outliers in the data using zscore
# - listing_type:
#     - removed 5 values which had 2 entries or less
#     - applied one hot encoding
#
# - location column:
#     - extracted neighborhood name only from full location
#     - removed trailing spaces
#     - removed values that have less than 10 entries
#     - applied one hot encoding
# - bedrooms column:
#     - removed rows that had 7+ value
#     - removed rows that have value of 7
#     - replace studio with 0
# - bathrooms column:
#     - remove rows that have "7+"
# - description:
#     - text cleaning
#     - converting text to numerical form using TF-IDF
#
# %% [markdown]
# ### Exporting the model

# %%
# converts data to sparse format


def to_sparse(data):
    return sp.sparse.csr_matrix(data)


def to_dense(data):
    return sp.sparse.csr_matrix.todense(data)


to_dense = FunctionTransformer(to_dense)
to_sparse_transformer = FunctionTransformer(to_sparse)


# selected model -> Ridge Regression

X = df.loc[:, df.columns != 'price']
y = df['price']

# all column names except description
X_cols = X.loc[:, X.columns != 'description']
X_cols = X_cols.columns
X_cols = X_cols.tolist()

# selected pipeline
columns_transformer = ColumnTransformer([
    ('tf-idf', TfidfVectorizer(ngram_range=(2, 2)), 'description'),
    ('std scalar', StandardScaler(with_mean=False), X_cols),
    ('tosparse', to_sparse_transformer, X_cols)
], remainder='passthrough')
final_pipeline = Pipeline(
    [('col transformer', columns_transformer), ('ridge', Ridge())])
model_pipeline = final_pipeline.fit(X, y)


# %%
model_pipeline


# %%


# %%
if __name__ == "__main__":
    model_pipeline.__module__ = "project_final"
    dump(model_pipeline, 'model_file.joblib')


# %%
