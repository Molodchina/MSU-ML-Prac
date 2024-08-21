from catboost import CatBoostRegressor
import pandas as pd
from hyperopt.pyll import scope
from hyperopt import hp, fmin, tpe, Trials
from sklearn.model_selection import cross_val_score
from numpy import ndarray
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor

"""
 Внимание!
 В проверяющей системе имеется проблема с catboost.
 При использовании этой библиотеки, в скрипте с решением необходимо инициализировать метод с использованием `train_dir` как показано тут:
 CatBoostRegressor(train_dir='/tmp/catboost_info')
"""


def fill_missing_values(df, features):
    for feature in features:
        df[feature] = df[feature].fillna(df[feature].mean())
    return df


def fill_missing_with_unknown(df, features):
    for feature in features:
        df[feature] = df[feature].fillna('unknown')
        mask = df[feature] != 'unknown'
        df.loc[mask, feature] = df.loc[mask, feature].str.join(' ')
    return df


def train_model_and_predict(train_file: str, test_file: str) -> ndarray:
    """
    This function reads dataset stored in the folder, trains predictor and returns predictions.
    :param train_file: the path to the training dataset
    :param test_file: the path to the testing dataset
    :return: predictions for the test file in the order of the file lines (ndarray of shape (n_samples,))
    """

    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    categorical_features = ["genres", "directors", "filming_locations", "keywords"]

    actor_features = ["actor_0_gender", "actor_1_gender", "actor_2_gender"]

    numeric_features = ["potions", "questions", "runtime", "critics_liked", "release_year"]

    df = pd.concat([df_train, df_test])

    df = fill_missing_with_unknown(df, categorical_features)
    df = fill_missing_with_unknown(df, actor_features)
    df = fill_missing_values(df, numeric_features)

    df_train = df.loc[:len(df_train)]
    df_test = df.loc[len(df_train):]

    v_genres = CountVectorizer(stop_words=["unknown"], decode_error='ignore')
    v_directors = CountVectorizer(stop_words=["unknown", "mrs", "mr", "ms", "miss"], decode_error='ignore')
    v_locations = CountVectorizer(stop_words=["unknown"], decode_error='ignore')
    v_keywords = CountVectorizer(stop_words=["unknown"], decode_error='ignore')

    genres_col = v_genres.fit_transform(df_train['genres'].values.astype('U'))
    directors_col = v_directors.fit_transform(df_train['directors'].values.astype('U'))
    locations_col = v_locations.fit_transform(df_train['filming_locations'].values.astype('U'))
    keywords_col = v_keywords.fit_transform(df_train['keywords'].values.astype('U'))

    v_genres = CountVectorizer(stop_words=["unknown"], decode_error='ignore')
    genres_col = v_genres.fit_transform(df_train['genres'].values.astype('U'))
    genres_col = v_genres.transform(df_test['genres'].values.astype('U'))

    df_train = pd.concat([df_train,
                          pd.DataFrame(genres_col.toarray()),
                          pd.DataFrame(directors_col.toarray()),
                          pd.DataFrame(locations_col.toarray()),
                          pd.DataFrame(keywords_col.toarray())
                          ], axis=1)

    genres_col = v_genres.transform(df_test['genres'].values.astype('U'))
    directors_col = v_directors.transform(df_test['directors'].values.astype('U'))
    locations_col = v_locations.transform(df_test['filming_locations'].values.astype('U'))
    keywords_col = v_keywords.transform(df_test['keywords'].values.astype('U'))

    df_test = pd.concat([df_test,
                         pd.DataFrame(genres_col.toarray()),
                         pd.DataFrame(directors_col.toarray()),
                         pd.DataFrame(locations_col.toarray()),
                         pd.DataFrame(keywords_col.toarray())
                         ], axis=1)

    s = (df_train.dtypes == 'object')
    cols = list(s[s].index)

    for col in cols:
        del df_train[col]
        del df_test[col]

    y_train = df_train["awards"]
    del df_train["awards"]

    model = CatBoostRegressor(n_estimators=1150,
                              max_depth=8,
                              learning_rate=0.02,
                              verbose=False)
    model.fit(df_train.to_numpy(), y_train.to_numpy())
    return model.predict(df_test.to_numpy())
