import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from typing import List
from logging import Logger


def feature_label_split(a_df: pd.DataFrame, label_col: str, logger: Logger):
    logger.info('Splitting labels from features.')
    try:
        label_df = a_df.pop(label_col)

        return a_df, label_df
    except Exception as e:
        logger.error(f'Splitting labels from features: FAILED. {e}')


def scale_tranform_num_cols(train_df: pd.DataFrame, config):
    num_cols = config.etl.num_cols
    scaler = StandardScaler()
    scaler.fit(train_df[num_cols].values)
    Xtrain_scaled = scaler.transform(train_df[num_cols].values)

    return Xtrain_scaled, scaler, num_cols


def transform_num_cols(test_df: pd.DataFrame, scaler, num_cols: List[str]):
    Xtest_scaled = scaler.transform(test_df[num_cols].values)

    return Xtest_scaled


def encode_transform_cat_cols(train_df: pd.DataFrame, config, logger: Logger):
    logger.info('Encoding categorical features')
    try:
        encoding_dict = dict(config.preprocessing.encoding)
        logger.info('Ordinal encodings')
        ord_enc = dict(config.preprocessing.ordinal_encodings)
        ord_cols = [k for k in encoding_dict.keys() if encoding_dict[k] == 'ordinal']
        ord_df = train_df[ord_cols]
        # filtering out features in encoding dict that are not present in data
        for c in ord_cols:
            ord_enc[c] = {k: v for (k, v) in ord_enc[c].items() if k in train_df[c].unique()}
            ord_df.loc[:, c] = ord_df.loc[:, c].map(ord_enc[c])
        Xtrain_ord = ord_df.values

        logger.info('One-hot encodings')
        ohe_cols = [k for k in encoding_dict.keys() if encoding_dict[k] == 'one-hot']
        Xtrain_ohe = train_df[ohe_cols].values
        ohe_enc = OneHotEncoder(handle_unknown='ignore')
        ohe_enc.fit(Xtrain_ohe)
        Xtrain_ohe = ohe_enc.transform(Xtrain_ohe).toarray()

        logger.info('Combining encoded data')
        Xtrain_enc = np.concatenate([Xtrain_ord, Xtrain_ohe], axis=1)
        logger.info(f'Final shape for encoded categorical features: {Xtrain_enc.shape}')

        return Xtrain_enc, ord_enc, ord_cols, ohe_enc, ohe_cols
    except Exception as e:
        logger.error(f'Encoding categorical features: FAILED.')


def transform_cat_cols(test_df: pd.DataFrame, ord_enc, ord_cols, ohe_enc, ohe_cols, logger: Logger):
    logger.info('Transforming categorical features')
    try:
        logger.info(f'Ordinal encodings: {ord_cols}')
        ord_df = test_df[ord_cols]
        for c in ord_cols:
            ord_df[c] = ord_df[c].map(ord_enc[c]).fillna(-1).astype(int)
        Xtest_ord = ord_df.values

        logger.info(f'OHE encodings: {ohe_cols}')
        Xtest_ohe = test_df[ohe_cols].values
        Xtest_ohe = ohe_enc.transform(Xtest_ohe).toarray()

        Xtest_enc = np.concatenate([Xtest_ord, Xtest_ohe], axis=1)
        logger.info(f'Final shape for encoded categorical features: {Xtest_enc.shape}')

        return Xtest_enc

    except Exception as e:
        logger.error(f'Transforming categorical features: FAILED. {e}')


def transform_labels(label_df: pd.Series, config, logger: Logger):
    logger.info('Transforming label column')
    try:
        label_enc = config.preprocessing.label_encoding
        label_df = label_df.replace(label_enc).astype(int).values
    except Exception as e:
        logger.error(f'Tranforming label column: FAILED. {e}')

    return label_df


def run_preprocessing(a_df: pd.DataFrame, config, logger: Logger):
    logger.info('Running preprocessing...')
    try:
        test_size = config.preprocessing.split.test
        train_df, test_df = train_test_split(a_df, test_size=test_size, random_state=0)

        label_col = config.preprocessing.label_col
        train_features, train_labels = feature_label_split(train_df, label_col, logger)
        test_features, test_labels = feature_label_split(test_df, label_col, logger)

        Xtrain_scaled, scaler, num_cols = scale_tranform_num_cols(train_df, config)
        Xtest_scaled = transform_num_cols(test_df, scaler, num_cols)
        Xtrain_enc, ord_enc, ord_cols, ohe_enc, ohe_cols = encode_transform_cat_cols(train_features, config, logger)
        Xtest_enc = transform_cat_cols(test_features, ord_enc, ord_cols, ohe_enc, ohe_cols, logger)
        Xtrain = np.concatenate([Xtrain_scaled, Xtrain_enc], axis=1)
        Xtest = np.concatenate([Xtest_scaled, Xtest_enc], axis=1)
        logger.info(f'Final train shape: {Xtrain.shape}; final test shape: {Xtest.shape}')
        ytrain = transform_labels(train_labels, config, logger)
        ytest = transform_labels(test_labels, config, logger)

        logger.info('Running preprocessing: SUCCESS.')

        return Xtrain, ytrain, Xtest, ytest
    except Exception:
        logger.error('Running preprocessing: FAILED.')


if __name__ == '__main__':
    pass
