import pandas as pd
import os
from logging import Logger


def load_data(data_dir, data_file, logger: Logger) -> pd.DataFrame:
    """
    loads data from data file
    :param logger:
    :param data_dir:
    :param data_file:
    :return: pandas dataframe
    """
    logger.info('Loading data from file(s)')
    try:
        if data_file is not None:
            logger.info(f'Data path: {os.path.join(data_dir, data_file)}')
            data_path = os.path.join(data_dir, data_file)
            df = pd.read_csv(data_path)

        else:
            logger.info(f'Data path: {data_dir}')
            files = os.listdir(data_dir)
            df_list = []
            for file in files:
                file_path = os.path.join(data_dir, file)
                temp_df = pd.read_csv(file_path, header=None, index_col=None, delimiter=';', quoting=3)
                df_list.append(temp_df)
            df = pd.concat(df_list, ignore_index=True)

        return df
    except Exception as e:
        logger.error(f'Loading data from file(s): FAILED. {e}')


def clean_and_format_data(a_df: pd.DataFrame, num_cols, logger: Logger) -> pd.DataFrame:
    """
    transformations that make the data suitable for feature engineering and preprocessing
    :param a_df:
    :param num_cols:
    :param logger:
    :return:
    """
    # removing quotation marks from string columns and column index
    logger.info('Cleaning and formatting data')
    try:
        logger.info('Removing quotation marks from column names')
        a_df.iloc[0, :] = a_df.iloc[0, :].str.replace(r'"', '')
        logger.info('Setting column names')
        a_df.columns = a_df.iloc[0, :]
        logger.info('Removing first row containing column names')
        a_df = a_df.iloc[1:, :].copy()
        logger.info('Converting to string type')
        a_df = a_df.apply(lambda s: s.astype(str).str.replace(r'"', ''))
        logger.info('Converting numeric columns to numeric type')
        a_df.loc[:, num_cols] = a_df.loc[:, num_cols].apply(lambda s: pd.to_numeric(s))
    except Exception as e:
        logger.error(f'Cleaning and formatting data: FAILED. {e}')

    return a_df


def age_range(a_df: pd.DataFrame, config, logger: Logger) -> pd.DataFrame:
    """
    feature engineering - create categorical feature called age_range
    :param a_df:
    :param config:
    :return:
    """
    logger.info('Creating feature - age_range')
    try:
        age_col = config.etl.age_col
        b = config.etl.age_bins
        a_df['age_range'] = a_df[age_col].transform(lambda x: f'{b * (x // b)}-{b * (x // b + 1)}')
        config.etl.cat_cols.append('age_range')
    except Exception as e:
        logger.error(f'Creating feature - age_range: FAILED. {e}')

    return a_df


def add_features(a_df: pd.DataFrame, config, logger: Logger) -> pd.DataFrame:
    """
    call all feature engineering functions
    :param a_df:
    :param config:
    :return:
    """
    logger.info('Creating engineered features')
    try:
        a_df = age_range(a_df, config, logger)
    except Exception as e:
        logger.error(f'Creating engineered features: FAILED. {e}')
    return a_df


def run_etl(config, logger: Logger) -> pd.DataFrame:
    logger.info('Running ETL...')

    df = pd.DataFrame()

    try:
        data_dir = config.data.data_dir
        data_file = config.data.data_file
        df = load_data(data_dir, data_file, logger)

        num_cols = config.etl.num_cols
        df = clean_and_format_data(df, num_cols, logger)

        df = add_features(df, config, logger)

        logger.info('Running ETL: SUCCESS.')
    except Exception as e:
        logger.error('Running ETL: FAILED.')

    return df


if __name__ == '__main__':
    pass
