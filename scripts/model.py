import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from logging import Logger
import os


def fit_model(Xtrain, ytrain, config, logger: Logger):
    model = None
    clf = None
    grid_search = {}
    logger.info(f'Fitting model. Algorithm: {config.model.algorithm}')
    try:
        if config.model.algorithm == 'Random Forrest':
            grid_search = {k: v for (k, v) in config.model.rf_grid_search.items()}
            grid_search['class_weight'] = [dict([(0, 1), (1, x)]) for x in grid_search['class_weight']]
            model = RandomForestClassifier(n_estimators=500, random_state=0)

        elif config.model.algorithm == 'Logistic Regression':
            grid_search = {k: v for (k, v) in config.model.lr_grid_search.items()}
            grid_search['class_weight'] = [dict([(0, 1), (1, x)]) for x in grid_search['class_weight']]
            model = LogisticRegression(random_state=0, max_iter=500)

        logger.info(f'Performing grid search. Param ranges: {grid_search}')
        clf = GridSearchCV(model, grid_search, scoring='f1', verbose=2, n_jobs=2)
        clf.fit(Xtrain, ytrain)

    except Exception as e:
        logger.error(f'Fitting mode: FAILED. {e}')

    return clf


def predict(model, Xtest, logger: Logger):
    predictions = None
    logger.info('Predicting for test set')
    try:
        predictions = model.predict(Xtest)
    except Exception as e:
        logger.error(f'Predicting for test set: FAILED. {e}')

    return predictions


def run_model(Xtrain, ytrain, Xtest, config, output_dir: str, logger: Logger):
    clf_model = None
    predictions = None
    logger.info('Running model...')
    try:
        clf_model = fit_model(Xtrain, ytrain, config, logger)
        predictions = predict(clf_model, Xtest, logger)
        pd.DataFrame(clf_model.cv_results_).to_csv(os.path.join(output_dir, 'CV_results'))

        logger.info('Running model: SUCCESS')
    except Exception as e:
        logger.error('Running model: FAILED.')

    return clf_model, predictions


if __name__ == '__main__':
    pass
