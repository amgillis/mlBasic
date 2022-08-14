from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, auc, confusion_matrix, \
    classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from logging import Logger
import os


def score_predictions(ytrue, ypred, logger: Logger):
    scores = {}
    logger.info('Scoring test results')
    try:
        scores['acc'] = accuracy_score(ytrue, ypred)
        scores['f1'] = f1_score(ytrue, ypred)
        scores['roc_auc'] = roc_auc_score(ytrue, ypred)

        logger.info(f'Test scores: {scores}')
    except Exception as e:
        logger.error(f'Scoring test results: FAILED. {e}')

    return scores


def get_classification_report(ytrue, ypred, logger):
    cr = classification_report(ytrue, ypred)
    logger.info(f'Classification report:\n{cr}')

    return cr


def plot_cm(ytrue, ypred, output_dir):
    cm = confusion_matrix(ytrue, ypred)

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    colors = sns.light_palette('blue', as_cmap=True)
    cm_disp = sns.heatmap(cm, cmap=colors, annot=True, fmt='g', cbar=False)
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Real Values')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

    return cm


def plot_roc_curve(ytrue, ypred, output_dir):
    fpr, tpr, thresh = roc_curve(ytrue, ypred)
    test_auc = auc(fpr, tpr)

    plt.subplots(1, 1, figsize=(10,7))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label=f'area = {test_auc:.3f}')

    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))


def plot_precision_recall_curve(ytrue, ypred, output_dir):
    precision, recall, thresh = precision_recall_curve(ytrue, ypred)
    avgprec = average_precision_score(ytrue, ypred)

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    plt.plot([0, 1], [1, 0], 'k--')
    plt.plot(precision, recall, label=f'Avg Precision Score = {avgprec:.3f}')
    ax.set_xlabel('Precision')
    ax.set_ylabel('Recall')
    plt.title('Precision-Recall curve')
    plt.legend(loc='best')
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))


def run_eval(ytrue, ypred, output_dir, logger):
    scores = score_predictions(ytrue, ypred, logger)
    cr = get_classification_report(ytrue, ypred, logger)
    cm = plot_cm(ytrue, ypred, output_dir)
    plot_roc_curve(ytrue, ypred, output_dir)
    plot_precision_recall_curve(ytrue, ypred, output_dir)
    
    return scores, cr, cm


if __name__ == '__main__':
    pass
