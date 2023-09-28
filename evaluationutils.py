import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14):

    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )

    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    axes.set_ylabel('True label')
    axes.set_xlabel('Predicted label')
    axes.set_title("CM for " + class_label)

def evaluate(y_test, prediction, y, name):
    print(name)
    ham = metrics.hamming_loss(y_test, prediction)
    print('Hamming loss on test data: {:.3f}'.format(ham))
    acc = metrics.accuracy_score(y_test, prediction)
    print('Accuracy on test data: {:.3f}'.format(acc))
    try:
        rocauc = metrics.roc_auc_score(y_test, prediction, multi_class='ovr')
        print('ROC AUC on test data: {:.3f}'.format(rocauc))
    except ValueError:
        pass

    print(metrics.classification_report(y_test, prediction, target_names=y, digits=3, zero_division=0))
    rep = metrics.classification_report(y_test, prediction, target_names=y, digits=3, zero_division=0, output_dict=True)

    vis_arr = metrics.multilabel_confusion_matrix(y_test, prediction)

    length = len(np.unique(y))
    fig, ax = plt.subplots(length//6, length//2, figsize=(6*length//2, 3*length//6), facecolor='white')

    for axes, cfs_matrix, label in zip(ax.flatten(), vis_arr, y):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"])

    fig.tight_layout()
    plt.show()
    try:
        rocauc
    except NameError:
        rocauc = 0
    return ham, acc, rocauc, rep