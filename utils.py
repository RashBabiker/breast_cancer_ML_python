import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pdb  # for debugging
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def correlation(data, size=11):
    corr = data.corr()
    plt.figure(figsize=(size, size))
    plt.title('Correlation matrix')
    plt.imshow(corr, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation='vertical')
    plt.yticks(range(len(corr)), corr.columns)
    plt.show()


def pca_analysis(data, diagnosis):
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data)

    PC1 = pca_result[:, 0]
    PC2 = pca_result[:, 1]

    # Create scatter plot
    plt.scatter(PC1, PC2, c=diagnosis.map({'B': 'blue', 'M': 'red'}))

    # Create legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Benign',
               markerfacecolor='blue', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Malign',
               markerfacecolor='red', markersize=10)
    ]
    plt.legend(handles=legend_elements, loc="best")

    # title and labels
    plt.title('PCA')
    plt.xlabel('PC1' + ' ' +
               str(round(pca.explained_variance_ratio_[0]*100, 2)) + '%')
    plt.ylabel('PC2' + ' ' +
               str(round(pca.explained_variance_ratio_[0]*100, 2)) + '%')

    plt.show()


def detect_outliers(df):
    outlier_count = {}
    for col in df.columns:
        data = df[col]
        # calculate interquartile range
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        # calculate lower and upper bounds
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        # count number of outliers
        outlier_count[col] = sum((data < lower_bound) | (data > upper_bound))
    return outlier_count


def validation(y, y_pred):
    # Compute the accuracy
    acc = accuracy_score(y, y_pred)
    print("Accuracy: {:.3f}".format(acc))

    # Compute the recall
    rec = recall_score(y, y_pred, pos_label='M')
    print("Recall: {:.3f}".format(rec))

    # Compute the confusion matrix
    cm = confusion_matrix(y, y_pred, labels=["M", "B"])

    # confusion matrix plot
    cm_display = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Malignant", "Benign"])
    print(cm_display.plot())
