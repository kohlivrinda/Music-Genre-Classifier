#Relevant imports

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os


from xgboost import XGBClassifier
import xgboost as xgb

from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, precision_score, recall_score, matthews_corrcoef


def helper_calcs(y_test, preds):
    
    """ Function to calculate values needed to calculate metrics
    
    Args:
        y_test (pd df): test labels
        preds: model predictions

    Returns:
        tp: true positives
        tn: true negatives
        fp: false positives
        fn: false negatives
        cm: confusion matrix
    """
    cm = confusion_matrix(y_test, preds)
    
    #multi-class shenanigans
    TP = np.diag(cm)
    TN = np.zeros_like(TP)
    FP = np.zeros_like(TP)
    FN = np.zeros_like(TP)
    num_classes = cm.shape[0]
    for i in range(num_classes):
        TN[i] = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
        FP[i] = np.sum(cm[:, i]) - cm[i, i]
        FN[i] = np.sum(cm[i, :]) - cm[i, i]
    tp = np.sum(TP)
    tn = np.sum(TN)
    fp = np.sum(FP)
    fn = np.sum(FN)
    
    return tp, tn, fp, fn, cm
    
    
def calculate_metrics(y_test, preds):
    
    """Function to calculate multi-class classification metrics
    
    Args:
        y_test: test labels
        preds: predicted labels

    Returns:
        specificity: TN/N
        sensitivity: TP/P
        accuracy: (TP+TN)/(TP+TN+FP+FN)
        precision: TP/(FP+TP)
        fpr: FP/N
        fnr: FN/P
        npv: TN/(TN+FN)
        fdr:FP/(FP+TP)
        f1: 2 * (Precision * Recall) / (Precision +Recall)
        mcc: (TP*TN)-(FP*FN) /SQRT((TP+FP)(TP+FN)(TN+FP)(TN+FN))
        cm: confusion matrix
    """
    
    sensitivity = recall_score(y_test, preds, average='macro')
    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average='macro')
    f1 = f1_score(y_test, preds, average='macro')
    mcc = matthews_corrcoef(y_test, preds)
    
    tp, tn, fp, fn , cm = helper_calcs(y_test, preds)
    
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    npv = tn / (tn + fn)
    fdr = fp / (fp + tp)
    specificity = tn / (tn + fp)
    
    return specificity, sensitivity, accuracy, precision, fpr, fnr, npv, fdr, f1, mcc, cm


def plot_heatmap(ratio_cap:str, cm, PATH):
    """function to plot heatmap 

    Args:
        ratio_cap (str): test/train ratio in str format for plot title
        cm (_type_): confusion matrix
        PATH (_type_): path to save plot image
    """
    
    file_name=f'{ratio_cap}.png'
    plt.figure(figsize = (16, 9))
    sns.heatmap(cm, cmap="GnBu", annot=True, 
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
           yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);
    plt.title(f"Confusion matrix for {ratio_cap} split")
    plt.savefig(os.path.join(PATH, '/conf_mat', f'/{file_name}'))
    
def plot_roc_ovr(coi, y_train, y_test, pred_prob, lab, ratio_cap, PATH):
    """function to plot OVR ROC curve for a selected class.

    Args:
        coi (int): class number [ 0 to 9 ]
        y_train (): train labels
        y_test (_type_): test labels
        pred_prob (_type_): predicted labels probability
        lab (str): class name [ music genres ]
        ratio_cap (str): test/train ratio in str format for plot title
    """
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)
    class_id = np.flatnonzero(label_binarizer.classes_ == coi)[0]
    
    RocCurveDisplay.from_predictions(
    y_onehot_test[:, class_id],
    pred_prob[:, class_id],
    name=f"{lab} vs the rest",
    color="darkorange",
    )
    file_name=f'{ratio_cap}.png'
    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"One-vs-Rest ROC curves:\n {lab} vs Rest for {ratio_cap} split")
    plt.legend()
    plt.show()
    plt.savefig(os.path.join(PATH, '/roc', f'/{file_name}'))
    
def train_model(model, ratio:float, X, y, PATH, coi, lab, ratio_cap:str):
    
    """function to train model and print out accuracy metrics and plots.
    Args:
        model: model instance
        ratio (float) : ratio for test/train split
        X : train dataframe
        y : test dataframe
        PATH : path to save plots
        coi (int): class of interest for OVR ROC
        lab: class name for OVR ROC
        ratio_cap: ratio for test/train split in str format for plot captions
    
    """
    
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=ratio, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    pred_prob=model.predict_proba(X_test)
    
    specificity, sensitivity, accuracy, precision, fpr, fnr, npv, fdr, f1, mcc , cm= calculate_metrics(y_test, preds)
    
    print('Specificity: {:.3f}'.format(specificity))
    print('Sensitivity: {:.3f}'.format(sensitivity))
    print('Accuracy: {:.3f}'.format(accuracy))
    print('Precision: {:.3f}'.format(precision))
    print('FPR: {:.3f}'.format(fpr))
    print('FNR: {:.3f}'.format(fnr))
    print('NPV: {:.3f}'.format(npv))
    print('FDR: {:.3f}'.format(fdr))
    print('F1-Score: {:.3f}'.format(f1))
    print('MCC: {:.3f}'.format(mcc))
    
    plot_heatmap(ratio_cap, cm, PATH)
    plot_roc_ovr(coi, y_train, y_test, pred_prob, lab, ratio_cap, PATH)

    



data=pd.read_csv(r'D:\college stuff\projects\Music-Genre-Classifier\features_3_sec.csv')
data=data.iloc[0:, 2:]
data['label'] = data['label'].map({'blues': 0, 'jazz': 1, 'metal': 2, 'pop': 3, 'reggae': 4, 'disco' : 5, 'classical': 6, 'hiphop' : 7 , 'rock' : 8, 'country': 9})

y = data['label'] # genre variable.
X = data.loc[:, data.columns != 'label'] 

FOREST_PATH=r'D:\college stuff\projects\Music-Genre-Classifier\plots\Random Forest'
XGB_PATH=r'D:\college stuff\projects\Music-Genre-Classifier\plots\XGBoost'

xgb_class = XGBClassifier(n_estimators=1000, learning_rate=0.05)
train_model(xgb_class, 0.3, X, y, XGB_PATH, 0, 'blues' , '30-70')

fig, ax = plt.subplots(figsize=(10, 20))
xgb.plot_importance(xgb_class, ax=ax, show_values=False)
ax.set_xlabel('F-Score', fontsize=14)
ax.set_ylabel('Features', fontsize=14)
ax.set_title('Feature Importance', fontsize=16)
plt.show()