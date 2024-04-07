import os 
import sklearn
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt
import seaborn as sns

model_dir = 'code/models/irmas_rf_comb_10c'

def merge_two_data():
    df1 = pd.read_csv('data/irmas/vggish/training/vggish-all.csv')
    df2 = pd.read_csv('data/irmas/IRMAS-Training.csv')
    
    # join two dataframes on filename
    df1 = df1.set_index('filename')
    df1.drop(columns=['class'], inplace=True)
    
    # for df1 filename, just take the part after the last '\'
    df1.index = df1.index.str.split('\\').str[-1]
    
    df2 = df2.set_index('filename')
    df = df1.join(df2, how='inner', lsuffix='_1', rsuffix='_2')
    
    df.to_csv('data/irmas/combined-librosa-vggish.csv')

def load_data():
    df = pd.read_csv('data/irmas/combined-librosa-vggish.csv')
    
    # get column names without class and filename
    feature_names = df.columns.tolist()
    feature_names.remove('class')
    feature_names.remove('filename')
    
    X = df.drop(columns=['class'])
    y = df['class']

    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, le, feature_names

def train_rf(X_train, X_test, y_train, y_test):
    rf = RandomForestClassifier(max_depth=50, n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')
    
    return rf

def show_feature_importance(rf, feature_names):
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # plot feature importance using sns
    plt.figure()
    plt.title('Feature importances')
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'feature_importance.png'))
    
    # save one more plot showing top 10 features
    # give sufficient space for x-axis labels
    plt.figure()
    plt.title('Top 10 Feature importances')
    sns.barplot(x=importances[indices[:10]], y=np.array(feature_names)[indices[:10]])
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'feature_importance_top10.png'))
    

def store_model(rf):
    joblib.dump(rf, os.path.join(model_dir, 'rf.joblib'))
    
def load_model():
    return joblib.load(os.path.join(model_dir, 'rf.joblib'))

def predict_rf(rf, X):
    return rf.predict(X)

if __name__ == "__main__":
    # merge_two_data()
    # exit()
    
    X, y, le, feature_names = load_data()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_filename = X_train['filename']
    X_test_filename = X_test['filename']
    
    X_train = X_train.drop(columns=['filename'])
    X_test = X_test.drop(columns=['filename'])
    rf = train_rf(X_train, X_test, y_train, y_test)
    show_feature_importance(rf, feature_names)
    
    store_model(rf)
    
    # plot a confusion matrix on testing data
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # plot confusion matrix using seaborn, showing class label names instead of numbers
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    
    # evaluate with some more metrics 
    print(sklearn.metrics.classification_report(y_test, y_pred, target_names=le.classes_))
    # save result to txt
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(sklearn.metrics.classification_report(y_test, y_pred, target_names=le.classes_))
    
    # show some sample predictions (filename, predicted label, actual label)
    for i in range(5):
        pred = predict_rf(rf, X_test.iloc[[i]])
        
        print(f'{X_test_filename.iloc[i]}: predicted label {le.inverse_transform(pred)}, ground truth {le.inverse_transform([y_test[i]])}')