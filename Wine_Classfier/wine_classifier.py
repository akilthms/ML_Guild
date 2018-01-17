from pathlib import Path
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sys
import os
import seaborn as sns
from sklearn.externals import joblib
import plotly.plotly as py
import plotly.graph_objs as go


#configure pandas
pd.set_option('display.max_columns',10)
pd.set_option('display.width',800)

parent_dir = Path('/Users/akilthms/Documents/CODE/ML_Guild')
train_path = (parent_dir / 'data' / 'train_data.csv').resolve()
test_path = (parent_dir / 'data' / 'test_data_withClass.csv').resolve()
out_sample = (parent_dir / 'data' / 'test_data.csv').resolve()

train_set = pd.read_csv(str(train_path))
test_set = pd.read_csv(str(test_path))

#train_set.groupby('wine_class').describe().to_csv("{}/data/describe_training.csv".format(parent_dir))

# #Visualize Realationships 
# sns.set(style="ticks", color_codes=True)
# g=sns.pairplot(train_set,hue='wine_class',vars=['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline'])
# g.savefig('Visualizations/Scatter_Matrix2.png')

# #KNN Classifier
# knn = KNeighborsClassifier(n_neighbors=3,n_jobs=-1)
# #Create feature dataframes
# feature_cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline']
# X = train_set[feature_cols]
# Y = train_set['wine_class']
# X_test = test_set[feature_cols]
# Y_test = test_set['wine_class']
# #tunning parameters
# knn.fit(X,Y)
# y_pred = knn.predict(X_test)
# evaluation = X_test
# evaluation['prediction'] = y_pred
# evaluation['wine_class'] = Y_test
# evaluation['equals?'] = evaluation[['wine_class','prediction']].apply(lambda x:  'match' if x[0] == x[1] else 'no_match',axis=1)
# accuracy=evaluation[evaluation['equals?']=='match'].shape[0]
# total = evaluation.shape[0]
# model_eval = accuracy/total
#66%

#Tuning phase

def knn_model(nn):
    knn = KNeighborsClassifier(n_neighbors=nn,n_jobs=-1)
    #Create feature dataframes
    feature_cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline']
    X = train_set[feature_cols]
    Y = train_set['wine_class']
    X_test = test_set[feature_cols]
    Y_test = test_set['wine_class']
    #tunning parameters
    knn.fit(X,Y)
    y_pred = knn.predict(X_test)
    evaluation = X_test
    evaluation['prediction'] = y_pred
    evaluation['wine_class'] = Y_test
    evaluation['equals?'] = evaluation[['wine_class','prediction']].apply(lambda x:  'match' if x[0] == x[1] else 'no_match',axis=1)
    accuracy=evaluation[evaluation['equals?']=='match'].shape[0]
    total = evaluation.shape[0]
    model_eval = accuracy/total
    print(model_eval) 
    return knn,model_eval
def knn_pred_probs(knn):
    pred_probs = knn.predict_proba(X_test)#X_test.drop('equals?',axis=1)
    return pred_probs
def eval_knn(rng): 
    evals=[knn_model(nn)[1] for nn in rng]
    trace = go.Scatter(
    x = rng,
    y = evals
    )
    data = [trace]
    return data
    #py.image.save_as(trace,'chris-plot.png')
    #py.iplot(data, filename='./Visualizations/knn_model_eval')

    
#knn_model(8) # 78 percent
#knn_model(6)
#PCA
#offline.plot(g)
#Predicted_Probablities
feature_cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline']
X = train_set[feature_cols]
Y = train_set['wine_class']

X_test=test_set[feature_cols]
Y_test = test_set['wine_class']
X_new = X_test.copy()
knn = knn_model(nn=8)[0]
foo = knn_pred_probs(knn)
baz=[p[0] for p in foo]
X_new['pred_prob'] = baz
X_new['wine_class'] = Y_test
X_new['pred_class'] = knn.predict(X_test)
threshold=X_new[['wine_class', 'pred_class','pred_prob']]
threshold[]
print(X_new[['wine_class', 'pred_class','pred_prob']])