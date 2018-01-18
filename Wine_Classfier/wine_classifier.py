from pathlib import Path
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import sys
import os
import seaborn as sns
from sklearn.externals import joblib
import plotly.plotly as py
import plotly.graph_objs as go
from sklearn.decomposition import PCA,NMF
from plotly import offline
from sklearn.externals import joblib


#configure pandas
pd.set_option('display.max_columns',20)
pd.set_option('display.width',800)

parent_dir = Path('/Users/akilthms/Documents/CODE/ML_Guild')
train_path = (parent_dir / 'data' / 'train_data.csv').resolve()
test_path = (parent_dir / 'data' / 'test_data_withClass.csv').resolve()
out_sample = (parent_dir / 'data' / 'test_data.csv').resolve()

os.chdir(str(parent_dir))
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
feature_cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline']
X = train_set[feature_cols]
Y = train_set['wine_class']
X_test=test_set[feature_cols]
Y_test = test_set['wine_class']

def knn_model(nn,train_test=[X,Y,X_test,Y_test]):
    knn = KNeighborsClassifier(n_neighbors=nn,n_jobs=-1)
    #Create feature dataframes
    feature_cols = ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity', 'hue', 'OD280/OD315_of_diluted_wines', 'proline']
    X = train_test[0]
    Y = train_test[1]
    X_test = train_test[2]
    Y_test = train_test[3]
    #tunning parameters
    knn.fit(X,Y)
    #print(X_test)
    y_pred = knn.predict(X_test)
    evaluation = X_test.copy()
    evaluation['prediction'] = y_pred
    evaluation['wine_class'] = Y_test
    evaluation['equals?'] = evaluation[['wine_class','prediction']].apply(lambda x:  'match' if x[0] == x[1] else 'no_match',axis=1)
    accuracy=evaluation[evaluation['equals?']=='match'].shape[0]
    total = evaluation.shape[0]
    model_eval = accuracy/total
    print("Single Model Evaluation Accuracy: ",model_eval) 
    return knn,model_eval,evaluation
def knn_pred_probs(knn):
    pred_probs = knn.predict_proba(X_test)#X_test.drop('equals?',axis=1)
    return pred_probs
def eval_knn(fname,rng,train_test=[X,Y,X_test,Y_test]): 
    evals=[knn_model(nn,train_test)[1] for nn in rng]
    trace = go.Scatter(
    x = rng,
    y = evals
    )
    data = [trace]
    offline.plot(data,fname)
    return data
    #py.image.save_as(trace,'chris-plot.png')
    #py.iplot(data, filename='./Visualizations/knn_model_eval')
def draw_plot(x,y,lables,fname):
    title = lables['title']
    x_axis = labels['x']
    y_axis = labels['y']
    trace = go.Scatter(
        x=x,
        y=y
    )
    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=x_axis,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=y_axis,
            titlefont=dict(
                family='Courier New, monospace',
                size=18,
                color='#7f7f7f'
            )
        )
    )
    data = [trace]
    fig = dict(data=data,layout=layout)
    offline.plot(fig,filename='./Visualizations/'+fname)


def reduce_vars(X,X_test):
    pca = PCA(n_components='mle',svd_solver='full')
    pca.fit(X)
    X_reduce = pca.transform(X)
    dfx = pd.DataFrame(X_reduce)
    pca.fit(X_test)
    X_test_reduce = pca.transform(X_test)
    dfx_test = pd.DataFrame(X_test_reduce)
    return dfx,dfx_test
def save_model(model,path):
    joblib.dump(model, path)


if __name__ == '__main__':
    dfx,dfx_test = reduce_vars(X,X_test)
    tt=[dfx,Y,dfx_test,Y_test]
    knn = knn_model(nn=8,train_test=tt)

    #save_model(knn,'knn_pca.pkl')
    #eval_knn('./Visualizations/knn_model_eval_pca_30.html',list(range(1,30)),train_test=tt)
    #Create feature dataframes
    #tunning parameters

    #K-fold cross-validation
    from sklearn.cross_validation import cross_val_score
    metrics = ["euclidean",
        "minkowski",		
    ]
    weight_options = ['uniform','distance']
    x_total = X.append(X_test)
    y_total = Y.append(Y_test)
    pca2 = PCA(n_components='mle',svd_solver='full')
    pca_x = pca2.fit_transform(x_total)
    knn2 = KNeighborsClassifier(n_neighbors=12,n_jobs=-1)
    scores=cross_val_score(knn2,pca_x,y_total,cv=10,scoring='accuracy')
    #take the mean of the scores
    print("Cross Validation Score: ", scores.mean())
    from sklearn.grid_search import GridSearchCV
    k_range = list(range(1,30))
    param_grid = dict(n_neighbors=k_range)
    grid = GridSearchCV(knn2, param_grid, cv=10,scoring='accuracy',n_jobs=-1)
    grid.fit(x_total,y_total)
    gs=grid.grid_scores_
    print("Best Score: ",grid.best_score_)
    print("Best Params: ",grid.best_params_)
    print("Best Estimator:",grid.best_estimator_)
    grid_mean_scores = [g.mean_validation_score for g in gs]
    labels = {
        'title': 'KNN Model Evaluation: Cross Validation 10-Folds',
        'x': '# of Neighbors',
        'y': 'Accuracy score'
    }
    #draw_plot(k_range,grid_mean_scores,labels,'KNN Model Eval_Test.html')
    #save_model(grid.best_estimator_, "./Models/Best_Estimator_knn.pkl")
    #Creating a pipline to optimize pca
    from sklearn.pipeline import Pipeline
    pipe=Pipeline([
        ('reduce_dim',PCA()),
        ('classify',KNeighborsClassifier())
    ])
    n_options = list(range(1,X.shape[1]+1))
    k_range = list(range(1,30))
    param_grid = dict(
        reduce_dim=[PCA(), NMF()],
        reduce_dim__n_components= n_options,
        classify__n_neighbors=k_range,
        classify__weights=weight_options)
    grid = GridSearchCV(pipe, param_grid, cv=10,scoring='accuracy',n_jobs=-1)
    grid.fit(x_total,y_total)
    print("Pipe ","Best Score: ",grid.best_score_)
    print("Pipe ","Best Params: ",grid.best_params_)
    print("Pipe ","Best Estimator:",grid.best_estimator_)
    #Evaluating our Classifier
    #Decision Tree
     