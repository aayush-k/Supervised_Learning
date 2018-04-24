from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, learning_curve, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import csv
import numpy as np
import re
import time
import datetime
import pickle
import os
import math
import ipdb

global_params = {
    'gym': {
        # General
        'data_location': '../Data/Campus Gym/',
        # kNN (manual search)
        'n_neighbors': 15,
        'weights': 'distance',
        # Decision Trees (TODO: optimize params)
        'max_depth': 29, # unbound gives 31
        'min_samples_split': 2,
        # Boosting (grid searched)
        'n_estimators': 300,
        'learning_rate': 0.001,
        # SVM (grid searched)
        'C': 100.0,
        'gamma': 1.0,
        'kernel': 'rbf',
        # NeuralNet (manual search: a: 5e-9, (50,50,50))
        'alpha': 5e-9,
        # 'alpha': 3e-9,
        'hidden_layer_sizes': (50,50,50),
        # 'hidden_layer_sizes': (60, 50, 30),
        'random_state': 1
    },
    'song': {
        # General
        'data_location': '../Data/YearPredict/',
        # kNN (manual search)
        'n_neighbors': 35, #35
        'weights': 'distance',
        # Decision Trees (grid searched)
        'max_depth': 7,
        'min_samples_split': 342,
        # Boosting (grid searched)
        'n_estimators': 400,
        'learning_rate': 0.01,
        # SVM (grid searched)
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 0.1,
        # NeuralNet (grid searched)
        'alpha': 5e-9,
        'hidden_layer_sizes': (40,20,40),
        'random_state': 1
    }
}

def PreProcess(prob_name, show_label_distribution):

    data = []
    if prob_name == 'gym':
        pickle_location = '{0}campus_gym_data.pickle'.format(global_params[prob_name]['data_location'])
        if not os.path.exists(pickle_location):
            print('preprocessing dataset')
            with open('{0}campus_gym_data.csv'.format(global_params[prob_name]['data_location'])) as gymDataFile:
                gymData = csv.reader(gymDataFile)
                heading = True
                for row in gymData:
                    if heading:
                        heading = False
                        continue
                    data.append([math.floor(int(row[0]) / 20)] + [float(dv) for dv in row[2:]] + [float(row[1][8:10])])

            data = np.asarray(data)
            data = data[np.random.choice(data.shape[0], 20000, replace=False), :]

            pickle_out = open(pickle_location,"wb")
            pickle.dump(data, pickle_out)
            pickle_out.close()
        else:
            print('loading pickled dataset')
            data = pickle.load(open(pickle_location,"rb"))

        # allocate training data (use k fold cross validation to tune hyperparameters)
        training_data = data[:15000,:]

        # hold out testing data till end
        testing_data = data[15000:,:]

    else:
        pickle_location = '{0}YearPredictionMSD.pickle'.format(global_params[prob_name]['data_location'])
        if not os.path.exists(pickle_location):
            print('preprocessing dataset')
            with open('{0}YearPredictionMSD.csv'.format(global_params[prob_name]['data_location'])) as songDataFile:
                songData = csv.reader(songDataFile)
                for row in songData:
                    data.append([math.floor(int(row[0]) / 10)] + [float(dv) for dv in row[1:13]])
            data = np.asarray(data)

            # respect the training and testing split defined in readme: http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
            training_data = data[np.random.choice(463715, 52500, replace=False), :]
            testing_data = data[-np.random.choice(51630, 22500, replace=False), :]

            pickle_out = open(pickle_location,"wb")
            pickle.dump(np.vstack((training_data,testing_data)), pickle_out)
            pickle_out.close()
        else:
            print('loading pickled dataset')
            data = pickle.load(open(pickle_location,"rb"))
            # allocate training data (use k fold cross validation to tune hyperparameters)
            training_data = data[:52500,:]

            # hold out testing data till end
            testing_data = data[52500:,:]


    x_train = training_data[:,1:]
    y_train = training_data[:,0]

    x_test = testing_data[:,1:]
    y_test = testing_data[:,0]

    if show_label_distribution:
        # visualize training vs testing data via histograms
        plt.ion()
        plt.hist([y_train, y_test], bins='auto', label='Training', histtype='bar')
        plt.show()

    return (x_train, y_train, x_test, y_test)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=3)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

def main():
    # choose which dataset(s) we are going to use ('gym' or 'song' prediction)
    for prob_name in [
        'gym',
        'song'
        ]:

        p = global_params[prob_name]
        folds = 5

        print('\n##### Learning on {0} dataset #####\n'.format(prob_name))
        x_train, y_train, x_test, y_test = PreProcess(prob_name, False)


        learning_algos = [
            (
                DecisionTreeClassifier(
                    max_depth= p['max_depth'],
                    min_samples_split = p['min_samples_split'],
                ),
                True, # Normalize Inputs
                { # Grid Search Params
                    'max_depth':range(1,30,2),
                    'min_samples_split':range(2,400,20)
                },
                # chosen params
                dict(max_depth= p['max_depth'], min_samples_split = p['min_samples_split'])
            ),
            (
                KNeighborsClassifier(
                    n_neighbors=p['n_neighbors'],
                    weights=p['weights']
                ),
                True, # Normalize Inputs
                { # Grid Search Params
                    'n_neighbors':[1] + [x*5 for x in range(1,9)],
                    'weights':['distance', 'uniform']
                },
                # chosen params
                dict(n_neighbors=p['n_neighbors'], weights=p['weights'])
            ),
            (
                AdaBoostClassifier(
                    n_estimators=p['n_estimators'],
                    learning_rate=p['learning_rate']
                ),
                True, # Normalize Inputs
                { # Grid Search Params
                    'n_estimators':[100, 250, 400, 550, 700],
                    'learning_rate':np.logspace(-5,0,6)
                },
                # chosen params
                dict(n_estimators=p['n_estimators'], learning_rate=p['learning_rate'])
            ),
            (
                MLPClassifier(
                    alpha=p['alpha'],
                    hidden_layer_sizes=p['hidden_layer_sizes'],
                    random_state=p['random_state']
                ),
                True, # Normalize Inputs
                { # Grid Search Params
                    'hidden_layer_sizes': [(x*10,) for x in range(2,7)] + [(x*10,y*10) for x in range(2,7) for y in range(2,7)] + [(x*10,y*10,z*10) for x in range(2,7) for y in range(2,7) for z in range(2,7)],
                    'alpha': [1e-9,5e-8,1e-6,1e-5,1e-4,1e-3]
                },
                # chosen params
                dict(alpha=p['alpha'],  hidden_layer_sizes=p['hidden_layer_sizes'], random_state=p['random_state'])
            ),
            (
                svm.SVC(
                    C = p['C'],
                    gamma = p['gamma'],
                    kernel=p['kernel']
                ),
                True, # Normalize Inputs
                [{ # Grid Search Params
                    'kernel':['rbf'],
                    'C': np.logspace(-3, 3, 7),
                    'gamma':np.logspace(-3, 3, 7)
                    },{
                    'kernel': ['linear'],
                    'C': np.logspace(-3, 3, 7),
                }],
                # chosen params
                dict(C = p['C'],  gamma = p['gamma'], kernel=p['kernel'])
            )
        ]

        learning_algos_chosen = {
            'Decision Tree': learning_algos[0],
            'K-Nearest Neighbors': learning_algos[1],
            'Boosting': learning_algos[2],
            'Neural Network': learning_algos[3],
            'Support Vector Machine': learning_algos[4]
        }

        skf = StratifiedKFold(n_splits=folds)

        plotting_learning_curve, grid_search, run_cv, testing = (False, False, False, False)
        testing = True;

        for algoName, (estimator, normalize_data, param_grid, params) in learning_algos_chosen.items():
            print('\n{0} Performance\n'.format(algoName))

            if normalize_data:
                # Normalize for less sensitivity: https://www.springboard.com/blog/beginners-guide-neural-network-in-python-scikit-learn-0-18/
                scaler = StandardScaler()
                scaler.fit(x_train)
                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)

            if plotting_learning_curve:
                plot_learning_curve(estimator, '{0} - {1}'.format(algoName, prob_name.upper()), x_train,y_train, train_sizes=np.linspace(.1, 1.0, 10), cv=skf)
                plt.ion()
                plt.savefig('{0}_{1}.png'.format(algoName, prob_name.upper()))
                plt.pause(0.001)
                plt.show()

            if grid_search:
                clf = None
                grid_pickle_loc = './Grid_Search_{0}_{1}'.format(prob_name,algoName)

                if not os.path.exists(grid_pickle_loc):
                    print('generating grid search results')
                    # grid search and then pickle the cv_results
                    clf = GridSearchCV(estimator, param_grid, verbose = 3, cv=folds)
                    clf.fit(x_train, y_train)
                    results = clf
                    pickle_out = open(grid_pickle_loc,"wb")
                    pickle.dump(results, pickle_out)
                    pickle_out.close()
                else:
                    print('loading pickled grid search results')
                    clf = pickle.load(open(grid_pickle_loc,"rb"))
                    import ipdb; ipdb.set_trace()


            if run_cv:
                # cross validation
                print(params)
                scores = cross_val_score(estimator, x_train, y_train, cv=folds, verbose=3)
                print("\tCross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

            if testing:
                # testing on held out test set
                print(params)
                t1 = time.time()
                estimator.fit(x_train, y_train)
                t2 = time.time()
                avg_runtime = str(datetime.timedelta(seconds=((t2-t1)/folds)))
                print('Wall Clock Time: {0}'.format(avg_runtime))
                y_predict = estimator.predict(x_test)
                print("\tTest Set Accuracy:{0}".format(np.count_nonzero(y_predict == y_test) / len(y_test)))


if __name__ == '__main__':
    main()