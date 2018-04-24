Package Dependencies
sklearn
numpy
ipdb (optional)
matplotlib

Note: The pickle files provided in my submission are specifically for loading my grid searches for each of the algorithms with each dataset. If not downloaded, my code will grid search from scratch which will indeed take quite a while, trust me.

Global Parameters (see global_params in Algos.py at line 20)

- Specify the download location of the data in the 'data_location' attribute

- To run cross validation, change the 'run_cv' setting to True.

- To alter parameters for any/each of the algorithms, simply change the desired
values for the specified dataset

Gym Data: https://www.kaggle.com/nsrose7224/crowdedness-at-the-campus-gym
Song Data: http://archive.ics.uci.edu/ml/datasets/YearPredictionMSD

For reference, here are the initial values I am providing for global_params:
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