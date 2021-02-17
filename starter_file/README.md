# Color shade classification

In this project we try to build a color shade classifier for RGB-triplets that can help colorblind people determine what color they are looking at. We train classifiers using AutoML and hyperparameter optimization and deploy the best model to a webservice that can then be accessed by a color shade app to determine the shade of a color match. This last step, however, is outside the scope of this project, which focuses on the machine learning part.

## Project Set Up and Installation
The project consists of two notebooks, [automl.ipynb](endpoint_automl.py) and [hyperparameter_tuning.ipynb](hyperparameter_tuning.ipynb), which run by default on any standard compute in AzureML. The only adaptation that we recommend to make to the standard configuration is to update the version of the AzureML SDK to match the one used on the compute clusters, thus preventing warnings when printing out AutoML model details. Another issue to remember is that the notebooks require to have the wokspace configuration (config.json) inside their directories. 

This repository included the environment (.yml) files and scoring scripts needed for deployment. In case of the AutoML experiment these are generated while running the notebook. The environment script for the deployment of the hyperdrive model is also generated within the notebook, but the scoring script [hyperdrive_score.py](hyperdrive_score.py) needs to be uploaded in advance to the directory where [hyperparameter_tuning.ipynb](hyperparameter_tuning.ipynb) resides. 

## Dataset

### Overview
The dataset that we use can be downloaded from: https://github.com/zgoey/azure_ml_capstone/blob/master/color_shades.csv. 
Each data sample contains of a RGB-triplet and an associated basic color shade. The shades that we discern are 
white, black, grey, yellow, red, blue, green, brown, pink, orange and purple. The notebook [automl.ipynb](automl.ipynb) explains in detail how this dataset was generated. 


### Task
The problem that we wish to solve is the determination of the color shade of color patches. The features that we are going to use for this are the values of the red, green and blue channel of the color patch. The notebook automl.ipynb trains AutoML mdoels to carry out this task and the notebook hyperparameter_tuning.ipynb tries to achieve the same using hyperparameter tuning of a k-nearest-neighbor model.

### Access
Both notebooks contain code to upload the data from its Web location (or, to be precise, from https://raw.githubusercontent.com/zgoey/azure_ml_capstone/master/color_shades.csv). Inside the notebooks a FileDataset is created in Azure from this web address. For the AutoML experiment the file is the downloaded locally and converted to a TabularDataset, which can be fed to AutoML. For the Hyperdrive experiment the FileDataset is passed to the training script as a mount.


## Automated ML
In our AutoML experiment, we set the task to classification, with the target column is set to "Shade", since that is what we wish to predict. We choose accuracy as our primary metric and apply 5-fold cross-validation to be enhance the stability of its estimate. To be sure that we do not run our experiment forever, we limit the time that the experiment will run to 1 hour. Concurrency is maximally used by setting the maximum number of concurrent iterations to 4, which is the maximum that our compute cluster can deliver.

### Results
The best AutoML model is a StackEnsemble, which reaches an accuracy of 81.05%. Its meta-learner (shallow view) is given by:

| Learner                                 | Hyperparameters                                | 
| ----------------------------------------|------------------------------------------------|
| LogisticRegressionCV                    | Cs=10, class_weight=None, cv=None, dual=False, |
|                                         | fit_intercept=True, intercept_scaling=1.0,     |
|                                         | l1_ratios=None, max_iter=100,                  |
|                                         | multi_class='auto', n_jobs=None,               |
|                                         | penalty='l2', random_state=None, refit=True    |
|                                         | scoring=<azureml.automl.runtime.stack_ensemble_|
|                                         | base.Scorer object at 0x7f3205acada0>,         |
|                                         | solver='lbfgs', tol=0.0001, verbose=0          |                


Its base learners are:

| Learner                                 | Hyperparameters                                |
| ----------------------------------------|------------------------------------------------|
|                                         | base_score=0.5, booster='gbtree',              |
|                                         | colsample_bylevel=1, colsample_bynode=1,       |
|                                         | colsample_bytree=1, eta=0.3, gamma=0.01,       |
|                                         | learning_rate=0.1, max_delta_step=0,           |
|                                         | max_depth=6, max_leaves=7,                     |
|                                         | min_child_weight=1, missing=nan,               |
|                                         | n_estimators=100, n_jobs=1, nthread=None,      |
|                                         | objective='multi:softprob', random_state=0,    |
|                                         | reg_alpha=1.4583333333333335,                  |
|                                         | reg_lambda=0.625, scale_pos_weight=1,          |
|                                         | seed=None, silent=None, subsample=0.5,         |
|                                         | tree_method='auto', verbose=-10,               |
|                                         | verbosity=0)                                   |
|                                         |                                                |
| StandardScalerWrapper XGBoostClassifier | base_score=0.5, booster='gbtree',              |
|                                         | colsample_bylevel=1, colsample_bynode=1,       |
|                                         | colsample_bytree=1, eta=0.05, gamma=0,         |
|                                         | learning_rate=0.1, max_delta_step=0,           |
|                                         | max_depth=5, max_leaves=0,                     |
|                                         | min_child_weight=1, missing=nan,               |
|                                         | n_estimators=100, n_jobs=1, nthread=None,      |
|                                         | objective='multi:softprob', random_state=0,    |
|                                         | reg_alpha=1.6666666666666667,                  |
|                                         | reg_lambda=0.9375, scale_pos_weight=1,         |
|                                         | seed=None, silent=None, subsample=0.6,         |
|                                         | tree_method='auto', verbose=-10,               |
|                                         | verbosity=0                                    | 
|                                         |                                                |
| StandardScalerWrapper XGBoostClassifier | base_score=0.5, booster='gbtree',              |
|                                         | colsample_bylevel=1, colsample_bynode=1,       |
|                                         | colsample_bytree=1, eta=0.001, gamma=1,        |
|                                         | learning_rate=0.1, max_delta_step=0,           |
|                                         | max_depth=6, max_leaves=31,                    |
|                                         | min_child_weight=1, missing=nan,               |
|                                         | n_estimators=100, n_jobs=1, nthread=None,      |
|                                         | objective='multi:softprob', random_state=0,    |
|                                         | reg_alpha=1.0416666666666667,                  |
|                                         | reg_lambda=2.0833333333333335,                 |
|                                         | scale_pos_weight=1, seed=None, silent=None,    |
|                                         | subsample=0.8, tree_method='auto', verbose=-10,|
|                                         | verbosity=0                                    |
|                                         |                                                |
| StandardScalerWrapper XGBoostClassifier | base_score=0.5, booster='gbtree',              |
|                                         | colsample_bylevel=0.9, colsample_bynode=1,     |
|                                         | colsample_bytree=1, eta=0.4, gamma=0,          |
|                                         | learning_rate=0.1, max_delta_step=0,           |
|                                         | max_depth=6, max_leaves=7,                     |
|                                         | min_child_weight=1, missing=nan,               |
|                                         | n_estimators=100, n_jobs=1, nthread=None,      |
|                                         | objective='multi:softprob', random_state=0,    |
|                                         | reg_alpha=0.8333333333333334,                  |
|                                         | reg_lambda=2.291666666666667,                  |
|                                         | scale_pos_weight=1, seed=None, silent=None,    |
|                                         | subsample=1, tree_method='auto', verbose=-10,  |
|                                         | verbosity=0                                    |
|                                         |                                                |
|                                         |                                                |
| StandardScalerWrapper LogisticRegression| C=1048.1131341546852, class_weight=None,       |
|                                         | dual=False, fit_intercept=True,                |
|                                         | intercept_scaling=1, l1_ratio=None,            |
|                                         | max_iter=100, multi_class='multinomial',       |
|                                         | n_jobs=1, penalty='l2', random_state=None,     |
|                                         | solver='lbfgs', tol=0.0001, verbose=0,         |
|                                         | warm_start=False                               |
|                                         |                                                |
| StandardScalerWrapper LogisticRegression| C=51.79474679231202, class_weight=None,        |
|                                         | dual=False, fit_intercept=True,                |
|                                         | intercept_scaling=1, l1_ratio=None,            |
|                                         | max_iter=100, multi_class='multinomial',       |
|                                         | n_jobs=1, penalty='l1', random_state=None,     |
|                                         | solver='saga', tol=0.0001, verbose=0,          |
|                                         | warm_start=False                               |
|                                         |                                                |
| StandardScalerWrapper LogisticRegression| C=75.43120063354607, class_weight=None,        |
|                                         | dual=False, fit_intercept=True,                |
|                                         | intercept_scaling=1, l1_ratio=None,            |
|                                         | max_iter=100, multi_class='multinomial',       |
|                                         | n_jobs=1, penalty='l2', random_state=None,     |
|                                         | solver='saga', tol=0.0001, verbose=0,          |
|                                         | warm_start=False                               |
|                                         |                                                |
| StandardScalerWrapper XGBoostClassifier | base_score=0.5, booster='gbtree',              |
|                                         | colsample_bylevel=1, colsample_bynode=1,       |
|                                         | colsample_bytree=1, eta=0.05, gamma=0,         |
|                                         | learning_rate=0.1, max_delta_step=0,           |
|                                         | max_depth=6, max_leaves=0,                     |        
|                                         | min_child_weight=1, missing=nan,               |
|                                         | n_estimators=200, n_jobs=1, nthread=None,      |
|                                         | objective='multi:softprob', random_state=0,    |
|                                         | reg_alpha=0.625,                               |
|                                         | reg_lambda=0.8333333333333334,                 |
|                                         | scale_pos_weight=1, seed=None, silent=None,    |
|                                         | subsample=0.8, tree_method='auto',             |
|                                         | verbose=-10, verbosity=0                       |
|                                         |                                                |
| RobustScaler LightGBMClassifier         | boosting_type='gbdt', class_weight=None,       |
|                                         | colsample_bytree=0.8911111111111111,           |
|                                         | importance_type='split',                       |
|                                         | learning_rate=0.07894947368421053,             |
|                                         | max_bin=90, max_depth=3,                       |
|                                         | min_child_samples=326, min_child_weight=0,     |
|                                         | min_split_gain=0.3157894736842105,             |
|                                         | n_estimators=600, n_jobs=1, num_leaves=56,     |
|                                         | objective=None, random_state=None,             |
|                                         | reg_alpha=0.3157894736842105,                  |
|                                         | reg_lambda=0.21052631578947367, silent=True,   |
|                                         | subsample=0.4457894736842105,                  |
|                                         | subsample_for_bin=200000, subsample_freq=0,    |
|                                         | verbose=-10                                    |

A better model may be found if we allow AutoML to run longer than 1 hour.

Below, is a screenshot of the RunDetails widget created while running auto_ml.ipynb:

 ![image](automl_run_details.png)
 
After registration, the best model looks like this in Azure ML Studio:
 ![image](automl_best_model.png)
 
## Hyperparameter Tuning
For the hyperparameter tuning, we use  a k-nearest-neighbor model, because it is simple and at the same time flexible enough to capture complicated decision boundaries. Using Bayesian parameter sampling, we try to optimize over three parameters:
 1. Number of neighbors (range = {1,2,...,100})
 2. Neighbor voting weigts (range = {'uniform', 'distance'})
 3. Embedding preceding neighbor search (range = {'none', 'lab', 'nac'}
Here 'lab' stands for an embedding in the (roughly) perceptually uniform  L\*a\*b\* color space, whereas 'nac' stands for Neighborhood Components Analysis. More details about these embeddings can be found in the notebook hyperparameter_tuning.ipynb.

In the hyperparameter tuning procedure, we use Bayesian parameter sampling, because our hyperparamter sample space is relatively small and we have enough budget to explore it. 
Like in our AutoML experiment, we choose accuracy as our primary metric and apply 5-fold cross-validation to be enhance the stability of its estimate. We do no set an early termination policy, because this is not supported when using Bayesian sampling. The running time, is hoever restricted by setting the maximum number of runs to 100. The maximum number of concurrent runs is set to 1, to let each run benefit fully from previously completed runs, which will enhance the sampling convergence.



### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.


## Model Deployment
We actually deployed both models as can be seen in the notebooks, but since the AutoML model performed slightly better we have only documented that one in detail. It takes in a list of Red-Green-Blue dictionaries and produces a list od color shade strings as a response. The exact way to address the model endpoint can be found in the [automl.ipynb](automl.ipynb) (second cell under "Model Deployment"), or alternatively in [endpoint_automl.py](endpoint_automl.py).

The picture below (taken from the screencast) shows the active model endpoint:

![image](automl_model_endpoint.png)

## Screen Recording
A screencast demoing the AutoML model can be found in https://youtu.be/SowYZMnj0Ik.


