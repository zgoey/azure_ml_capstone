*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Color shade classification

In this project we try to build a color shade classifier for RGB-triplets that can help colorblind people determine what color they are looking at. We train classifiers using AutoML and hyperparameter optimization and deploy the best model to a webservice that can then be accessed by a color shae app to determine the shade of a color mpatch. This last step, hpwever, is outside the scope of this project, which focuses on the machine learning part.

## Project Set Up and Installation
The project consists of two notebooks, automl.ipynb and hyperparameter_tuning.ipynb, which run by default on any standard compute in AzureML. For the deployed webservice that is created in one of the notebooks, a .yml-file is included that defines the environment that is required to run the service.

## Dataset

### Overview
The dataset that we use can be downloaded from: https://github.com/zgoey/azure_ml_capstone/blob/master/color_shades.csv. 
Each data sample contains of a RGB-triplet and an associated basic color shade. The shades that we discern are 
white, black, grey, yellow, red, blue, green, brown, pink, orange and purple. The notebook automl.ipynb expplians in detail how this dataset was generated.

### Task
We are going to determine the color shae of color patches using the dataset above. The features that we are going to use for this are the values of the red, green and blue channel of the color patch.

### Access
Both notebooks contain code to upload the data from its Web location (or, to be precise, from https://raw.githubusercontent.com/zgoey/azure_ml_capstone/master/color_shades.csv). After uploading the dataset is registered in Azure and from then on, all the code can access it from there.

## Automated ML
In our AutoML experiment, we set the task to classification, with the target column is set to "Shade", since that is what we wish to predict. We choose accuracy as our primary metric and apply 5-fold cross-validation to be enhance the stability of its estimate. To be sure that we do not run our experiment forever, we limit the time that the experiment will run to 1 hour. Concurrency is maximally used bu setting the maximum number of concurrent iterations to four, which is the maximum that our compute cluster can deliver.

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
For the hyperparameter tuning, we use  a k-nearest neighbor model, because it is simple and at the same time flexible enough to capture complicated decision boundaries. Using Bayesian parameter sampleing, we try to optimize over three parameters:
 1. Number of neighbors (range = {1,2,...,100})
 2. Neighbor voting weigts (range = {'uniform', 'distance'})
 3. Embedding preceding neighbor search (range = {'none', 'lab', 'nac'}
Here 'lab' stands for an embedding in the (roughly) perceptually uniform  L\*a\*b\* color space, whereas 'nac' stands for Neighborhood Components Analysis. More details about these embeddings can be found in the notebook hyperparameter_tuning.ipynb.


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
