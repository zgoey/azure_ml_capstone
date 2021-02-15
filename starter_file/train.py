import argparse
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.base import BaseEstimator, TransformerMixin
from azureml.core.run import Run
from azureml.core import Dataset, Workspace
import numpy as np
import pandas
from skimage import color
import joblib
import os
       
    
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_neighbors', type=int, default=10, 
    help="Number of neighbors to consider")
    parser.add_argument('--weights',  type=int, choices=range(2), 
    default=0, help="Sample weighting method")
    parser.add_argument('--embedding', type=int, choices=range(3), 
    default=0, help="Type of feature embedding")
    
    args = parser.parse_args()
    
    class LabTransformer():
        # here you define the operation it should perform
        def transform(self, X, y=None, **fit_params):
            return color.rgb2lab(X.astype(np.uint8)).astype(np.float32)

        # just return self
        def fit(self, X, y=None, **fit_params):
            return self

    
    weights_dict = {0:'uniform', 1:'distance'}
    embedding_dict = {0:'none', 1:'lab', 2:'nca'}
    
    # Fetch the data
    df = pandas.read_csv(
    "https://raw.githubusercontent.com/zgoey/azure_ml_capstone/master/color_shades.csv")
    #ws = Workspace.from_config()
    #df = Dataset.get_by_name(workspace, 'color_shades').to_pandas_dataframe()

    # Separate features and target
    x = df[['Red','Green','Blue']].to_numpy()
    le = LabelEncoder()
    y = le.fit_transform(df['Shade'])

    # Setup the run
    run = Run.get_context()
    run.log("Number of neighbors:", np.int(args.n_neighbors))
    run.log("Sample weights:", weights_dict[args.weights])
    run.log("Feature embedding:", embedding_dict[args.embedding])
    
    neigh = KNeighborsClassifier(n_neighbors=args.n_neighbors, 
    weights=weights_dict[args.weights])
    
    # Setup the model
    if embedding_dict[args.embedding] == 'lab':
        lab = LabTransformer()
        model = Pipeline([('embedder', lab), ('knn', neigh)])
    elif embedding_dict[args.embedding] == 'nca':
        nca = NeighborhoodComponentsAnalysis()
        model = Pipeline([('embedder', nca), ('knn', neigh)])
    else:
        model = neigh

    # Log the accuracy
    cv_results = cross_validate(model, x, y, cv=5)
    accuracy = np.mean(cv_results['test_score'])
    run.log("Accuracy", np.float(accuracy))
    
    # Save the model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.pkl')


if __name__ == '__main__':
    main()
