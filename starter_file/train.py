import argparse
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from azureml.core.run import Run
from azureml.core import Dataset
import numpy as np
import pandas
from matplotlib import colors as mcolors

def lab(x):
    return color.rgb2lab(x.astype(np.uint8)).astype(np.float32)
    
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_neighbors', type=int, default=10, 
    help="Number of neighbors to consider")
    parser.add_argument('--weights', choices=['uniform', 'distance'], 
    default='None', help="Sample weighting method")
    parser.add_argument('--embedding', choices=['none', 'lab', 'nca'], 
    default='None', help="Type of feature embedding")
    
    args = parser.parse_args()
    
    # Fetch the data
    df = pandas.csv_read(
    "https://github.com/zgoey/azure_ml_capstone/blob/master/color_shades.csv")
    #df = Dataset.get_by_name(workspace, 'color_shades').to_pandas_dataframe()

    # Separate features and target
    x = df[['Red','Green','Blue']].to_numpy()
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df['Shade'])

    # Setup the run
    run = Run.get_context()
    run.log("Number of neighbors:", np.int(args.n_neighnors))
    run.log("Sample weights:", args.weights)
    run.log("Feature embedding:", args.embedding)
    
    neigh = KNeighborsClassifier(n_neighbors=args.n_neighbors, 
    weights=args.weights)
    
    # Setup the model
    if args.embedding == 'lab':
        model = Pipeline([('embedder', lab), ('knn', neigh)])
    elif args.embedding == 'nca':
        nca = NeighborhoodComponentsAnalysis()
        model = Pipeline([('embedder', nca), ('knn', neigh)])
    else:
        model = neigh

    # Log the accuracy
    cv_results = cross_validate(model, x, y, cv=5)
    accuracy = np.mean(cv_results['test_score'])
    run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
