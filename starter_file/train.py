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
import glob
from labtransformer import LabTransformer

# Add arguments to script
parser = argparse.ArgumentParser()

parser.add_argument('--n_neighbors', type=int, default=10, 
help="Number of neighbors to consider")
parser.add_argument('--weights',  type=int, choices=range(2), 
default=0, help="Sample weighting method")
parser.add_argument('--embedding', type=int, choices=range(3), 
default=0, help="Type of feature embedding")
parser.add_argument('--data_folder', type=str, default='data', help="Data folder mounting point")

args = parser.parse_args()

weights_dict = {0:'uniform', 1:'distance'}
embedding_dict = {0:'none', 1:'lab', 2:'nca'}

# Fetch the data
#df = pandas.read_csv(
#"https://raw.githubusercontent.com/zgoey/azure_ml_capstone/master/color_shades.csv")
data_path = glob.glob(args.data_folder, recursive=True)[0]
df = pandas.read_csv(data_path)

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

# Fit model
model.fit(x,y)

# Save the model
os.makedirs('outputs', exist_ok=True)
joblib.dump(model, 'outputs/model.pkl')

