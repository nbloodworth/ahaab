'''
AHAAB manage_data submodule
Part of the AHAAB tools module

ahaab/src/
└──data
    └──manage_data.py

Submodule list:
    
    commit_features
    commit_weights
'''

# AHAAB module imports
from tools import formats

# pandas
import pandas as pd

# pytorch
import torch

# python
import shutil

def commmit_features(features):
    '''
    Usage:
    $ commit_features(*args)

    Positional arguments:
    > features: pandas dataframe containing feature data 
    
    Keyword arguments:

    Outputs:
    > Writes the file 'AHAAB_atom_features.csv' to 
      ahaab/src/data/features. If the file exists, attempts to
      append existing data.
    '''
    features_path=os.path.abspath(os.path.join(Path(__file__).parents[0],"data","features"))
    features_file=os.path.join(features_path, "AHAAB_atom_features.csv")
    Path(features_path).mkdir(parents=True,exist_ok=True)
    # Check if a features file already exists
    if Path(features_file).is_file():
        formats.notice(f"Existing features file detected in {features_path}! Appending data...")
        old_features=pd.read_csv(features_file)
        # Attempt to append data, but abort attempt if feature columns differ 
        if len(old_features.columns.difference(features.columns))==0:
            features=pd.concat([old_features,features])
        else:
            formats.error(f"Different feature labels for pre-existing features file and new features file, unable to combine. Manually delete old features file in ahaab/src/data/features to overwrite.")
            return
    features.to_csv(features_file)

    return

def commit_weights(model_filenames):
    '''
    Usage:
    $ commit_weights(*args)

    Positional arguments:
    > model_filenames: List with the pytorch filename
                       to copy into ahaab/data/weights 
    
    Keyword arguments:

    Outputs:
    > Writes the file 'AHAAB_atom_classifier.pt' to 
      ahaab/src/data/weights. If the file exists, will
      abort.
    '''
    weights_path=os.path.abspath(os.path.join(Path(__file__).parents[0],"data","weights"))
    weights_file=os.path.join(weights_path, "AHAAB_atom_classifier.pt")
    Path(weights_path).mkdir(parents=True,exist_ok=True)
    # Check if a features file already exists
    if Path(weights_file).is_file():
        formats.error(f"Existing pytorch file detected in {weights_path}! Manually delete old .pt file in ahaab/src/data/weights to overwrite.")
        return

    shutil.copyfile(model_filenames[0],weights_file)
    
    return