'''
AHAAB validate_input submodule
Part of the AHAAB tools module

ahaab/
└──tools
    └──validate_input.py

Submodule list:
    
    === validate_featurize_input ===
    === validate_training_input  ===
'''
# AHAAB libraries
from tools import formats

# Pandas
import pandas as pd

# Numpy
import numpy as np

# Python base libraries
from pathlib import Path
import os
import sys

def validate_featurize_input(input_data):
    '''
    Usage:
    $ file_list=validate_featurize_input(*args)

    Positional arguments:
    > input_data: single PDB, directory of PDBs,
                  or list of PDBs
    
    Keyword arguments:

    Outputs:
    > List of valid PDB files to process. If
      file_list is empty, no valid files were
      found
    '''
    file_list=[]
    input_data=os.path.abspath(input_data)
    if Path(input_data).is_file():
    # Valid files include a single PDB
        if any(i in input_data for i in [".pdb",".ent",".brk"]):
            file_list.append(input_data)
            print("Single PDB input file detected.")
        else:
    # Or a file containing a list of PDBs
            with open(input_data) as f:
                filedata= [l.strip() for l in f]
            for item in filedata:
                if Path(os.path.abspath(item)).is_file() and any(i in item for i in [".pdb",".ent",".brk"]):
                    file_list.append(os.path.abspath(item))
            print(f"File list detected. {len(filedata)-len(file_list)} invalid file(s) ignored.")
    # Or a directory containing PDBs
    elif Path(input_data).is_dir():
        filedata=os.listdir(input_data)
        for item in filedata:
            if Path(os.path.abspath(os.path.join(input_data,item))).is_file() and any(i in item for i in [".pdb",".ent",".brk"]):
                    file_list.append(os.path.abspath(os.path.join(input_data,item)))
        print(f"PDB-containing directory detected. {len(filedata)-len(file_list)} invalid file(s) ignored.")
    else:
        formats.warn(f"Input file or directory not detected.")

    return file_list

def validate_training_input(input_data, pkd_values="pkd",k_fold=5):
    '''
    Usage:
    $ validate_training_input(*args)

    Positional arguments:
    > input_data: CSV file containing feature-level data 
    
    Keyword arguments:
    > pkd_values: CSV file containing pkd values in the
                  same index order as feature-level data,
                  or the column name in the feature-level
                  data that contains pkd/Kd values.
    > k_fold:     Number of groups for k_fold-fold cross-
                  validation. Default is 5.

    Outputs:
    > List of filenames for training and testing (also 
    creates these files)
    > Returns Null if failure
    '''

    input_data=os.path.abspath(input_data)
    if not Path(input_data).is_file():
        formats.error(f"Unable to locate feature data file: {input_data}")
        return

    if k_fold<1:
        formats.error(f"Unable to create {k_fold} training/testing sets")
        formats.notice("If you wish to create a training set with all available data, pass --kfold 1")
        return

    # Read and sanitize feature data file and pKd values
    feature_data=pd.read_csv(input_data)
    formats.notice(f"Building training/testing sets from {input_data}")
    if pkd_values not in feature_data.columns.tolist():
        pkd_filepath=os.path.abspath(pkd_values)
        if not Path(pkd_filepath).is_file():
            formats.error(f"Unable to locate file containing pKd values: {pkd_filepath}")
            return
        else:
            pkd_values=pd.read_csv(pkd_filepath)
            # Make sure we don't have a mismatch between number of feature vectors and number of pKd values
            if len(feature_data)!=len(pkd_values):
                formats.error(f"Unequal number of featurized models and pKd values!")
                return
            # Test following scenarios:
            if len(pkd_values.columns)>1:
                # File data has more than one column. If so, check if column label is present - if so, take those values. If not, take column 0 by default.
                if "pkd" not in [x.lower() for x in pkd_values.columns.tolist()]:
                    formats.warning(f"More than 1 column of data found in file containing pKd values, and no label for 'pKd' found. Will use column 1 by default.")
                    feature_data["pkd"]=pkd_values[pkd_values.columns[0]]
                else:
                    feature_data["pkd"]=pkd_values[pkd_values.columns[pkd_values.columns.tolist().index([x for x in pkd_values.columns.tolist() if x.lower()=="pkd"])]]
            else:
                feature_data["pkd"]=pkd_values[pkd_values.columns[0]]

    # Now that the input is sanitized, assign training and testing groups
    feature_data["group"]=np.random.randint(k_fold,size=(len(feature_data),1))
    # Write the data to files
    feat_train_filenames=["ahaab_atom_features_"+str(x)+"_train.csv" for x in range(0,k_fold)]
    feat_test_filenames=["ahaab_atom_features_"+str(x)+"_test.csv" for x in range(0,k_fold)]
    pkd_train_filenames=["ahaab_pkd_"+str(x)+"_train.csv" for x in range(0,k_fold)]
    pkd_test_filenames=["ahaab_pkd_"+str(x)+"_test.csv" for x in range(0,k_fold)]
    # Quick comment on the format of 'feature_data':
        # It contains X columns. Column at index 0 is a list of PDB filenames (sans '.pdb' extension) from which the features were derived. Columns 1 -> consist of features in lists, written to a CSV file as strings demarcated by '[ ]' with elements separated by ','. The last two columns are 'pkd' and 'group', which were just added.

    feature_cols=feature_data.columns[1:-2]
    feature_data["all"]=np.empty((len(feature_data),0)).tolist()
    for c in feature_cols:
        feature_data[c]=feature_data[c].apply(lambda s: [float(x.strip(" []")) for x in s.split(",")])
        feature_data["all"]=feature_data["all"]+feature_data[c]

    print("{:^15}{:^20}{:^20}".format("Set Number","Training Set Size","Testing Set Size"))
    if k_fold==1:
        np.savetxt(feat_train_filenames[0],np.vstack(feature_data["all"].to_numpy()),delimiter=",")
        np.savetxt(pkd_train_filenames[0],np.vstack(feature_data["pkd"].to_numpy()),delimiter=",")
        print(f'{i:^15}{len(feature_data["all"]):^20}')
        i=k_fold
    else:
        for i in range(0,k_fold):
            np.savetxt(feat_train_filenames[i],np.vstack(feature_data["all"].loc[feature_data["group"]!=i].to_numpy()),delimiter=",")
            np.savetxt(feat_test_filenames[i],np.vstack(feature_data["all"].loc[feature_data["group"]==i].to_numpy()),delimiter=",")
            np.savetxt(pkd_train_filenames[i],np.vstack(feature_data["pkd"].loc[feature_data["group"]!=i].to_numpy()),delimiter=",")
            np.savetxt(pkd_test_filenames[i],np.vstack(feature_data["pkd"].loc[feature_data["group"]==i].to_numpy()),delimiter=",")
            print(f'{i:^15}{len(feature_data["all"].loc[feature_data["group"]!=i]):^20}{len(feature_data["all"].loc[feature_data["group"]==i]):^20}')
    formats.notice(f"Succesfully wrote {i} training/testing sets and pKd values to {i*4} files")

    return pd.DataFrame({"features_train":feat_train_filenames,"features_test":feat_test_filenames,"pkd_train":pkd_train_filenames,"pkd_test":pkd_test_filenames})

def validate_predict_input(input_data,test=True):
    '''
    Usage:
    $ validate_predict_input(*args)

    Positional arguments:
    > input_data: AHAAB atom feature data 

    Outputs:
    > numpy array where each row corresponds to a
      featurized hla/peptide structure, and a list
      of structure names/labels
    > Returns None if failure
    '''

    input_data=os.path.abspath(input_data)
    if not Path(input_data).is_file():
        formats.error(f"Unable to locate feature data file: {input_data}")
        return

    # Quick line for test
    if test:
        feature_data=np.loadtxt(input_data,delimiter=",")
        rows=feature_data.shape[0]
        feature_labels=[str(x) for x in range(0,rows)]
        return feature_data,feature_labels

    # Read and sanitize feature data file
    feature_data=pd.read_csv(input_data)

    # Remove any columns that contain pKd labels or labels from prior group assignments from randomization (legacy check that may not be necessary later)
    cols=[x.lower() for x in feature_data.columns.tolist()]
    if "pkd" in cols:
        i=cols.index("pkd")
        feature_data.drop(labels=[feature_data.columns.tolist()[i]], inplace=True, axis=1)
    if "group" in cols:
        i=cols.index("pkd")
        feature_data.drop(labels=[feature_data.columns.tolist()[i]], inplace=True, axis=1)

    # Quick comment on the format of 'feature_data':
        # It contains X columns. Column at index 0 is a list of PDB filenames (sans '.pdb' extension) from which the features were derived. Columns 1 -> consist of features in lists, written to a CSV file as strings demarcated by '[ ]' with elements separated by ','.

    feature_cols=feature_data.columns[1:]
    feature_labels=feature_data[feature_data.columns[0]].tolist()
    feature_data["all"]=np.empty((len(feature_data),0)).tolist()
    for c in feature_cols:
        feature_data[c]=feature_data[c].apply(lambda s: [float(x.strip(" []")) for x in s.split(",")])
        feature_data["all"]=feature_data["all"]+feature_data[c]

    feature_data=np.vstack(feature_data["all"].to_numpy())
    
    return feature_data, feature_labels