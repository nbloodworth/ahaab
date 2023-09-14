'''
AHAAB handle_input submodule
Part of the AHAAB tools module

ahaab/src/
└──tools/
    └──handle_input.py

Submodule list:
    === handle_featurize_input ===
    === handle_training_input  ===
    ===  handle_predict_input  ===
    ===   check_feature_list   ===
'''
# AHAAB libraries
from tools import formats
from tools.utils import standardize_feature_data
from tools.data_loaders import AhaabAtomDataset
from features import atom_features
from features.atom_features import bin_default
from features.atom_features import scope_default

# Pandas
import pandas as pd

# Numpy
import numpy as np

# Python
from pathlib import Path
import os
from inspect import getmembers,isfunction
import sys

def handle_featurize_input(input_data):
    '''
    Usage:
    $ file_list=handle_featurize_input(*args)

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

def handle_training_input(input_data, k_fold, pkd_values="pkd", train_all=False):
    '''
    Usage:
    $ handle_training_input(*args)

    Positional arguments:
    > input_data: CSV file containing feature-level data
    > k_fold: number of training/testing sets for k-fold cross-
              validation 
    
    Keyword arguments:
    > pkd_values: CSV file containing pkd values in the
                  same index order as feature-level data,
                  or the column name in the feature-level
                  data that contains pkd/Kd values.
    > train_all:  Boolean to indicate if model should be
                  trained with all available data.

    Outputs:
    > List of tuples with the following format:
        [(training_dataset, training_index, testing_dataset, testing_index)...] --> includes k_fold tuples total.
        training_dataset: pytorch dataset with data used to train
        model for that iteration.
        testing_dataset: pytorch dataset used to test model.
        "_index" values can be used to cross-reference actual pKd
        values from the original dataset labels for accuracy 
        correlation.
    > Integer indicating the number of features in the featurized
      dataset. Passed to train_ahaab.
    '''

    input_data=os.path.abspath(input_data)
    if not Path(input_data).is_file():
        formats.error(f"Unable to locate feature data file: {input_data}")
        return

    # Read and sanitize feature data file and pKd values
    feature_data=pd.read_csv(input_data)
    formats.notice(f"Building training/testing sets from {input_data}")
    # Check case where pkd_values is a column name in the features dataset (if user passed label data in features)
    if pkd_values not in feature_data.columns.tolist():
        # Next case: the user passed label data as a separate file
        pkd_filepath=os.path.abspath(pkd_values)
        if not Path(pkd_filepath).is_file():
            # If labels are not in the feature data or a valid file then raise an error
            formats.error(f"Unable to locate column label {pkd_values} or file {pkd_filepath} containing pKd values")
            return
        else:
            pkd=pd.read_csv(pkd_filepath)
            # Make sure we don't have a mismatch between number of feature vectors and number of pKd values
            if len(feature_data)!=len(pkd):
                formats.error(f"Unequal number of featurized models and pKd values!")
                return
            # Test following scenarios:
            if len(pkd.columns)>1:
                # File data has more than one column. If so, check if column label is present - if so, take those values. If not, take column 0 by default. Make case invariant to be generous and ensure we find a likely match.
                if "pkd" not in [x.lower() for x in pkd.columns.tolist()]:
                    formats.warning(f"More than 1 column of data found in file containing pKd values, and no label for 'pKd' found. Will use first column by default.")
                    pkd=pkd.iloc[:,0]
                else:
                    pkd_col_label=pkd.columns[pkd.columns.tolist().index([x for x in pkd.columns.tolist() if x.lower()=="pkd"])]
                    formats.warning(f"More than 1 column of data found in file containing pKd values. Using column labled {pkd_col_label}")
                    pkd=pkd[pkd_col_label]
    else:
        pkd=feature_data[pkd_values]
        feature_data.drop(labels=pkd_values,inplace=True)

    # Now remove any nan pKd values and corresponding feature data rows:
    if pkd.isna().any().values[0]:
        bad_features=len(feature_data[pkd.isna().values])
        feature_data=feature_data[~pkd.isna().values]
        feature_data=feature_data.reindex(
            labels=range(0,len(feature_data)),
            axis=0
        )
        pkd=pkd[~pkd.isna().values]
        print(f"pKd values missing for {bad_features} featurized peptide/HLA complexes. Vectors removed from training set(s)")
    # Check for a valid value for k_fold:
    if k_fold<1 or k_fold>=len(feature_data):
        formats.error(f"User specified {k_fold}-fold training/testing splits for k-fold cross validation. k must be an integer >=1 and less than the size of the dataset.")
        return
    
    # Now we split into train/test datasets.
    # Create our index of random values that effectivley assigns each row to a group for testing/training
    # Testing and training features are standardized to each training set.
    if train_all:
        k_fold=1
        kfold_idx=np.zeros(len(feature_data))+1
    else:
        kfold_idx=np.random.randint(0,high=k_fold,size=len(feature_data))
    train_test_groups=[]
    feature_labels=feature_data.iloc[:,1:].columns.tolist()
    num_features=len(feature_labels)
    print(f"Creating {k_fold} training/testing datasets...")
    print("{:^7}{:^12}{:^12}{:^12}".format("Set #","Train size","Test size","Total"))
    # For every testing group, create datasets for both the testing group and the training group by using the kfold_idx variable and masking based on indexed positions in the features dataframe. This method will produce k_fold non-overlapping testing sets (training data may overlap, but testing data will not).
    for k in range(0,k_fold):
        msk=kfold_idx==k
        train_data_std=pd.DataFrame(
            standardize_feature_data(feature_data.iloc[:,1:][~msk].to_numpy()),
            columns=feature_labels
            )
        train_group=AhaabAtomDataset(train_data_std,pkd[~msk])
        train_index=pkd[~msk].index
        if not train_all:
            test_data_std=pd.DataFrame(
                standardize_feature_data(feature_data.iloc[:,1:][msk].to_numpy()),
                columns=feature_labels
                )
            test_group=AhaabAtomDataset(test_data_std,pkd[msk])
            test_index=pkd[msk].index
        else:
            test_group=[]
            test_index=[]

        train_test_groups.append(
            (train_group,
             train_index,
             test_group,
             test_index)
            )
        print(f"{k:^7}{len(pkd[~msk]):^12}{len(pkd[msk]):^12}{len(pkd[~msk])+len(pkd[msk]):^12}")

    return train_test_groups,num_features

def handle_predict_input(input_data,test=True):
    '''
    Usage:
    $ handle_predict_input(*args)

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

def check_feature_list(feature_list):
    '''
    Usage:
    $ check_feature_list(*args)

    Positional arguments:
    > feature_list: File or list of features to retrieve 

    Outputs:
    > A list of valid features. Raises a warning if invalid
      features are found
    '''
    def parse_features(feature_list=False):
        '''
        Usage:
        $ parse_features(**kwargs)

        Keyword arguments:
        > feature_list: File or list of features to retrieve
                        (defaults to False, indicating all
                        available default features should
                        be used) 

        Outputs:
        > A tuple containing a list of recognized features,
        and a flag indicating that invalid features were
        found.
        '''
        atom_feature_names=[x[0] for x in getmembers(atom_features,isfunction)]
        distance_bins=bin_default
        interface_scope=scope_default

        invalid_features=False
        # Create the default feature list:
        if not feature_list:
            features=[]
            for feature_name in atom_feature_names:
                features.append({
                    "atom_feature": feature_name,
                    }|distance_bins|interface_scope)
            features=pd.DataFrame.from_dict(features)
        else:
            # First case: did the user pass a file?
            feature_keys=["atom_feature"]+list(bin_default.keys())+list(scope_default.keys())
            if Path(os.path.abspath(feature_list[0])).is_file():
                # If so, open the file with pandas and create a dataframe:
                feature_list=os.path.abspath(feature_list[0])
                features=pd.read_csv(feature_list,names=feature_keys,dtype=str)
            else:
                # if not, attempt to parse features from command line input and pass to a dataframe:
                features=pd.DataFrame(columns=feature_keys)
                for f in feature_list:
                    # Parse the input features using hyphen delimiters (specified in the usage instructions)
                    tmp_features=f.split("-")
                    # Index the parsed features to ensure the data length is equal or less than the number of feature keys
                    tmp_features=tmp_features[0:len(feature_keys)]
                    # Index the feature keys in a similar fashion. This method will necessitate that the user passes empty values ("--") for fields they wish to use default values for.
                    tmp_cols=feature_keys[0:len(tmp_features)]
                    tmp_df=pd.DataFrame([tmp_features],columns=tmp_cols)
                    features=pd.concat([features,tmp_df],ignore_index=True)
            # Next, find invalid atom feature names, bin ranges and size, and interface scopes:
            # Invalid names:
            invalid_features=features.loc[~features["atom_feature"].isin(atom_feature_names)]
            features.drop(invalid_features.index,inplace=True)
            invalid_features=invalid_features.assign(Error="Incorrect atom feature name")

            # Invalid bins (non-numeric bins):
            tmp_invalid=features["bin_min"].loc[~features["bin_min"].isnull()].apply(pd.to_numeric,errors="coerce").isna()
            tmp_invalid=features.loc[[i for i,j in tmp_invalid.items() if j==True]]
            features.drop(tmp_invalid.index,inplace=True)
            tmp_invalid=tmp_invalid.assign(Error="Non-numeric minimum bin value")
            invalid_features=pd.concat([
                invalid_features,
                tmp_invalid
            ])

            tmp_invalid=features["bin_max"].loc[~features["bin_max"].isnull()].apply(pd.to_numeric,errors="coerce").isna()
            tmp_invalid=features.loc[[i for i,j in tmp_invalid.items() if j==True]]
            features.drop(tmp_invalid.index,inplace=True)
            tmp_invalid=tmp_invalid.assign(Error="Non-numeric maximum bin value")
            invalid_features=pd.concat([
                invalid_features,
                tmp_invalid
            ])

            tmp_invalid=features["bin_size"].loc[~features["bin_size"].isnull()].apply(pd.to_numeric,errors="coerce").isna()
            tmp_invalid=features.loc[[i for i,j in tmp_invalid.items() if j==True]]
            features.drop(tmp_invalid.index,inplace=True)
            tmp_invalid=tmp_invalid.assign(Error="Non-numeric bin size")
            invalid_features=pd.concat([
                invalid_features,
                tmp_invalid
            ])

            # Bin correction - replace non-assigned bin values with defaults:
            features[["bin_min","bin_max","bin_size"]]=features[["bin_min","bin_max","bin_size"]].apply(pd.to_numeric,errors="coerce")
            features["bin_min"].loc[features["bin_min"].isna()]=bin_default["bin_min"]
            features["bin_max"].loc[features["bin_max"].isna()]=bin_default["bin_max"]
            features["bin_size"].loc[features["bin_size"].isna()]=bin_default["bin_size"]
            
            # Invalid bins (bin_min<0, bin_min>=bin_max, bin_size>=bin_max):
            tmp_invalid=features.loc[(features["bin_min"]<0)|(features["bin_min"]>=features["bin_max"])|(features["bin_size"]>=features["bin_max"])]
            features.drop(tmp_invalid.index,inplace=True)
            tmp_invalid=tmp_invalid.assign(Error="Invalid bin size values")
            invalid_features=pd.concat([
                invalid_features,
                tmp_invalid
            ])
            # Invalid interface scopes:
            # First correct missing values
            features["interface_scope"].loc[features["interface_scope"].isna()]=scope_default["interface_scope"]
            tmp_invalid=features.loc[~features["interface_scope"].isin(["res","pep"])]
            features.drop(tmp_invalid.index,inplace=True)
            tmp_invalid=tmp_invalid.assign(Error="Invalid interface scope")
            invalid_features=pd.concat([
                invalid_features,
                tmp_invalid
            ])

            # Finally, replace missing ("NaN") feature values with defaults:
            features.loc[features["interface_scope"].isna()]=scope_default["interface_scope"]
            features.loc[features["bin_min"].isna()]=bin_default["bin_min"]
            features.loc[features["bin_max"].isna()]=bin_default["bin_max"]
            features.loc[features["bin_size"].isna()]=bin_default["bin_size"]
            
            # If invalid features are found, then write the sanitized features to a file for the user to review before a second call, and set the invalid_features flag.
            if not invalid_features.empty:
                sanitized_features_filename=os.path.abspath("AHAAB_feature_list.csv")
                formats.warn(f"Invalid features found. Valid features sanitized and written to file {sanitized_features_filename}. Call AHAAB again using the flag --feature_list {sanitized_features_filename} to featurize using sanitized features.")
                print(invalid_features)
                if not features.empty:
                    features.to_csv(sanitized_features_filename,index=False,header=False)
                else:
                    formats.error(f"No valid features found! File {sanitized_features_filename} not created. Call 'ahaab.py -f help --feature_list' to view usage instructions for creating customized features.")
                invalid_features=True
            else:
                invalid_features=False

        return features,invalid_features 
    
    def feature_list_help():
        '''
        Usage:
        $ feature_list_help()

        Outputs:
        > A formatted string listing 
        '''
        helpdoc = '''
A guide to using the AHAAB feature list generation tool.

AHAAB features are defined by the following:
    1.  Atom feature name
        The atom-based feature in question. Atom feature
        names correspond to the name of the function in 
        the features.atom_features submodule used to generate
        them. Currently, the following atom feature names
        are available:

        {}

    2.  Distance bins
        Describes the discretized distance bins in which each
        feature is calculated. Essentially, a distribution of
        features across an interface between atoms. Bins are 
        described by 3 quantities: (1) minimum distance (in
        angstroms); (2) maximum distance; (3) bin size.
        The default bins settings are:

        {}
    
    3.  Interface scope
        How to calculate the features. Default is to perform
        featurization on each residue in the peptide ("res"). 
        Alternatively, this can be over the entire peptide
        ("pep").

AHAAB features are defined using the above 3 criteria using
a string, each criteria separated by a '-' delimiter.
The general syntax is:

> atom_property-min_distance-max_distance-bin_size-interface_scope

For example:

> atom_hbonds-0-10-1-res

Will generate a count of hydrogen bond donor/acceptor pairs
at each 1 angstrom bin between 0 and 10 angstroms, at each
residue position. 
To use default feature settings, leave the space between
the delimiteres for the aspect in question blank. To use

> atom_hbonds----res

A feature list may also be passed as a CSV, with each feature-
bins-interface scope paring on a separate line. For example,
consider the contents of the file feature_list.txt:

> atom-hbonds,,,,pep
> atom-hbonds,2,5,0.25,res

This file will calculate the number of hydrogen bond donor/
acceptor pairs at default distance bins summed over the
entire peptide, and also at every 0.25 angstroms between
2 and 5 angstroms distance for each atom in each residue.
        '''.format(
            "\n        ".join([str(i+1)+". "+str(x[0]) for i,x in enumerate(getmembers(atom_features,isfunction))]),
            "\n        ".join([str(i)+": "+str(bin_default[i]) for i in bin_default])
            )
        return helpdoc
    
    if not feature_list:
        formats.notice("Guide to using AHAAB atom features:")
        print(feature_list_help())
        return
    elif feature_list[0]=="default":
        formats.message("No feature list found. Using default features and settings.")
        parsed_feature_list=parse_features()
    else:
        parsed_feature_list=parse_features(feature_list=feature_list)
        formats.message(f"Features succesfully parsed:\n{parsed_feature_list[0]}")

    return parsed_feature_list