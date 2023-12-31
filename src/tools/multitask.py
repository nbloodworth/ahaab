'''
AHAAB multitask submodule
Part of the AHAAB features module

ahaab/src
└──tools
    └──multitask.py

Submodule list:

    batch_files
    recombine_features
    multiprocess_batches
'''

# AHAAB module imports
from tools import formats
from features.get_features import get_features_atom 

# pandas
import pandas as pd

# Python base libraries
import math
import multiprocessing
from pathlib import Path
import os

def batch_files(file_list):
    '''
    Usage:
    $ batch_files(*args,**kwargs)

    Positional arguments:
    > file_list: List of files to batch

    Keyword arguments:

    Returns:
    > A list of lists, where each element
    corresponds to a list of files to process
    in a different subroutine
    '''

    num_batches=multiprocessing.cpu_count()-1
    num_files=len(file_list)
    batch_size=math.ceil(num_files/num_batches)
    batch_list=[]
    for i in range(0, len(file_list), batch_size):
        if i+batch_size>num_files:
            batch_list.append(file_list[i:])
        else:
            batch_list.append(file_list[i:i+batch_size])

    formats.notice(f"Preparing to multitask featurization:")
    formats.message(f"{len(batch_list)} batches created\n{batch_size} files per batch")
    return batch_list

def recombine_features(batch_suffix):
        '''
        Usage:
        $ recombine_features(*args,**kwargs)

        Positional arguments:
        > batch_suffix: List of suffix values generated
                        by multiprocess_batches

        Keyword arguments:

        Returns:
        > Combines features and metadata files generated
          by multiprocess_batches
        '''

        # Given that the size of the json file would be immense, we will only combine feature data for now...
        # Append new feature data to old AHAAB file if present
        if Path("AHAAB_atom_features.csv").is_file():
            out_feature_data=pd.read_csv("AHAAB_atom_features.csv")
            formats.warning("File named 'AHAAB_atom_features.csv' detected. File will be appended with new feature data.")
        else:
            out_feature_data=pd.DataFrame()

        out_metadata=[]
        for s in batch_suffix:
            if Path(f"AHAAB_atom_features_{s}.csv").is_file():
                tmp_data=pd.read_csv(f"AHAAB_atom_features_{s}.csv")
                out_feature_data=pd.concat([out_feature_data,tmp_data],ignore_index=True)
                os.remove(f"AHAAB_atom_features_{s}.csv")

        formats.notice("Recombining featurized data from multitask operation")
        out_feature_data.to_csv(f"AHAAB_atom_features.csv",index=False)

        return out_feature_data

def multiprocess_batches(batch_list, feature_list, get_metadata=False):
    '''
    Usage:
    $ multiprocess_batches(*args,**kwargs)

    Positional arguments:
    > batch_list: List of file batches from
                  batch_files
    > feature_list: List of ahaab features
                    to generate

    Keyword arguments:

    Returns:
    > Given a list of batches and features, uses multiprocessing
      to produce an AHAAB_atom_features.csv file. Does not currently
      work on Windows machines.
    '''

    # Get file suffixes and assign to pool
    batch_suffix=[str(i) for i in range(0,len(batch_list))]
    pool=multiprocessing.Pool(multiprocessing.cpu_count()-1)
    for d,suf in zip(batch_list,batch_suffix):
        pool.apply_async(get_features_atom, (d,feature_list),dict(output_suffix=suf,toprint=False,get_metadata=get_metadata))
    pool.close()
    pool.join()

    feature_dataframe=recombine_features(batch_suffix)

    formats.notice("Featurization complete!")
    formats.message("Features written to AHHAB_atom_features.csv")
    if get_metadata:
        formats.message("Feature metadata written to AHAAB_atom_features_metadata.json")
    
    return feature_dataframe