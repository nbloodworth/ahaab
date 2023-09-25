'''
AHAAB scripts
Generate pKd File

Usage examples:

    1. Generate file of indexed-matched pKd values for an existing
       AHAAB features file named "AHAAB_atom_features.csv":

       $ python gen_pkd_file.py --key AHAAB_atom_features.csv

       > This command generates 2 files:
            1. AHAAB_pKd_classifier.csv
            2. AHAAB_pKd_regression.csv
         
         File (1) is a .csv file containing binary classifications
         for each epitope (0=nonbinder, 1=binder) to be used for
         training the AHAAB binder classifier model.
         File (2) is a .csv file containing log2 scaled pKd values
         for all epitopes. Non-binders (default pKd threshold >= 
         1000nm) are assigned NaN values and can be filtered by
         the user with pandas or numpy.

    2. Use a custom list of epitopes (stored in a file called
       "custom_epitopes.csv", containing a list of IEDB epitope IDs):

       $ python gen_pkd_file.py --key cutom_epitopes.csv

    3. Change the pKd threshold to descriminate binder from non-
       binder:

       $ python gen_pkd_file.py --key custom_epitopes.csv \\
                                --binding_threshold 500

    4. Use a customized download of the IEDB  to search and
       match pKd values (otherwise, attempts to search the most 
       recently available version of the IEDB):

       $ python gen_pkd_file.py --key custom_epitopes.csv \\
                                --binding_threshold 500 \\
                                --iedb custom_iedb.csv 
 '''
# pandas
import pandas as pd

# numpy
import numpy as np

# python
import argparse
import requests
from zipfile import ZipFile
from pathlib import Path
import math
import os
import sys

def filter_iedb(**kwargs):
    '''
    Usage:

        $ filter_iedb()
    
    Keyword arguments: 
    > iedb_data:    Prefiltered iedb database with
                    the "HLA_epitope" column added
    > key_data:     List of HLA/epitope IDs to find
                    pKd values for
    Outputs:
    pandas dataframe containing HLA/epitope data,
    with the following filters applied:
    1. Two records with only one pubmed ID, remove 
       record without 'PMID'
    2. Two records with no pubmed ID and differ by 
       more than 10nm, remove both
    3. Two records, both with pubmed IDs, and differ 
       by less than 10nm, remove the first
    4. Two records, both without pubmed IDs, and 
       differ by less than 10nm, remove the first
    5. Two records all with pubmed IDs not meeting 
       criteria 1-4, remove (and flag for manual 
       review)
    6. >2 records, keep record(s) with pubmed IDs.
       If remaining records differ, remove all 
       (flag for manual review). Otherwise, keep 1 
       and remove the rest.
    7. >2 records and no pubmed IDs, keep first if
       all the same. Otherwise, remove and flag.
    '''
    iedb_data=kwargs["iedb_data"]
    key_data=kwargs["key_data"]

    iedb_data.insert(0,"Flag",False) # Set a flag value here in case we need to do a manual review
    iedb_data_duplicated=iedb_data[iedb_data.duplicated("HLA_epitope",keep=False)==True]
    if iedb_data_duplicated.empty:
        print("No HLA/epitope combindations with duplicate pKd values located")
        return iedb_data
    else:
        print(f"{len(iedb_data_duplicated)} HLA/epitope combinations with duplicate records located")
        iedb_data=iedb_data[iedb_data.duplicated("HLA_epitope",keep=False)==False]
        enames = iedb_data_duplicated["HLA_epitope"].unique()
        start_len=len(iedb_data_duplicated)

    print("Filtering duplicate epitopes...")
    for e in enames:
        tmp=iedb_data_duplicated.loc[(iedb_data_duplicated['HLA_epitope']==e)]
        # First let's deal with case of two identical records (most common occurance)
        if len(tmp)==2:
        # First case: if only two records with only one having a pubmed ID, remove one without 'PMID'
            if pd.isna(tmp['PMID']).any() and not pd.isna(tmp['PMID']).all():
                iedb_data_duplicated=iedb_data_duplicated.drop(tmp.index[(pd.isna(tmp['PMID']))].tolist()[0])
        # Second case: if both have no pubmed ID and differ by more than 10nm, remove them both
            elif pd.isna(tmp['PMID']).all() and abs((tmp['Quantitative measurement'].iloc[0]-tmp['Quantitative measurement'].iloc[1]))>10:
                iedb_data_duplicated=iedb_data_duplicated.drop(tmp.index)
        # Third case: if both have pubmed IDs and do NOT differ by more than 10nm, remove the first (arbitrary)
            elif ~pd.isna(tmp['PMID']).all() and abs((tmp['Quantitative measurement'].iloc[0]-tmp['Quantitative measurement'].iloc[1]))<10:
                iedb_data_duplicated=iedb_data_duplicated.drop(tmp.index[0])
        # Forth case: if both have pubmedIDs and do not meet these criteria, flag them for review
            elif ~pd.isna(tmp['PMID']).all():
                iedb_data_duplicated.at[tmp.index[0],'Flag']=True
                iedb_data_duplicated.at[tmp.index[1],'Flag']=True
        # Fifth case: if both have no pubmed IDs and differ by less than 10nm, remove the first (arbitrary)
            else:
                iedb_data_duplicated=iedb_data_duplicated.drop(tmp.index[0])

        # More than 2 records:
        elif len(tmp)>2:
            # If at least one record has a pubmed ID, remove the ones without IDs
            if pd.isna(tmp['PMID']).any() and not pd.isna(tmp['PMID']).all():
                iedb_data_duplicated=iedb_data_duplicated.drop(
                    tmp.index[(pd.isna(tmp['PMID']))].tolist()
                    )
                tmp=tmp.drop(tmp.index[(pd.isna(tmp['PMID']))].tolist())
                # Next check if the values remaining are all the same. If not, then flag the remaining records for review
                if len(tmp['Quantitative measurement'].unique())>1:
                    for i in tmp.index.tolist():
                        iedb_data_duplicated.at[i,'Flag']=True
                # If all the values are equal, drop everything but the first (arbitrary)
                else:
                    iedb_data_duplicated=iedb_data_duplicated.drop(tmp.index.tolist()[1::])
            # If all of them have valid PMIDs, then repeat this process:
            elif not pd.isna(tmp["PMID"]).any():
                if len(tmp['Quantitative measurement'].unique())>1:
                    for i in tmp.index.tolist():
                        iedb_data_duplicated.at[i,'Flag']=True
                # If all the values are equal, drop everything but the first (arbitrary)
                else:
                    iedb_data_duplicated=iedb_data_duplicated.drop(tmp.index.tolist()[1::])
            # If none of the records has a pubmed ID and they are all different, remove them
            elif pd.isna(tmp['PMID']).all(): 
                if len(tmp['Quantitative measurement'].unique())>1:
                    iedb_data_duplicated=iedb_data_duplicated.drop(tmp.index.tolist())
            # If none of the records has a pubmed ID and they are all the same, drop everything but the first 
                else:
                    iedb_data_duplicated=iedb_data_duplicated.drop(tmp.index.tolist()[1::])

    
    # Save filtered IEDB data for postprocessing later
    columns_tosave=iedb_data.columns.tolist()
    columns_tosave.remove("Flag")

    # Next, remove the records we flagged for manual review from the remaining duplicates and save to a .csv file for later review
    save_dir=os.getcwd()
    save_filename="IEDB_data.csv"
    print('{} records dropped'.format(start_len-len(iedb_data_duplicated)))
    flagged_records=pd.DataFrame(columns=columns_tosave)
    if iedb_data_duplicated["Flag"].any():
        flagged_records=iedb_data_duplicated.loc[
            (iedb_data_duplicated['Flag']==True)
            ].sort_values(by='HLA_epitope')
        print('{} flagged records'.format(len(flagged_records)))
        iedb_data_duplicated=iedb_data_duplicated.drop(flagged_records.index.tolist())
        flagged_records=flagged_records.drop("Flag",axis=1)
        flagged_fn=os.path.join(save_dir,"flagged_"+save_filename)
        flagged_records.to_csv(flagged_fn, index=False)

    print('{} records remaining'.format(len(iedb_data_duplicated)))
    print(f'Flagged epitope data located in {flagged_fn}')

    # Add the filtered duplicates back to our iedb data
    iedb_data=pd.concat([iedb_data,iedb_data_duplicated],axis=0,ignore_index=True)
    # Make sure we have no duplicates after removal...(sanity check)
    iedb_data_duplicated=iedb_data[iedb_data.duplicated("HLA_epitope",keep=False)==True]
    if not iedb_data_duplicated.empty:
        print(f"WARNING: {len(iedb_data_duplicated)} duplicates remaining after filter!")
        print(iedb_data_duplicated[["HLA_epitope","Quantitative measurement","PMID"]])
    
    if len(iedb_data)!=len(key_data):
        print(f"WARNING: {len(key_data)} HLA/epitope combinations in original key file, but {len(iedb_data)} HLA/epitopes after filtering. pKd values for these HLA/eptiope combinations will be assigned NaN and will be filtered when the data is passed to the training module.")

    return iedb_data[columns_tosave]

def fetch_iedb():
    '''
    Usage:

        $ fetch_iedb()
    
    Arguments: none
    Outputs:
    Filename of the downloaded and extracted iedb.
    '''
    print("Retrieving IEDB copy...")
    # IEDB_DATA_LOC="https://www.iedb.org/downloader.php?file_name=doc/epitope_full_v3.zip"
    IEDB_DATA_LOC="https://www.iedb.org/downloader.php?file_name=doc/mhc_ligand_full_single_file.zip"
    iedb_data_fn=os.path.abspath("mhc_ligand_full.csv")
    try:
        success=True
        test_download=requests.get(IEDB_DATA_LOC)
        if test_download.status_code != 200:
            success=False
    except requests.exceptions.RequestException:
        success=False
    if not success:
        print("Unable to obtain local copy of IEDB. Terminating execution.")
        return False
    else:
        print("Database retrieval successful. Downloading zip file...")
        iedb_data_zip_fn=os.path.join(os.path.dirname(iedb_data_fn),"IEDB_data.zip")
        iedb_data_zip=requests.get(IEDB_DATA_LOC, stream=True)
        with open(iedb_data_zip_fn, mode="wb") as f:
            for chunk in iedb_data_zip.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Extracting IEDB...")
        with ZipFile(iedb_data_zip_fn,"r") as zfile:
            zfile.extractall(path=os.path.dirname(iedb_data_zip_fn))
        os.remove(iedb_data_zip_fn)
        print(f"Local copy of IEDB downloaded to {iedb_data_fn}")
        return iedb_data_fn

def get_iedb_pKd_data(**kwargs):
    '''
    Usage:

        $ get_iedb_pKd_data(**kwargs)
    
    Keyword arguments:
    > key_file:     IEDB epitope IDs to retrieve pKd
                    data for.
    > use_custom_iedb: If true, evaluates to a local copy
                    of the IEDB.
    > scale:        Scale pKd values (log<base> format,
                    default linear)
    > binding_threshold: The pKd value equal to and 
                    above which peptides are classified 
                    as "nonbinders"
    
    Outputs:
    pKd values indexed-matched to the key file provided
    by the user. Writes 2 files:
    1. pKd_<scale>_ahaab_training.csv
       File containing linear or log-scaled pKd values. If
       one of the HLA/epitope combinations was not matched
       with a pKd value from the iedb or provided iedb file,
       a placeholder NaN value will be provided at that index.
    2. pKd_binary_ahaab_training.csv
       File containing binary values to denote "binder" (1)
       or "nonbinder" (0). Binding thresholds can be set by
       the user using the --binding_threshold command line
       argument.
    '''
    key_data=kwargs["key_data"]
    use_custom_iedb=kwargs["use_custom_iedb"]
    scale=kwargs["scale"]
    binding_threshold=kwargs["binding_threshold"]

    # First read in epitope data:
    # If the user provides a local copy of the IEDB, ensure it has the expected fields. If it doesn't, default to using the requests library to find the needed values from the online database.
    required_cols=['PMID',
                   'Epitope IRI',
                   'Name',
                   'Response measured',
                   'Quantitative measurement',
                   'Name.6',
                   'Class'
                   ]
    # Try to infer a header if a custom database file is provided. It should contain the columns listed in required_cols:
    header=0
    header_found=False
    if use_custom_iedb:
        iedb_pKd_data=pd.read_csv(use_custom_iedb,header=header,nrows=10)
        while header<10:
            if len(required_cols)!=len([x for x in required_cols if x in iedb_pKd_data.columns.tolist()]):
                header+=1
            else:
                header_found=True
                break
    
    if not header_found and use_custom_iedb:
        print(f"One or more required fields not found in {use_custom_iedb}.")
        use_custom_iedb=False
        # Set the default header value for the downloaded iedb
        header=1
    elif header_found and use_custom_iedb:
        print(f"Valid iedb file provided")

    # If not using a custom file containing pre-filtered iedb pKd values, we will need to retrieve the iedb and filter it ourselves. Do this with the following function calls. At the end, we will be left with a filtered version of the IEDB that contains all HLA/epitope combinations with quantitative binding data.
    if use_custom_iedb:
        iedb_pKd_data_fn=use_custom_iedb
    else:
        iedb_pKd_data_fn=fetch_iedb()
        header=1

    # Next, read in the iedb data, keeping only values that match with our HLA-epitope ID keys.
    iedb_data=pd.DataFrame()
    chunksread=0
    chunksz=50000
    key_data=key_data.tolist()
    assay_values=[
        'dissociation constant KD (~EC50)',
        'dissociation constant KD (~IC50)',
        'dissociation constant KD'
        ]
    with pd.read_csv(iedb_pKd_data_fn,header=header,chunksize=chunksz,low_memory=False,usecols=required_cols) as reader:
        print("Locating pKd values for HLA-epitope combinations in provided key file...")
        for chunk in reader:
            # Filter to include only HLA-A, -B, and -C alleles, and quantiatative pKd values:
            chunk=chunk.loc[
                (chunk["Class"]=="I") &
                (chunk["Name.6"].str.contains("HLA-A*|HLA-B*|HLA-C*",regex=True)) &
                (chunk["Name.6"].str.contains(":",regex=False)) &
                (chunk["Response measured"].isin(assay_values))
            ]
            # And hla/epitope combination:
            # >>> df["new"]=df["allele"].apply(lambda x: x[0:5]+x[6:8]+x[9:]+"_")+df["IRI"].apply(lambda x: x.split("/")[-1])
            chunk["HLA_epitope"]=chunk["Name.6"].apply(
                lambda x: x[0:5]+x[6:8]+x[9:]+"_"
            ) + chunk["Epitope IRI"].apply(
                lambda x: x.split("/")[-1]
            )
            chunk=chunk.loc[
                chunk["HLA_epitope"].isin(key_data)
            ]
            chunksread+=1
            iedb_data=pd.concat([iedb_data,chunk],axis=0)
            print(f"Epitopes processed: {chunksread*chunksz:<10}Epitopes found: {len(iedb_data):<10}",end="\r")
    print("\nAll epitopes processed")
    if len(iedb_data)==0:
        print(f"No epitopes with values for {assay_values} found in the IEDB!")
        return False
    elif len(iedb_data["HLA_epitope"].unique().tolist())<len(key_data):
        print(f"WARNING: {len(iedb_data['HLA_epitope'].unique().tolist())} HLA/epitope combinations in key file, but only {len(key_data)} located in IEDB!")
    # iedb_data.to_csv("test_iedb.csv",index=False)

    # Now filter the data to remove epitopes with duplicate pKd values
    iedb_data=filter_iedb(
        iedb_data=iedb_data,
        key_data=key_data
        )
    
    # Finally, sort the pKd values by the order they appear in the key file provided by the user
    pKd=[]
    pKd_binary=[]
    if scale!="e" and scale.isnumeric() and int(scale)>1:
        scale=int(scale)
    elif scale!="e" and scale!="1":
        print(f"WARNING: Invalid scale value of {scale}. pKd scale value must be an integer value >1 (1=linear scale, >1 will scale to log base <scale value>)")
        scale=1
    elif scale=="1":
        scale=1
    for key in key_data:
        df_idx=iedb_data.loc[iedb_data["HLA_epitope"]==key].index.tolist()
        # Make sure we find an index (in other words, make sure we have an filtered pKd value from the iedb that matches our desired HLA_epitope combination in the key file)
        if len(df_idx)!=0:
            pKd_val_binary=1
            pKd_val=iedb_data["Quantitative measurement"].iloc[df_idx].values[0]
            if pKd_val>=binding_threshold:
                pKd_val_binary=0
            # Log-scale the pKd value if requested by the user
            if scale=="e":
                pKd_val=math.log(pKd_val)
            elif scale>1:
                pKd_val=math.log(pKd_val,scale)
            pKd.append(pKd_val)
            pKd_binary.append(pKd_val_binary)
        else:
            # Append NaN values as placeholders that can be used to filter out featurized HLA/epitope models without known pKd values in later training.
            pKd.append(np.nan)
            pKd_binary.append(np.nan)

    pKd=pd.DataFrame(pKd)
    pKd_binary=pd.DataFrame(pKd_binary)
    print("pKd values retrieve, filtered, and sorted. Now writing files...")
    # Write our files
    if scale==1:
        scale="linear"
    else:
        scale="log"+str(scale)
    pKd_filename=os.path.abspath(f"pKd_{scale}_ahaab_training.csv")
    pKd_binary_filename=os.path.abspath("pKd_binary_ahaab_training.csv")
    pKd.to_csv(pKd_filename,index=False)
    print(f"{scale} scaled pKd values written to {pKd_filename}")
    pKd_binary.to_csv(pKd_binary_filename,index=False)
    print(f"Binary pKd values written to {pKd_binary_filename}")

    return 
    

parser = argparse.ArgumentParser(description="A python script for generating a .csv file containing pKd values index-matched to values in an existing AHAAB_atom_features.csv file.", epilog="If you use AHAAB for your research, please cite the following manuscript:\n", usage=__doc__)

# ==== Arguments ====
parser.add_argument("--key",nargs=1,required=True,help=".csv file specififying which epiopes in which order should be located. Provide either an AHAAB_atom_features.csv file or .csv column where the first column contains HLA allele and IEDB epitope ID in the following format: HLA-<gene><allele group><protein>_<IEDB epitope ID>. For example: HLA-A0201_109418.")
parser.add_argument("--iedb",nargs=1,default=[False],help="Provide the IEDB database file for matching pKd values. If omitted, gen_pkd_file searches the IEDB database for matches (if internet is available)")
parser.add_argument("--binding_threshold",nargs=1,default=[1000],type=int,help="Threshold pKd value to classify an epitope as a binder or non-binder (linear scale)")
parser.add_argument("--scale",nargs=1,default=["1"],help="Optional argument to log-scale pKd values. Default is 1, for linear (as reported). Input integer >1 to indicate the log base to scale pKd values to. Input e to scale to natural log.")

args=parser.parse_args()

key_file=os.path.abspath(args.key[0])
if not Path(key_file).is_file():
    print(f"Unable to locate {key_file}")
else:
    key_data=pd.read_csv(key_file)
    key_data=key_data.iloc[:,0]
    # Cursory test to ensure the provided HLA allele/IEDB epitope data in the key file is correctly formatted (the correct format of "HLA-<gene,allele group, protein>_<iedb epitope ID> will pass all 3 of these tests)")
    if not key_data.apply(lambda x: x.split("_")[-1].isnumeric()).all() or not key_data.str.contains("HLA-").all() or not key_data.apply(lambda x: len(x.split("_")[0].split("-"))==2).all():
        print(f"pKd key file {key_file} incorrectly formatted. Call gen_pkd_file --help for usage details.")
    else:
    # Now retrieve the IEDB epitope datafile
        use_custom_iedb=args.iedb[0]
        if use_custom_iedb:
            # If argument to use a custom iedb file passed, make sure the file exists.
            if Path(os.path.abspath(args.iedb[0])).is_file():
                use_custom_iedb=os.path.abspath(args.iedb[0])
            else:
                print(f"IEDB file {os.path.abspath(args.iedb[0])} not found.")
                use_custom_iedb=False
        # Now retrieve pKd values, match order to those in the provided featurized data set or key file, and write file(s)
        get_iedb_pKd_data(
            use_custom_iedb=use_custom_iedb,
            key_data=key_data,
            binding_threshold=args.binding_threshold[0],
            scale=args.scale[0]
            )
    