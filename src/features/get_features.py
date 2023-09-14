'''
AHAAB get_features submodule
Part of the AHAAB features module

ahaab/
└──features
    └──get_features.py

Submodule list:

    get_features_atom
'''

# AHAAB module imports
from tools import utils
from tools import formats
from features import atom_features

# rdkit imports
from rdkit.Chem import rdmolfiles
from rdkit.Chem import rdmolops
from rdkit.Chem import ChemicalFeatures
from rdkit.Chem import rdPartialCharges
from rdkit import RDConfig 

# Pandas import
import pandas as pd

# Python base libraries
import os
import json

def get_features_atom(file_list,features,get_metadata=False,update_features=False,toprint=True,output_suffix=""):
    '''
    Usage:
    $ get_features_atom(*args,**kwargs)

    Positional arguments:
    > file_list:    List of PDBs filenames to process
    > features:     Pandas dataframe containing feature
                    names  

    Keyword arguments:
    > get_metadata: Retrieve metadata for assigned features
                    and write to .json file (normally false;
                    metadata is consists of details re:
                    atom-atom pairings, so these files can
                    be fairly large)
    > update_features: Update existing dataset features in the
                    provided file
    > toprint:       Whether to print progress details. 
                    Set to false when running multiprocess
    > output_suffix: Suffix to add to output 
                    features file. Assigned during 
                    multiprocess.

    Returns:
    > Writes the following files:
      1. AHAAB_atom_features.csv
        .csv file containing indexed feature vectors for
        each peptide/MHC-I complex featurized. The first
        column corresponds to the file name.
      2. AHAAB_atom_features_metadata.json
        <Optional> .json file containing metadata for pdbs
        and features (can be quite large)
      3. AHAAB_error_log.csv
        .csv file containing two column
    > Returns:
        Pandas dataframe containing feature data for
        use in any downstream applications
    ''' 
    # Initialize our pandas dataframe, metadata, and rdkit feature factory
    features_data=pd.DataFrame()
    error_log=pd.DataFrame(columns=["PDB file name", "Error description"])
    metadata=[]
    feature_factory=ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,"BaseFeatures.fdef"))

    # Loop through pdbs and attempt to featurize each complex
    for i,pdb in enumerate(file_list):
        # Initialize the temporary dataframe to hold feature data
        pdb_name=os.path.basename(pdb)[0:-4]
        pdb_features=pd.DataFrame(data=[pdb_name],columns=["PDB file name"])
        formats.notice(f"Assigning features to {i+1:>7}/{len(file_list):<7} model(s) ({pdb_name})")
        
        # Read the file as an rdkit molecule
        formats.message(f"Reading and sanitizing {pdb_name:<20}...",toprint=toprint)
        rdkit_mol=rdmolfiles.MolFromPDBFile(pdb,sanitize=True)
        if not rdkit_mol:
            error_description=f"RDKit failed to read {pdb_name}"
            formats.error(error_description)
            error_log=pd.concat([error_log,pd.Series([pdb_name,error_description])])
            continue

        # Retrieve corresponding coordinate data from PDB
        xyz_mol=utils.get_xyz(pdb)

        # Infer MHC-I and peptide chains based on chain length
        if len(xyz_mol["chain_id"].unique())>2:
            formats.warn(f"More than 2 chains in PDB file {pdb_name}. Attempting HLA and peptide chain assignments...",toprint=toprint)
        chains=utils.get_hla_peptide_chains(xyz_mol,toprint=toprint)
        # Check correct chain assignment (get_hla_peptide_chains returns an error message in the output tuple if failed)
        if not chains[0]:
            error_description=chains[1]
            error_log=pd.concat([error_log,pd.Series([pdb_name,error_description])])
            formats.error(error_description)
            continue

        # Split molecule by chain ID and assign components to HLA and peptide
        mol_dict=rdmolops.SplitMolByPDBChainId(rdkit_mol)
        hla_mol=mol_dict[chains[0]]
        pep_mol=mol_dict[chains[1]]

        # Sanitize (again, for some reason aromaticity lost after splitting) and assign base rdkit features
        rdmolops.SanitizeMol(hla_mol)
        rdmolops.SanitizeMol(pep_mol)
        if not hla_mol or not pep_mol:
            error_description=f"Failed sanitation of {pdb_name} chains"
            error_log=pd.concat([error_log,pd.Series([pdb_name,error_description])])
            formats.error(error_description)
            continue

        # Retrieve base features for both peptide and HLA and assign partial charges to atoms
        formats.message("Assigning base rdkit features...",toprint=toprint)
        try:
            hla_feat=feature_factory.GetFeaturesForMol(hla_mol)
            pep_feat=feature_factory.GetFeaturesForMol(pep_mol)
            rdPartialCharges.ComputeGasteigerCharges(pep_mol)
            rdPartialCharges.ComputeGasteigerCharges(hla_mol)
        except:
            error_description=f"Failed to retrieve base rdkit features for {pdb_name}"
            error_log=pd.concat([error_log,pd.Series([pdb_name,error_description])])
            formats.error(error_description)
            continue
        
        # Retrieve all-atom distance matrix
        formats.message(f"Retrieving all-atom distance matrix...",toprint=toprint)
        all_atom_dist=utils.get_distance_matrix(pep_mol.GetAtoms(),hla_mol.GetAtoms(),xyz_mol)

        # Perform featurization:
        featurization_error_flag=False
        for idx in range(0,len(features)):
            # Check if feature should be calculated over every residue or on every peptide:
            if features["interface_scope"].iloc[idx]=="res":
                resnum=xyz_mol["res_num"].loc[xyz_mol["chain_id"]==chains[1]].unique().tolist()
            else:
                resnum=["all"]
            # Retrieve the feature function:
            feature_func=getattr(atom_features, features["atom_feature"].iloc[idx])
            try:
                tmp_features=feature_func(
                    pep_feat=pep_feat,
                    pep_mol=pep_mol,
                    hla_feat=hla_feat,
                    hla_mol=hla_mol,
                    xyz_mol=xyz_mol,
                    all_atom_dist=all_atom_dist,
                    bin_min=features["bin_min"].iloc[idx],
                    bin_max=features["bin_max"].iloc[idx],
                    bin_size=features["bin_size"].iloc[idx],
                    resnum=resnum,
                    get_metadata=get_metadata,
                    rdkit_mol=rdkit_mol,
                    toprint=toprint
                )
                pdb_features=pd.concat([pdb_features,tmp_features],axis=1)
            except:
                error_description=f"Failed to featurize {pdb_name} with feature {features.iloc[idx]}"
                error_log=pd.concat([error_log,pd.Series([pdb_name,error_description])])
                formats.error(error_description)
                featurization_error_flag=True
                break
        
        if not featurization_error_flag:
            features_data=pd.concat([features_data,pdb_features])
        else:
            continue

    # Write the features and metadata to file(s)
    formats.notice("Featurization complete!",toprint=toprint)
    if output_suffix:
        outfile_name=os.path.abspath("AHAAB_atom_features_"+output_suffix+".csv")
        errorlog_filename=os.path.abspath("AHAAB_atom_features_error"+output_suffix+".log")
        metadata_filename=os.path.abspath("AHAAB_atom_features_metadata_"+output_suffix+".json")
    else:
        outfile_name=os.path.abspath("AHAAB_atom_features.csv")
        errorlog_filename=os.path.abspath("AHAAB_atom_features_error.log")
        metadata_filename=os.path.abspath("AHAAB_atom_features_metadata.json")

    formats.message(f"Features written to {outfile_name}",toprint=toprint)
    features_data.to_csv(outfile_name,index=False)

    if get_metadata:
        formats.message(f"Feature metadata written to {metadata_filename}",toprint=toprint)
        with open(metadata_filename,"w") as mf:
            json.dump(metadata,mf)
    
    if not error_log.empty:
        error_log.to_csv(errorlog_filename,index=False)
        formats.message(f"Error log written to {errorlog_filename}")

    return features_data
