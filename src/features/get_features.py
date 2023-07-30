'''
AHAAB get_features submodule
Part of the AHAAB features module

ahaab/
└──features
    └──get_features.py

Submodule list:

    === get_features_atom ===
'''

# AHAAB module imports
from tools import retrieve_data
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
import sys

def get_features_atom(file_list,get_metadata=False,toprint=True,output_suffix=""):
    '''
    Usage:
    $ get_features_atom(*args,**kwargs)

    Positional arguments:
    > file_list: List of PDBs filenames to process

    Keyword arguments:
    > get_metadata:  Retrieve metadata for assigned features
                     and write to .json file (normally false;
                     metadata is consists of details re:
                     atom-atom pairings, so these files can
                     be fairly large)
    > toprint:       Whether to print progress details. 
                     Set to false when running multiprocess
    > output_suffix: Suffix to add to output 
                     features file. Assigned during 
                     multiprocess.

    Returns:
    > Writes two files:
      1. File named "AHAAB_atom_features{_[output_suffix]}.csv"
         which contains feature assignments for each complex.
      2. File named "AHAAB_atom_features_metadata.json"


    '''

    # Initialize our pandas dataframe
    features_data=pd.DataFrame(columns=["PDB file name"])
    metadata=[]
    num_to_featurize=len(file_list)

    # Loop through pdbs and attempt to featurize each complex
    for i,pdb in enumerate(file_list):
        # Initialize the temporary dataframe to hold feature data
        pdb_name=os.path.basename(pdb)[0:-4]
        pdb_features=pd.DataFrame(data=[pdb_name],columns=["PDB file name"])
        pdb_metadata={"PDB Name":pdb_name}
        formats.notice(f"Assigning features to {i+1:>7}/{len(file_list):<7} model(s) ({pdb_name})")
        
        # Read the file as an rdkit molecule
        formats.message(f"Reading and sanitizing {pdb_name}...",toprint=toprint, end="\r")
        try:
            rdkit_mol=rdmolfiles.MolFromPDBFile(pdb,sanitize=True)
        except:
            formats.error(f"\nFailed to read {pdb_name}")
            continue

        if not rdkit_mol:
            formats.error(f"\nFailed to read {pdb_name}")
            continue

        # Retrieve corresponding coordinate data from PDB
        xyz_mol=retrieve_data.get_xyz(pdb)

        # Infer MHC-I and peptide chains based on chain length
        if len(xyz_mol["chain_id"].unique())>2:
            formats.warn(f"\nMore than 2 chains in PDB file {pdb_name}. Attempting HLA and peptide chain assignments...")
        chains=retrieve_data.get_hla_peptide_chains(xyz_mol)

        # Check correct chain assignment (get_hla_peptide_chains returns an error message in the output tuple if failed)
        if not chains[0]:
            formats.error(chains[1])
            continue

        # Split molecule by chain ID and assign components to HLA and peptide
        mol_dict=rdmolops.SplitMolByPDBChainId(rdkit_mol)
        hla_mol=mol_dict[chains[0]]
        pep_mol=mol_dict[chains[1]]

        # Sanitize (again, for some reason aromaticity lost after splitting) and assign base rdkit features
        try:
            rdmolops.SanitizeMol(hla_mol)
        except:
            formats.error(f"\nError sanitizing HLA chain for {pdb_name} (chain {chains[0]})")
            continue

        try:
            rdmolops.SanitizeMol(pep_mol)
        except:
            formats.error(f"\nError sanitizing peptide chain for {pdb_name} (chain {chains[1]})")
            continue

        if not hla_mol or not pep_mol:
            formats.error(f"\nFailed sanitation of {pdb_name} chains")
            continue

        # Retrieve base features for both peptide and HLA
        try:
            feature_factory=ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir,"BaseFeatures.fdef"))
            hla_feat=feature_factory.GetFeaturesForMol(hla_mol)
            pep_feat=feature_factory.GetFeaturesForMol(pep_mol)
        except:
            formats.error(f"\nFailed to retrieve base rdkit features for {pdb_name}")
            continue

        rdPartialCharges.ComputeGasteigerCharges(pep_mol)
        rdPartialCharges.ComputeGasteigerCharges(hla_mol)

        # Perform featurization:
        try:
            # Hydrogen bond data)
            formats.message("\x1b[2K",toprint=toprint, end="\r")
            formats.message("Counting hydrogen bond donor/acceptor pairs...",toprint=toprint, end="\r")
            hbonds=atom_features.atom_hbonds(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, get_metadata=get_metadata)
            pdb_features=pd.concat([pdb_features,hbonds[0]],axis=1)
            pdb_metadata["h_bonds"]=hbonds[1]

            # Hydrophobic pairings
            formats.message("\x1b[2K",toprint=toprint, end="\r")
            formats.message("Getting hydrophobic interactions...",toprint=toprint, end="\r")
            hydrophobic_pairs=atom_features.atom_hydrophobic(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, get_metadata=get_metadata)
            pdb_features=pd.concat([pdb_features,hydrophobic_pairs[0]],axis=1)
            pdb_metadata["hydrophobic"]=hydrophobic_pairs[1]

            # Electrostatic interactions (culombic forces between atoms with partial charges)
            formats.message("\x1b[2K",toprint=toprint, end="\r")
            formats.message("Determining electrostatic interactions...",toprint=toprint, end="\r")
            electrostatic_interactions=atom_features.atom_electrostatic(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, get_metadata=get_metadata)
            pdb_features=pd.concat([pdb_features,electrostatic_interactions[0]],axis=1)
            pdb_metadata["electrostatic"]=electrostatic_interactions[1]

            # VdW interactions
            formats.message("\x1b[2K",toprint=toprint, end="\r")
            formats.message("Calculating Lennard-Jones potential energies...",toprint=toprint, end="\r")
            VdW_interactions=atom_features.atom_VdW(rdkit_mol, pep_mol, hla_mol, xyz_mol, get_metadata=get_metadata)
            pdb_features=pd.concat([pdb_features,VdW_interactions[0]],axis=1)
            pdb_metadata["Lennard-Jones"]=VdW_interactions[1]

            # Aromatic interactions
            # pi-stacking
            formats.message("\x1b[2K",toprint=toprint, end="\r")
            formats.message("Counting pi-pi interactions...",toprint=toprint, end="\r")
            pipi_interactions=atom_features.atom_pipi(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, get_metadata=get_metadata)
            pdb_features=pd.concat([pdb_features,pipi_interactions[0]],axis=1)
            pdb_metadata["pi-pi"]=pipi_interactions[1]

            # cation-pi
            formats.message("\x1b[2K",toprint=toprint, end="\r")
            formats.message("Counting cation-pi interactions...",toprint=toprint, end="\r")
            catpi_interactions=atom_features.atom_catpi(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, get_metadata=get_metadata)
            pdb_features=pd.concat([pdb_features,catpi_interactions[0]],axis=1)
            pdb_metadata["cation-pi"]=catpi_interactions[1]

            # Concatenate to growing features dataframe
            features_data=pd.concat([features_data, pdb_features],ignore_index=True)

            # Add metadata to list
            metadata.append(pdb_metadata)

        except:
            formats.error(f"\nError featurizing {pdb_name}")
            continue

        # ASCI control sequence to return to line 1 and delete contents.
        formats.message("\033[1A",toprint=toprint, end="\r")
        formats.message("\x1b[2K",toprint=toprint, end="\r")

    # Write the features and metadata to file(s)
    formats.notice("Featurization complete!")
    if output_suffix:
        outfile_name=os.path.abspath("AHAAB_atom_features_"+output_suffix+".csv")
        metadata_filename=os.path.abspath("AHAAB_atom_features_metadata_"+output_suffix+".json")

    else:
        outfile_name=os.path.abspath("AHAAB_atom_features.csv")
        metadata_filename=os.path.abspath("AHAAB_atom_features_metadata.json")

    formats.message(f"Features written to {outfile_name}",toprint=toprint)
    features_data.to_csv(outfile_name,index=False)

    if get_metadata:
        formats.message(f"Feature metadata written to {metadata_filename}",toprint=toprint)
        with open(metadata_filename,"w") as mf:
            json.dump(metadata,mf)
    print(features_data)
    return features_data
