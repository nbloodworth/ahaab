'''
AHAAB utils submodule
Part of the AHAAB tools module

ahaab/
└──tools
    └──utils.py

Submodule list:
    
    get_xyz
    get_hla_peptide_chains
    get_distance_matrix
    standardize_feature_data
'''

# AHAAB module imports
from tools import formats

# Pandas
import pandas as pd

# numpy
import numpy as np

# Python base libraries
import sys
import os
import warnings
warnings.filterwarnings("ignore")

def get_xyz(pdb_file):
    '''
    Usage:
    $ xyz_pdb=get_xyz(*args)

    Positional arguments:
    > input_data: single PDB file containing an
                  hla-peptide complex
    
    Keyword arguments:

    Outputs:
    > Pandas dataframe containing ATOM and HETATM
      records for the input PDB
    '''

    # Open PDB and read in only lines with ATOM and HETATM records
    pdb_file=os.path.abspath(pdb_file)
    with open(pdb_file) as f:
        pdb_rec= [l.strip() for l in f if any(x in l for x in ["ATOM","HETATM"])]

    # Cycle through, collect relevant column data, and return pandas dataframe with info
    data=[]
    for rec in pdb_rec:
        rec=rec.split()
        data.append(rec[1:9]+[rec[-1]])

    xyz_data=pd.DataFrame(data,columns=["atom_num","atom_name","res_name","chain_id","res_num","x","y","z","elem"])
    xyz_data[["atom_num","x","y","z","res_num"]]=xyz_data[["atom_num","x","y","z","res_num"]].apply(pd.to_numeric)
    
    return xyz_data

def get_hla_peptide_chains(xyz, keep_ligands=False):
    '''
    Usage:
    $ chains=get_hla_peptide_chains(*args,**kwargs)

    Positional arguments:
    > xyz: coordinate file for PDB
    
    Keyword arguments:
    > keep_ligands: If True, will keep all chains
                    composed of HETATMs regardless
                    of length
    Outputs:
    > A tuple containing chain ID of HLA (index 0)
      and of the peptide (index 1). Returns tuple
      with False (index 1) and error message (index 
      2) if it fails.
    '''

    # Default behavior: assume HLA and peptide are first two chains in PDB, and HLA is the longer chain
    chain_ids=xyz["chain_id"].unique()
    chain1=chain_ids[0]
    chain2=chain_ids[1]
    chain1_resnum=len(xyz["res_num"].loc[xyz["chain_id"]==chain1].unique().tolist())
    chain2_resnum=len(xyz["res_num"].loc[xyz["chain_id"]==chain2].unique().tolist())
    if chain1_resnum == chain2_resnum:
        return False, f"Unable to infer HLA and peptide chains. Complex not featurized."
    elif chain1_resnum > chain2_resnum:
        hla_chain=chain1
        pep_chain=chain2
        formats.message(f"Inferred peptide chain: {pep_chain}, {chain2_resnum} residues")
    else:
        hla_chain=chain2
        pep_chain=chain1
        formats.message(f"Inferred peptide chain: {pep_chain}, {chain1_resnum} residues")

    return hla_chain, pep_chain

def get_distance_matrix(atoms_1, atoms_2, xyz):
    '''
    Usage:
    $ matrix=get_distance_matrix(*args,**kwargs)

    Positional arguments:
    > atoms_1: rdkit atoms of the first molecule
    > atoms_2: rdkit atoms of the second molecule
    > xyz:     Coordinate file for PDB
    
    Keyword arguments:
    > 

    Outputs:
    > A pandas dataframe containing euclidean distances 
      between atoms in atoms_1 and atoms_2. Index and
      column names are in following format:
      <Residue number>_<Chain ID>_<RDkit atom index>
    '''

    # Build x, y, and z coordinate vectors
    atoms_1x=[]
    atoms_1y=[]
    atoms_1z=[]
    atoms_1_id=[]
    atoms_2x=[]
    atoms_2y=[]
    atoms_2z=[]
    atoms_2_id=[]
    for a in atoms_1:
        atoms_1x.append(xyz["x"].loc[(xyz["chain_id"]==a.GetPDBResidueInfo().GetChainId()) & 
                                     (xyz["res_name"]==a.GetPDBResidueInfo().GetResidueName()) & 
                                     (xyz["atom_name"]==a.GetPDBResidueInfo().GetName().strip()) & 
                                     (xyz["res_num"]==a.GetPDBResidueInfo().GetResidueNumber())].iloc[0])
        atoms_1y.append(xyz["y"].loc[(xyz["chain_id"]==a.GetPDBResidueInfo().GetChainId()) & 
                                     (xyz["res_name"]==a.GetPDBResidueInfo().GetResidueName()) & 
                                     (xyz["atom_name"]==a.GetPDBResidueInfo().GetName().strip()) & 
                                     (xyz["res_num"]==a.GetPDBResidueInfo().GetResidueNumber())].iloc[0])
        atoms_1z.append(xyz["z"].loc[(xyz["chain_id"]==a.GetPDBResidueInfo().GetChainId()) & 
                                     (xyz["res_name"]==a.GetPDBResidueInfo().GetResidueName()) & 
                                     (xyz["atom_name"]==a.GetPDBResidueInfo().GetName().strip()) & 
                                     (xyz["res_num"]==a.GetPDBResidueInfo().GetResidueNumber())].iloc[0])
        atoms_1_id.append(str(a.GetPDBResidueInfo().GetResidueNumber())+"_"+
                          a.GetPDBResidueInfo().GetChainId()+"_"+
                          str(a.GetIdx()))
    for a in atoms_2:
        atoms_2x.append(xyz["x"].loc[(xyz["chain_id"]==a.GetPDBResidueInfo().GetChainId()) & 
                                     (xyz["res_name"]==a.GetPDBResidueInfo().GetResidueName()) & 
                                     (xyz["atom_name"]==a.GetPDBResidueInfo().GetName().strip()) & 
                                     (xyz["res_num"]==a.GetPDBResidueInfo().GetResidueNumber())].iloc[0])
        atoms_2y.append(xyz["y"].loc[(xyz["chain_id"]==a.GetPDBResidueInfo().GetChainId()) & 
                                     (xyz["res_name"]==a.GetPDBResidueInfo().GetResidueName()) & 
                                     (xyz["atom_name"]==a.GetPDBResidueInfo().GetName().strip()) & 
                                     (xyz["res_num"]==a.GetPDBResidueInfo().GetResidueNumber())].iloc[0])
        atoms_2z.append(xyz["z"].loc[(xyz["chain_id"]==a.GetPDBResidueInfo().GetChainId()) & 
                                     (xyz["res_name"]==a.GetPDBResidueInfo().GetResidueName()) & 
                                     (xyz["atom_name"]==a.GetPDBResidueInfo().GetName().strip()) & 
                                     (xyz["res_num"]==a.GetPDBResidueInfo().GetResidueNumber())].iloc[0])
        atoms_2_id.append(str(a.GetPDBResidueInfo().GetResidueNumber())+"_"+
                          a.GetPDBResidueInfo().GetChainId()+"_"+
                          str(a.GetIdx()))

    # Create numpy meshgrid(s)
    xx_pep,xx_hla=np.meshgrid(atoms_1x,atoms_2x)
    yy_pep,yy_hla=np.meshgrid(atoms_1y,atoms_2y)
    zz_pep,zz_hla=np.meshgrid(atoms_1z,atoms_2z)

    # rows=HLA (seconda argument)
    # columns=peptide (1st argument)
    # IF you call the function with peptide atoms, hla atoms as arguments in that order
    dist_mat=np.sqrt((xx_pep-xx_hla)**2+(yy_pep-yy_hla)**2+(zz_pep-zz_hla)**2)
    dist_mat=pd.DataFrame(dist_mat,index=atoms_2_id,columns=atoms_1_id)
    return dist_mat

def standardize_feature_data(toscale,scaleto=None):
    '''
    Usage:
    $ standardize_feature_data(*args,**kwargs)

    Positional arguments:
    > toscale: The features to scale
    
    Keyword arguments:
    > scaleto: The reference feature data used to set
               parameters for scaling. If omitted,
               will scale input data to itself,
               assuming input data >= 2 entry rows.

    Outputs:
    > A numpy array containing scaled features
    > Returns an empty numpy array if failed to scale
      (mostly to allow use of np.any() for checking
      success down the pipeline)
    '''

    # This function takes in two numpy matrices representing unscaled feature data and returns the "toscale" data after peforming a scaling operation. If the user passes reference data, it scales the data to the reference mean and standard deviation; otherwise, the data is scaled to itself.

    if not scaleto:
        scaleto=toscale
    
    rows_cols=np.shape(toscale)
    rows=rows_cols[0]
    if rows<=1 or len(rows_cols)<2:
        formats.error("Unable to scale feature data without a reference!")
        return np.empty()
    else:
        cols=rows_cols[1]

    # Standardize the data (z=(x-u)/stdev)
    scaled_data=np.nan_to_num((toscale-np.mean(scaleto,axis=0))/np.std(scaleto,axis=0))

    return scaled_data