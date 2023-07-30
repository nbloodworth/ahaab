'''
AHAAB atom_features submodule
Part of the AHAAB features module

ahaab/
└──features
    └──atom_features.py

Submodule list:

    === atom_hbonds ===
'''
# AHAAB import
from tools import retrieve_data

# rdkit
from rdkit.Chem import rdForceFieldHelpers

# Pandas
import pandas as pd

# numpy
import numpy as np

# Python base libraries
import math
import sys

def atom_hbonds(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, bin_min=0, bin_max=5, bin_size=0.5, get_metadata=False):
    '''
    Usage:
    $ atom_hbonds(*args,**kwargs)

    Positional arguments:
    > pep_feat: rdkit feature object for the peptide
    > pep_mol:  rdkit mol object for the peptide
    > hla_feat: rdkit feature object for the HLA
    > hla_mol:  rdkit mol object for the hla
    > xyz_mol:  pandas dataframe with atomic 
                coordinates

    Keyword arguments:
    > bin_min:  Minimum distance to calculate
                feature in angstroms
    > bin_max:  Maximum distance to calculate
                feature in angstroms
    > bin_size: Size in angstroms of distance 
                bins
    > get_metadata: Flag indicating metadata should
                    be retrieved
    
    Note: bins are determined [bin_min,bind_max)

    Outputs:
    > Tuple containing dataframe with h-bond 
      feature counts and calculation details.
      
      hbonds[0]: pandas dataframe
      Each column contains h_bond_donors or
      h_bond_acceptors at a given distance bin.
      The columns are named with a suffix that
      indicates the bin index (corresponds to
      the value reported in the metadata).

      hbonds[1]: List of dicts
      A list where each element is a dictionary
      containing information about a specific
      hydrogen bond. Includes the following
      fields:

      "h_bond_donor_name"
      "h_bond_donor_resName"
      "h_bond_donor_resNum"
      "h_bond_acceptor_name"
      "h_bond_acceptor_resName"
      "h_bond_acceptor_resNum"
      "h_bond_dist"
      "h_bond_angle"
      "h_bond_bin_low"
      "h_bond_bin_high"
      "h_bond_bin_index"

    '''

    def get_hbond_info(donor_atom,acceptor_atom,pep_mol,hla_mol,xyz_mol):
        '''
        Usage:
        $ hbond_info=get_hbond_info(*args,**kwargs)

        Positional arguments:
        > donor_atom:    rdkit atom object for h-
                         bond donor
        > acceptor_atom: rdkit atom object for h-
                         bond acceptor
        > pep_mol:  rdkit mol object for the peptide
        > hla_mol:  rdkit mol object for the hla
        > xyz_mol:  pandas dataframe with atomic 
                    coordinates

        Keyword arguments:

        Outputs:
        > Tuple containing the h-bond distance
          (index 0) and angle in degrees (index
          1).
        '''

        # Retrieve our coordinate data and atoms that neighbor our donor atom
        neighbors=donor_atom.GetNeighbors()
        donor_xyz=xyz_mol[["x","y","z"]].loc[(xyz_mol["chain_id"]==donor_atom.GetPDBResidueInfo().GetChainId()) & (xyz_mol["res_name"]==donor_atom.GetPDBResidueInfo().GetResidueName()) & (xyz_mol["atom_name"]==donor_atom.GetPDBResidueInfo().GetName().strip()) & (xyz_mol["res_num"]==donor_atom.GetPDBResidueInfo().GetResidueNumber())]
        acceptor_xyz=xyz_mol[["x","y","z"]].loc[(xyz_mol["chain_id"]==acceptor_atom.GetPDBResidueInfo().GetChainId()) & (xyz_mol["res_name"]==acceptor_atom.GetPDBResidueInfo().GetResidueName()) & (xyz_mol["atom_name"]==acceptor_atom.GetPDBResidueInfo().GetName().strip()) & (xyz_mol["res_num"]==acceptor_atom.GetPDBResidueInfo().GetResidueNumber())]

        # Step 1 (less straightfoward): determine if donor atom is tertiary with freely rotatable H-bond
        if len(neighbors)==1 and neighbors[0].GetBonds()[0].GetBondType().name=="SINGLE":
            # Calculate Hbond distance and angle if donor bond is freely rotatable
            neighbors=neighbors[0]
            neighbor_xyz=xyz_mol[["x","y","z"]].loc[(xyz_mol["chain_id"]==neighbors.GetPDBResidueInfo().GetChainId()) & (xyz_mol["res_name"]==neighbors.GetPDBResidueInfo().GetResidueName()) & (xyz_mol["atom_name"]==neighbors.GetPDBResidueInfo().GetName().strip()) & (xyz_mol["res_num"]==neighbors.GetPDBResidueInfo().GetResidueNumber())]
            # Step one: calculate the angle, theta, between donor->hydrogen and donor->acceptor vectors
            v1=np.array([acceptor_xyz["x"].iloc[0]-donor_xyz["x"].iloc[0],acceptor_xyz["y"].iloc[0]-donor_xyz["y"].iloc[0],acceptor_xyz["z"].iloc[0]-donor_xyz["z"].iloc[0]])
            v2=np.array([donor_xyz["x"].iloc[0]-neighbor_xyz["x"].iloc[0],donor_xyz["y"].iloc[0]-neighbor_xyz["y"].iloc[0],donor_xyz["z"].iloc[0]-neighbor_xyz["z"].iloc[0]])
            # Step one: calculate theta, an angle necessary to determining hbond angle
            theta=135-np.degrees(np.arccos(np.dot(v1/np.linalg.norm(v1),v2/np.linalg.norm(v2))))
            # Step two: calculate distance between acceptor and donor
            d_da=np.linalg.norm(v1)
            # Step three: Use law of cosines to calculate distance from H to acceptor atom
            hbond_dist=np.sqrt(1+d_da**2-2*d_da*np.cos(np.radians(theta)))
            # Step four: calculate the angle of the h-bond using law of sins
            hbond_angle=np.degrees(np.arcsin(np.sin(np.radians(theta))/hbond_dist*d_da))
        else:
            # Get the attached hydrogen coordinates. The hydrogen in question will the one in the residue that is closest to the donor atom.
            donor_h=xyz_mol.loc[(xyz_mol["chain_id"]==donor_atom.GetPDBResidueInfo().GetChainId()) & (xyz_mol["res_name"]==donor_atom.GetPDBResidueInfo().GetResidueName()) & (xyz_mol["elem"]=="H") & (xyz_mol["res_num"]==donor_atom.GetPDBResidueInfo().GetResidueNumber())].copy(deep=True)
            dist_vals=[]
            for i in range(len(donor_h)):
                d_v=np.array([donor_h["x"].iloc[i]-donor_xyz["x"].iloc[0],donor_h["y"].iloc[i]-donor_xyz["y"].iloc[0],donor_h["z"].iloc[i]-donor_xyz["z"].iloc[0]])
                dist_vals.append(np.linalg.norm(d_v))
            donor_h["dist_vals"]=dist_vals
            h_xyz=donor_h[["x","y","z"]].loc[donor_h["dist_vals"]==donor_h["dist_vals"].min()]
            # Now calculate the distance from the hydrogen to the acceptor atom
            v1=np.array([h_xyz["x"].iloc[0]-acceptor_xyz["x"].iloc[0],h_xyz["y"].iloc[0]-acceptor_xyz["y"].iloc[0],h_xyz["z"].iloc[0]-acceptor_xyz["z"].iloc[0]])
            hbond_dist=np.linalg.norm(v1)
            # Finally, calculate donor-acceptor distance and use law of cosines to calculate H-bond angle
            d_da=np.linalg.norm(np.array([acceptor_xyz["x"].iloc[0]-donor_xyz["x"].iloc[0],acceptor_xyz["y"].iloc[0]-donor_xyz["y"].iloc[0],acceptor_xyz["z"].iloc[0]-donor_xyz["z"].iloc[0]]))
            d=donor_h["dist_vals"].min()
            hbond_angle=np.degrees(np.arccos((d**2+hbond_dist**2-d_da**2)/(2*d*hbond_dist)))

        return hbond_dist,hbond_angle

    # Hydrogen bond identification works under the operating principle of 'how many hydrogen bonds *could* a given donor/acceptor atom create' in a given distance bin, and counts these potential bonds. It does not attempt to make actual pairings. It does not include or exclude bonds based on angle or distance either, but does return that data if the user wants to filter these out of their feature count later.
        # Identify all donor and acceptors
        # Iterate through distance bins. For each peptide h-bond donor, identify all HLA h-bond acceptors in that bin.
        # Pair each donor with all available receptors, within the distance bin.
        # Pair each acceptor with all available donors, within the distnace bin.
        # Calculate hydrogen bond metadata, including distance and angle

    # Identify h-bond donors and acceptors in peptide and hla
    pep_hbond_donors=[]
    pep_hbond_acceptors=[]
    hla_hbond_donors=[]
    hla_hbond_acceptors=[]

    # Get donors and acceptors for peptide
    for f in pep_feat:
        if f.GetFamily() == "Acceptor":
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                pep_hbond_acceptors.append(pep_mol.GetAtomWithIdx(i))
        elif f.GetFamily() == "Donor":
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                pep_hbond_donors.append(pep_mol.GetAtomWithIdx(i))

    for f in hla_feat:
        if f.GetFamily() == "Acceptor":
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                hla_hbond_acceptors.append(hla_mol.GetAtomWithIdx(i))
        elif f.GetFamily() == "Donor":
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                hla_hbond_donors.append(hla_mol.GetAtomWithIdx(i))

    # Calculate hydrogen bond distance matrix
    hbond_dist_matrix_donors=retrieve_data.get_distance_matrix(pep_hbond_donors, hla_hbond_acceptors, xyz_mol)
    hbond_dist_matrix_acceptors=retrieve_data.get_distance_matrix(pep_hbond_acceptors, hla_hbond_donors, xyz_mol)
    # Sum donor/acceptor pair totals for each donor or acceptor in peptide over each bin distance
    bin_num=int((bin_max-bin_min)/bin_size)

    hbond_donors=[]
    hbond_acceptors=[]
    hbond_metadata=[]
    for b in range(bin_num):
        # Get atoms with distance values from bin:
        # For the index matrix, the first column = index of hla atom, the second column=index of peptide atom
        donors_bin_indx=np.transpose(np.logical_and(hbond_dist_matrix_donors<b*bin_size+bin_size, hbond_dist_matrix_donors>=b*bin_size).nonzero())
        hbond_donors.append(len(donors_bin_indx))
        acceptors_bin_indx=np.transpose(np.logical_and(hbond_dist_matrix_acceptors<b*bin_size+bin_size, hbond_dist_matrix_acceptors>=b*bin_size).nonzero())
        hbond_acceptors.append(len(acceptors_bin_indx))
        # Retrieve metadata
        if get_metadata:
            for i in range(len(donors_bin_indx)):
                donor_atom=pep_hbond_donors[donors_bin_indx[i,1]]
                acceptor_atom=hla_hbond_acceptors[donors_bin_indx[i,0]]
                meta={}
                meta["h_bond_donor_name"]=donor_atom.GetPDBResidueInfo().GetName().strip()
                meta["h_bond_donor_resName"]=donor_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["h_bond_donor_resNum"]=donor_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["h_bond_acceptor_name"]=acceptor_atom.GetPDBResidueInfo().GetName().strip()
                meta["h_bond_acceptor_resName"]=acceptor_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["h_bond_acceptor_resNum"]=acceptor_atom.GetPDBResidueInfo().GetResidueNumber()
                hbond_dist_angle=get_hbond_info(donor_atom,acceptor_atom,pep_mol,hla_mol,xyz_mol)
                meta["h_bond_dist"]=hbond_dist_angle[0]
                meta["h_bond_angle"]=hbond_dist_angle[1]
                meta["h_bond_bin_low"]=b*bin_size
                meta["h_bond_bin_high"]=b*bin_size+bin_size
                meta["h_bond_bin_index"]=b
                hbond_metadata.append(meta)

            for i in range(len(acceptors_bin_indx)):
                donor_atom=hla_hbond_donors[acceptors_bin_indx[i,0]]
                acceptor_atom=pep_hbond_acceptors[acceptors_bin_indx[i,1]]
                meta={}
                meta["h_bond_donor_name"]=donor_atom.GetPDBResidueInfo().GetName().strip()
                meta["h_bond_donor_resName"]=donor_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["h_bond_donor_resNum"]=donor_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["h_bond_acceptor_name"]=acceptor_atom.GetPDBResidueInfo().GetName().strip()
                meta["h_bond_acceptor_resName"]=acceptor_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["h_bond_acceptor_resNum"]=acceptor_atom.GetPDBResidueInfo().GetResidueNumber()
                hbond_dist_angle=get_hbond_info(donor_atom,acceptor_atom,pep_mol,hla_mol,xyz_mol)
                meta["h_bond_dist"]=hbond_dist_angle[0]
                meta["h_bond_angle"]=hbond_dist_angle[1]
                meta["h_bond_bin_low"]=b*bin_size
                meta["h_bond_bin_high"]=b*bin_size+bin_size
                meta["h_bond_bin_index"]=b
                hbond_metadata.append(meta)

    # Create dataframe to hold the features
    hbonds=pd.DataFrame([hbond_donors
        + hbond_acceptors], columns=[
            "h_bond_donors_"+str(b) for b in range(bin_num)]+[
            "h_bond_acceptors_"+str(b) for b in range(bin_num)
            ]
        )

    return hbonds,hbond_metadata

def atom_hydrophobic(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, bin_min=0, bin_max=5, bin_size=0.5, get_metadata=False):
    '''
    Usage:
    $ atom_hydrophobic(*args,**kwargs)

    Positional arguments:
    > pep_feat: rdkit feature object for the peptide
    > pep_mol:  rdkit mol object for the peptide
    > hla_feat: rdkit feature object for the HLA
    > hla_mol:  rdkit mol object for the hla
    > xyz_mol:  pandas dataframe with atomic 
                coordinates

    Keyword arguments:
    > bin_min:  Minimum distance to calculate
                feature in angstroms
    > bin_max:  Maximum distance to calculate
                feature in angstroms
    > bin_size: Size in angstroms of distance 
                bins
    
    Note: bins are determined [bin_min,bind_max)

    Outputs:
    > Tuple containing:
      [0] Dataframe with counts of hydrophobic/
      hydrophobic and hydrophobic/non-hydrophobic
      pairings.
      [1] Metadata concerning pairings in
      dict format
      
      Dataframe contents:
      ["hydrophobic_nonhydrophobic"]
      ["hydrophobic_hydrophobic"]
      ["nonhydrophobic_hydrophobic"]
      ["nonhydrophobic_nonhydrophobic"]
      A list where each value corresponds to the
      number of hydrophobic pairings for
      a given hydrophobic or non-hydrophobic atom
      in the peptide at that distance bin.

      Metadata fields:
      "hydrophobic_peptide_resName"
      "hydrophobic_peptide_resNum"
      "hydrophobic_peptide_name"
      "hydrophobic_peptide_hydrophobicity"
      "hydrophobic_hla_resName"
      "hydrophobic_hla_resNum"
      "hydrophobic_hla_name"
      "hydrophobic_hla_hydrophobicity"
      "hydrophobic_bin_low"
      "hydrophobic_bin_high"
      "hydrophobic_bin_index"
      '''
    # Approach:
    # Hydrophobicity is treated as a binary property. Pairings are made with respect to the peptide atom and the hla atom. For instance, "hydrophobic_nonhydrophobic" counts the number of polar HLA atoms within distance [bin_low, bin_high)

    # Retrieve our hydrophobic and nonhydrophobic atoms from the peptide and HLA
    pep_hydrophobic=[]
    pep_nonhydrophobic=[]
    hla_hydrophobic=[]
    hla_nonhydrophobic=[]

    # Get hydrohpobic and non hydrophobic atoms for peptide and HLA
    for f in pep_feat:
        if f.GetFamily() in ["Hydrophobe","LumpedHydrophobe"]:
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                pep_hydrophobic.append(pep_mol.GetAtomWithIdx(i))
        else:
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                pep_nonhydrophobic.append(pep_mol.GetAtomWithIdx(i))

    for f in hla_feat:
        if f.GetFamily() in ["Hydrophobe","LumpedHydrophobe"]:
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                hla_hydrophobic.append(hla_mol.GetAtomWithIdx(i))
        else:
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                hla_nonhydrophobic.append(hla_mol.GetAtomWithIdx(i))

    # Calculate hydrophobic interactions distance matrices
    hydrophobic_nonhydrophobic_dist_mat=retrieve_data.get_distance_matrix(pep_hydrophobic, hla_nonhydrophobic, xyz_mol)
    hydrophobic_hydrophobic_dist_mat=retrieve_data.get_distance_matrix(pep_hydrophobic, hla_hydrophobic, xyz_mol)
    nonhydrophobic_hydrophobic_dist_mat=retrieve_data.get_distance_matrix(pep_nonhydrophobic, hla_hydrophobic, xyz_mol)
    nonhydrophobic_nonhydrophobic_dist_mat=retrieve_data.get_distance_matrix(pep_nonhydrophobic, hla_nonhydrophobic, xyz_mol)

    # Sum donor/acceptor pair totals for each donor or acceptor in peptide over each bin distance
    bin_num=int((bin_max-bin_min)/bin_size)

    # Initialize lists for pairing counts at each bin
    hydrophobic_nonhydrophobic=[]
    hydrophobic_hydrophobic=[]
    nonhydrophobic_hydrophobic=[]
    nonhydrophobic_nonhydrophobic=[]
    hydrophobic_metadata=[]
    for b in range(bin_num):
        # Get atoms with distance values from bin for each hydrophobic/nonhydrophobic pairing:
        # For the index matrix, the first column = index of hla atom, the second column=index of peptide atom
        hydrophobic_nonhydrophobic_bin_indx=np.transpose(np.logical_and(hydrophobic_nonhydrophobic_dist_mat<b*bin_size+bin_size, hydrophobic_nonhydrophobic_dist_mat>=b*bin_size).nonzero())
        hydrophobic_hydrophobic_bin_indx=np.transpose(np.logical_and(hydrophobic_hydrophobic_dist_mat<b*bin_size+bin_size, hydrophobic_hydrophobic_dist_mat>=b*bin_size).nonzero())
        nonhydrophobic_hydrophobic_bin_indx=np.transpose(np.logical_and(nonhydrophobic_hydrophobic_dist_mat<b*bin_size+bin_size, nonhydrophobic_hydrophobic_dist_mat>=b*bin_size).nonzero())
        nonhydrophobic_nonhydrophobic_bin_indx=np.transpose(np.logical_and(nonhydrophobic_nonhydrophobic_dist_mat<b*bin_size+bin_size, nonhydrophobic_nonhydrophobic_dist_mat>=b*bin_size).nonzero())

        hydrophobic_nonhydrophobic.append(len(hydrophobic_nonhydrophobic_bin_indx))
        hydrophobic_hydrophobic.append(len(hydrophobic_hydrophobic_bin_indx))
        nonhydrophobic_hydrophobic.append(len(nonhydrophobic_hydrophobic_bin_indx))
        nonhydrophobic_nonhydrophobic.append(len(nonhydrophobic_nonhydrophobic_bin_indx))

        # Retrieve metadata
        if get_metadata:
            for i in range(len(hydrophobic_nonhydrophobic_bin_indx)):
                pep_atom=pep_hydrophobic[hydrophobic_nonhydrophobic_bin_indx[i,1]]
                hla_atom=hla_nonhydrophobic[hydrophobic_nonhydrophobic_bin_indx[i,0]]
                meta={}
                meta["hydrophobic_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["hydrophobic_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["hydrophobic_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["hydrophobic_peptide_hydrophobicity"]="hydrophobic"
                meta["hydrophobic_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["hydrophobic_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["hydrophobic_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["hydrophobic_hla_hydrophobicity"]="nonhydrophobic"
                meta["hydrophobic_bin_low"]=b*bin_size
                meta["hydrophobic_bin_high"]=b*bin_size+bin_size
                meta["hydrophobic_bin_index"]=b

                hydrophobic_metadata.append(meta)

            for i in range(len(hydrophobic_hydrophobic_bin_indx)):
                pep_atom=pep_hydrophobic[hydrophobic_hydrophobic_bin_indx[i,1]]
                hla_atom=hla_hydrophobic[hydrophobic_hydrophobic_bin_indx[i,0]]
                meta={}
                meta["hydrophobic_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["hydrophobic_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["hydrophobic_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["hydrophobic_peptide_hydrophobicity"]="hydrophobic"
                meta["hydrophobic_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["hydrophobic_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["hydrophobic_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["hydrophobic_hla_hydrophobicity"]="hydrophobic"
                meta["hydrophobic_bin_low"]=b*bin_size
                meta["hydrophobic_bin_high"]=b*bin_size+bin_size
                meta["hydrophobic_bin_index"]=b

                hydrophobic_metadata.append(meta)

            for i in range(len(nonhydrophobic_hydrophobic_bin_indx)):
                pep_atom=pep_nonhydrophobic[nonhydrophobic_hydrophobic_bin_indx[i,1]]
                hla_atom=hla_hydrophobic[nonhydrophobic_hydrophobic_bin_indx[i,0]]
                meta={}
                meta["hydrophobic_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["hydrophobic_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["hydrophobic_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["hydrophobic_peptide_hydrophobicity"]="nonhydrophobic"
                meta["hydrophobic_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["hydrophobic_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["hydrophobic_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["hydrophobic_hla_hydrophobicity"]="hydrophobic"
                meta["hydrophobic_bin_low"]=b*bin_size
                meta["hydrophobic_bin_high"]=b*bin_size+bin_size
                meta["hydrophobic_bin_index"]=b

                hydrophobic_metadata.append(meta)

            for i in range(len(nonhydrophobic_nonhydrophobic_bin_indx)):
                pep_atom=pep_nonhydrophobic[nonhydrophobic_nonhydrophobic_bin_indx[i,1]]
                hla_atom=hla_nonhydrophobic[nonhydrophobic_nonhydrophobic_bin_indx[i,0]]
                meta={}
                meta["hydrophobic_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["hydrophobic_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["hydrophobic_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["hydrophobic_peptide_hydrophobicity"]="nonhydrophobic"
                meta["hydrophobic_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["hydrophobic_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["hydrophobic_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["hydrophobic_hla_hydrophobicity"]="nonhydrophobic"
                meta["hydrophobic_bin_low"]=b*bin_size
                meta["hydrophobic_bin_high"]=b*bin_size+bin_size
                meta["hydrophobic_bin_index"]=b

                hydrophobic_metadata.append(meta)

    # Output dataframe
    hydrophobic=pd.DataFrame([hydrophobic_nonhydrophobic
        + hydrophobic_hydrophobic
        + nonhydrophobic_hydrophobic
        + nonhydrophobic_nonhydrophobic], columns=[
            "hydrophobic_nonhydrophobic_"+str(b) for b in range(bin_num)]+[
            "hydrophobic_hydrophobic_"+str(b) for b in range(bin_num)]+[
            "nonhydrophobic_hydrophobic_"+str(b) for b in range(bin_num)]+[
            "nonhydrophobic_nonhydrophobic_"+str(b) for b in range(bin_num)
            ]
        )

    return hydrophobic,hydrophobic_metadata

def atom_electrostatic(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, bin_min=0, bin_max=5, bin_size=0.5, get_metadata=False):
    '''
    Usage:
    $ atom_electrostatic(*args,**kwargs)

    Positional arguments:
    > pep_feat: rdkit feature object for the peptide
    > pep_mol:  rdkit mol object for the peptide
    > hla_feat: rdkit feature object for the HLA
    > hla_mol:  rdkit mol object for the hla
    > xyz_mol:  pandas dataframe with atomic 
                coordinates

    Keyword arguments:
    > bin_min:  Minimum distance to calculate
                feature in angstroms
    > bin_max:  Maximum distance to calculate
                feature in angstroms
    > bin_size: Size in angstroms of distance 
                bins
    
    Note: bins are determined [bin_min,bind_max)

    Outputs:
    > Tuple containing:
      [0] Dataframe with counts of electrostatic
      pairings and net attractive/repulsive forces
      at each distance bin for each peptide atom
      [1] Metadata concerning pairings in
      dict format
      
      Dataframe contents:
      ["neg_neg"]
      ["pos_neg"]
      ["neg_pos"]
      ["pos_pos"]
      ["coulomb_net"]

      Metadata fields:
      "electrostatic_peptide_resName"
      "electrostatic_peptide_resNum"
      "electrostatic_peptide_name"
      "electrostatic_peptide_charge"
      "electrostatic_hla_resName"
      "electrostatic_hla_resNum"
      "electrostatic_hla_name"
      "electrostatic_hla_charge"
      "electrostatic_bin_low"
      "electrostatic_bin_high"
      "electrostatic_bin_index"
      "electrostatic_coulombic"
    '''

    def get_coulomb_force(pep_atom,hla_atom,xyz_mol):
        '''
        Usage:
        $ get_coulomb_force(*args,**kwargs)

        Positional arguments:
        > pep_atom: rdkit feature object for the 
                    peptide atom
        > hla_atom: rdkit atom object for the hla
                    atom 
        > xyz_mol:  pandas dataframe with atomic 
                    coordinates

        Keyword arguments:

        Outputs:
        > Coulomb force between atoms with 
        "_GeistegerCharge" property, calculated with
        (pep_atom charge)*(hla_atom charge)/distance^2
        Dielectric constant is assumed to be 1.
        '''

        pep_xyz=xyz_mol[["x","y","z"]].loc[(xyz_mol["chain_id"]==pep_atom.GetPDBResidueInfo().GetChainId()) & (xyz_mol["res_name"]==pep_atom.GetPDBResidueInfo().GetResidueName()) & (xyz_mol["atom_name"]==pep_atom.GetPDBResidueInfo().GetName().strip()) & (xyz_mol["res_num"]==pep_atom.GetPDBResidueInfo().GetResidueNumber())]
        hla_xyz=xyz_mol[["x","y","z"]].loc[(xyz_mol["chain_id"]==hla_atom.GetPDBResidueInfo().GetChainId()) & (xyz_mol["res_name"]==hla_atom.GetPDBResidueInfo().GetResidueName()) & (xyz_mol["atom_name"]==hla_atom.GetPDBResidueInfo().GetName().strip()) & (xyz_mol["res_num"]==hla_atom.GetPDBResidueInfo().GetResidueNumber())]

        d=np.linalg.norm(np.array([pep_xyz["x"].iloc[0]-hla_xyz["x"].iloc[0],pep_xyz["y"].iloc[0]-hla_xyz["y"].iloc[0],pep_xyz["z"].iloc[0]-hla_xyz["z"].iloc[0]]))

        pep_q=float(pep_atom.GetProp("_GasteigerCharge"))
        hla_q=float(hla_atom.GetProp("_GasteigerCharge"))

        coulomb_force=((pep_q)*(hla_q))/(d**2)

        return coulomb_force

    # Approach:
    # We will consider both the binary property of opposite/same charge neighbors, and the net magnitude of the columbic forces on a peptide atom at a given distance bin (neg=attractive, pos=repulsive) 

    # Initialize the output dataframe
    electrostatic=pd.DataFrame(columns=["neg_neg","pos_neg","neg_pos","pos_pos","coulomb_net"])

    # Retrieve our postive and negatively atoms from the peptide and HLA
    pep_pos=[]
    pep_neg=[]
    hla_pos=[]
    hla_neg=[]

    # Get negative and positively charged non-hydrophobes
    pep_exclude_hydrophobe_ids=[]
    for f in pep_feat:
        if f.GetFamily() in ["Hydrophobe","LumpedHydrophobe"]:
            pep_exclude_hydrophobe_ids.extend(list(f.GetAtomIds()))
    hla_exclude_hydrophobe_ids=[]
    for f in hla_feat:
        if f.GetFamily() in ["Hydrophobe","LumpedHydrophobe"]:
            hla_exclude_hydrophobe_ids.extend(list(f.GetAtomIds()))

    numatoms_pep=pep_mol.GetNumAtoms()
    for i in range(numatoms_pep):
        a=pep_mol.GetAtomWithIdx(i)
        gc=float(a.GetProp("_GasteigerCharge"))
        if i not in pep_exclude_hydrophobe_ids:
            if gc>0:
                pep_pos.append(a)
            elif gc<0:
                pep_neg.append(a)
    numatoms_hla=hla_mol.GetNumAtoms()
    for i in range(numatoms_hla):
        a=hla_mol.GetAtomWithIdx(i)
        gc=float(a.GetProp("_GasteigerCharge"))
        if i not in pep_exclude_hydrophobe_ids:
            if gc>0:
                hla_pos.append(a)
            elif gc<0:
                hla_neg.append(a)
    
    # Calculate hydrogen bond distance matrix
    pos_pos_dist_mat=retrieve_data.get_distance_matrix(pep_pos, hla_pos, xyz_mol)
    neg_pos_dist_mat=retrieve_data.get_distance_matrix(pep_neg, hla_pos, xyz_mol)
    pos_neg_dist_mat=retrieve_data.get_distance_matrix(pep_pos, hla_neg, xyz_mol)
    neg_neg_dist_mat=retrieve_data.get_distance_matrix(pep_neg, hla_neg, xyz_mol)

    # Sum donor/acceptor pair totals for each donor or acceptor in peptide over each bin distance
    bin_num=int((bin_max-bin_min)/bin_size)

    # Initialize lists for pairing counts at each bin
    neg_neg=[]
    pos_neg=[]
    neg_pos=[]
    pos_pos=[]
    coulomb_net=[]
    electrostatic_metadata=[]
    for b in range(bin_num):
        # Get atoms with distance values from bin for each electrostatic pairing:
        # For the index matrix, the first column = index of hla atom, the second column=index of peptide atom
        neg_neg_bin_indx=np.transpose(np.logical_and(neg_neg_dist_mat<b*bin_size+bin_size, neg_neg_dist_mat>=b*bin_size).nonzero())
        pos_neg_bin_indx=np.transpose(np.logical_and(pos_neg_dist_mat<b*bin_size+bin_size, pos_neg_dist_mat>=b*bin_size).nonzero())
        neg_pos_bin_indx=np.transpose(np.logical_and(neg_pos_dist_mat<b*bin_size+bin_size, neg_pos_dist_mat>=b*bin_size).nonzero())
        pos_pos_bin_indx=np.transpose(np.logical_and(pos_pos_dist_mat<b*bin_size+bin_size, pos_pos_dist_mat>=b*bin_size).nonzero())

        neg_neg.append(len(neg_neg_bin_indx))
        pos_neg.append(len(pos_neg_bin_indx))
        neg_pos.append(len(neg_pos_bin_indx))
        pos_pos.append(len(pos_pos_bin_indx))

        # Retrieve metadata and calculate charges
        coulomb_net_calc=0
        for i in range(len(neg_neg_bin_indx)):
            pep_atom=pep_neg[neg_neg_bin_indx[i,1]]
            hla_atom=hla_neg[neg_neg_bin_indx[i,0]]
            coulomb_net_calc+=get_coulomb_force(pep_atom,hla_atom,xyz_mol)
            if get_metadata:
                meta={}
                meta["electrostatic_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["electrostatic_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["electrostatic_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["electrostatic_peptide_charge"]=float(pep_atom.GetProp("_GasteigerCharge"))
                meta["electrostatic_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["electrostatic_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["electrostatic_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["electrostatic_hla_charge"]=float(hla_atom.GetProp("_GasteigerCharge"))
                meta["electrostatic_bin_low"]=b*bin_size
                meta["electrostatic_bin_high"]=b*bin_size+bin_size
                meta["electrostatic_bin_index"]=b
                electrostatic_metadata.append(meta)

        for i in range(len(pos_neg_bin_indx)):
            pep_atom=pep_pos[pos_neg_bin_indx[i,1]]
            hla_atom=hla_neg[pos_neg_bin_indx[i,0]]
            coulomb_net_calc+=get_coulomb_force(pep_atom,hla_atom,xyz_mol)
            if get_metadata:
                meta={}
                meta["electrostatic_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["electrostatic_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["electrostatic_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["electrostatic_peptide_charge"]=float(pep_atom.GetProp("_GasteigerCharge"))
                meta["electrostatic_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["electrostatic_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["electrostatic_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["electrostatic_peptide_charge"]=float(hla_atom.GetProp("_GasteigerCharge"))
                meta["electrostatic_bin_low"]=b*bin_size
                meta["electrostatic_bin_high"]=b*bin_size+bin_size
                meta["electrostatic_bin_index"]=b
                electrostatic_metadata.append(meta)

        for i in range(len(neg_pos_bin_indx)):
            pep_atom=pep_neg[neg_pos_bin_indx[i,1]]
            hla_atom=hla_neg[neg_pos_bin_indx[i,0]]
            coulomb_net_calc+=get_coulomb_force(pep_atom,hla_atom,xyz_mol)
            if get_metadata:
                meta={}
                meta["electrostatic_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["electrostatic_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["electrostatic_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["electrostatic_peptide_charge"]=float(pep_atom.GetProp("_GasteigerCharge"))
                meta["electrostatic_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["electrostatic_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["electrostatic_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["electrostatic_peptide_charge"]=float(hla_atom.GetProp("_GasteigerCharge"))
                meta["electrostatic_bin_low"]=b*bin_size
                meta["electrostatic_bin_high"]=b*bin_size+bin_size
                meta["electrostatic_bin_index"]=b
                electrostatic_metadata.append(meta)

        for i in range(len(pos_pos_bin_indx)):
            pep_atom=pep_pos[pos_pos_bin_indx[i,1]]
            hla_atom=hla_pos[pos_pos_bin_indx[i,0]]
            coulomb_net_calc+=get_coulomb_force(pep_atom,hla_atom,xyz_mol)
            if get_metadata:
                meta={}
                meta["electrostatic_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["electrostatic_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["electrostatic_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["electrostatic_peptide_charge"]=float(pep_atom.GetProp("_GasteigerCharge"))
                meta["electrostatic_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["electrostatic_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["electrostatic_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["electrostatic_peptide_charge"]=float(hla_atom.GetProp("_GasteigerCharge"))
                meta["electrostatic_bin_low"]=b*bin_size
                meta["electrostatic_bin_high"]=b*bin_size+bin_size
                meta["electrostatic_bin_index"]=b
                electrostatic_metadata.append(meta)

        coulomb_net.append(coulomb_net_calc)

    # Output data frame
    electrostatic=pd.DataFrame([neg_neg
        +pos_neg
        +neg_pos
        +pos_pos
        +coulomb_net], columns=[
            "neg_neg_"+str(b) for b in range(bin_num)]+[
            "pos_neg_"+str(b) for b in range(bin_num)]+[
            "neg_pos_"+str(b) for b in range(bin_num)]+[
            "pos_pos_"+str(b) for b in range(bin_num)]+[
            "coulomb_net_"+str(b) for b in range(bin_num)
            ]
        )

    return electrostatic,electrostatic_metadata

def atom_VdW(rdkit_mol, pep_mol, hla_mol, xyz_mol, bin_min=0, bin_max=5, bin_size=0.5, get_metadata=False):
    '''
    Usage:
    $ atom_VdW(*args,**kwargs)

    Positional arguments:
    > rdkit_mol:rdkit object for the peptide/HLA
    > pep_mol:  rdkit mol object for the peptide
    > hla_mol:  rdkit mol object for the hla
    > xyz_mol:  pandas dataframe with atomic 
                coordinates

    Keyword arguments:
    > bin_min:  Minimum distance to calculate
                feature in angstroms
    > bin_max:  Maximum distance to calculate
                feature in angstroms
    > bin_size: Size in angstroms of distance 
                bins
    
    Note: bins are determined [bin_min,bind_max)

    Outputs:
    > Tuple containing:
      [0] Dataframe with summed Lennard-Jones
      potentials at each distance bin for each 
      peptide atom
      [1] Metadata contianing pairings and
      calculated energy terms in dict format
      
      Dataframe contents:
      ["LJ_potential"]

      Metadata fields:
      "VdW_peptide_resName"
      "VdW_peptide_resNum"
      "VdW_peptide_name"
      "VdW_hla_resName"
      "VdW_hla_resNum"
      "VdW_hla_name"
      "VdW_hla_charge"
      "VdW_bin_low"
      "VdW_bin_high"
      "VdW_bin_index"
      "VdW_epsilon"
      "VdW_sigma"
    '''
    def get_LJ_energy(pep_atom,hla_atom,rdkit_mol,xyz_mol,epsilon,sigma):
        '''
        Usage:
        $ atom_VdW(*args,**kwargs)

        Positional arguments:
        > pep_atom:  rdkit mol object for the peptide
        > hla_atom:  rdkit mol object for the hla
        > rdkit_mol: rdkit object for the peptide/HLA
        > xyz_mol:   Atom coordinate information
        > epsilon:   VdW well depth
        > sigma:     VdW radius

        Keyword arguments:

        Outputs:
        > Calculated Lennard-Jones potential energy
        '''
        pep_xyz=xyz_mol[["x","y","z"]].loc[(xyz_mol["chain_id"]==pep_atom.GetPDBResidueInfo().GetChainId()) & (xyz_mol["res_name"]==pep_atom.GetPDBResidueInfo().GetResidueName()) & (xyz_mol["atom_name"]==pep_atom.GetPDBResidueInfo().GetName().strip()) & (xyz_mol["res_num"]==pep_atom.GetPDBResidueInfo().GetResidueNumber())]
        hla_xyz=xyz_mol[["x","y","z"]].loc[(xyz_mol["chain_id"]==hla_atom.GetPDBResidueInfo().GetChainId()) & (xyz_mol["res_name"]==hla_atom.GetPDBResidueInfo().GetResidueName()) & (xyz_mol["atom_name"]==hla_atom.GetPDBResidueInfo().GetName().strip()) & (xyz_mol["res_num"]==hla_atom.GetPDBResidueInfo().GetResidueNumber())]

        r=np.linalg.norm(np.array([pep_xyz["x"].iloc[0]-hla_xyz["x"].iloc[0],pep_xyz["y"].iloc[0]-hla_xyz["y"].iloc[0],pep_xyz["z"].iloc[0]-hla_xyz["z"].iloc[0]]))
        V=4*epsilon*((sigma/r)**12-(sigma/r)**6)
        
        return V
    # Approach:
    # A simple calculation of net LJ potential energy for each peptide atom at each distance bin 

    # Assign atoms to peptide or hla (we can't use the split molecules because rdkit's built in sigma/epsilon calculator only works on two atoms from a single molecule/mol object):
    pep_atoms=[]
    hla_atoms=[]
    numatoms=rdkit_mol.GetNumAtoms()
    pep_chainID=pep_mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetChainId()
    hla_chainID=hla_mol.GetAtomWithIdx(0).GetPDBResidueInfo().GetChainId()
    for i in range(numatoms):
        a=rdkit_mol.GetAtomWithIdx(i)
        if a.GetPDBResidueInfo().GetChainId()==pep_chainID:
            pep_atoms.append(a)
        elif a.GetPDBResidueInfo().GetChainId()==hla_chainID:
            hla_atoms.append(a)
    
    # Calculate distance matrix
    dist_mat=retrieve_data.get_distance_matrix(pep_atoms, hla_atoms, xyz_mol)

    # Sum donor/acceptor pair totals for each donor or acceptor in peptide over each bin distance
    bin_num=int((bin_max-bin_min)/bin_size)

    # Initialize lists for pairing counts at each bin
    LJ_energy=[]
    VdW_metadata=[]
    for b in range(bin_num):
        # Get atoms with distance values from bin for each pairing:
        # For the index matrix, the first column = index of hla atom, the second column=index of peptide atom
        bin_indx=np.transpose(np.logical_and(dist_mat<b*bin_size+bin_size, dist_mat>=b*bin_size).nonzero())

        # Retrieve metadata and calculate LJ energies
        LJ_energy_net=0
        for i in range(len(bin_indx)):
            pep_atom=pep_atoms[bin_indx[i,1]]
            hla_atom=hla_atoms[bin_indx[i,0]]

            sigma_epsilon=rdForceFieldHelpers.GetUFFVdWParams(rdkit_mol,pep_atom.GetIdx(),hla_atom.GetIdx())
            sigma=sigma_epsilon[0]
            epsilon=sigma_epsilon[1]
            LJ_energy_net+=get_LJ_energy(pep_atom,hla_atom,rdkit_mol,xyz_mol,epsilon,sigma)

            if get_metadata:
                meta={}
                meta["VdW_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["VdW_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["VdW_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["VdW_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["VdW_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["VdW_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["VdW_bin_low"]=b*bin_size
                meta["VdW_bin_high"]=b*bin_size+bin_size
                meta["VdW_bin_index"]=b
                meta["VdW_epsilon"]=epsilon
                meta["VdW sigma"]=sigma
                VdW_metadata.append(meta)

        LJ_energy.append(LJ_energy_net)

    # The output dataframe
    VdW=pd.DataFrame([LJ_energy], columns=[
            "L-J_energy_"+str(b) for b in range(bin_num)]
            )

    return VdW, VdW_metadata

def atom_pipi(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, bin_min=0, bin_max=5, bin_size=0.5, get_metadata=False):
    '''
    Usage:
    $ atom_pipi(*args,**kwargs)

    > pep_feat: rdkit feature object for the peptide
    > pep_mol:  rdkit mol object for the peptide
    > hla_feat: rdkit feature object for the HLA
    > hla_mol:  rdkit mol object for the hla
    > xyz_mol:  pandas dataframe with atomic 
                coordinates

    Keyword arguments:
    > bin_min:  Minimum distance to calculate
                feature in angstroms
    > bin_max:  Maximum distance to calculate
                feature in angstroms
    > bin_size: Size in angstroms of distance 
                bins
    
    Note: bins are determined [bin_min,bind_max)

    Outputs:
    > Tuple containing:
      [0] Dataframe with summed counts of pi-pi
      interactions at a given distance bin (per
      atom in aromatic ring)
      [1] Metadata contianing info concerning
      rings, members, and distances
      
      Dataframe contents:
      ["pi-pi"]

      Metadata fields:
      "pipi_peptide_resName"
      "pipi_peptide_resNum"
      "pipi_peptide_name"
      "pipi_hla_resName"
      "pipi_hla_resNum"
      "pipi_hla_name"
      "pipi_bin_high"
      "pipi_bin_index"
    '''

    # Going to keep this simple to avoid extending beyond rdKit's capabilities or introducing uncessary complexities into our approach. Identify all atoms belonging to aromatic rings in peptide and HLA. Build distance matrix. Iterate through bins, counting # of atomic pairings within specified bin distnace between peptide and HLA. That's it.

    # Identify atoms with pi-pi bonds in peptide and HLA (those with "aromatic" label)
    pep_atoms=[]
    hla_atoms=[]
    numatoms_pep=pep_mol.GetNumAtoms()
    for i in range(numatoms_pep):
        a=pep_mol.GetAtomWithIdx(i)
        if a.GetIsAromatic():
            pep_atoms.append(a)
    numatoms_hla=hla_mol.GetNumAtoms()
    for i in range(numatoms_hla):
        a=hla_mol.GetAtomWithIdx(i)
        if a.GetIsAromatic():
            hla_atoms.append(a)

    # Calculate distance matrix
    dist_mat=retrieve_data.get_distance_matrix(pep_atoms, hla_atoms, xyz_mol)

    # Sum donor/acceptor pair totals for each donor or acceptor in peptide over each bin distance
    bin_num=int((bin_max-bin_min)/bin_size)

    # Initialize lists for pairing counts at each bin
    pipi_contacts=[]
    pipi_metadata=[]
    for b in range(bin_num):
        # Get atoms with distance values from bin for each pairing:
        # For the index matrix, the first column = index of hla atom, the second column=index of peptide atom
        bin_indx=np.transpose(np.logical_and(dist_mat<b*bin_size+bin_size, dist_mat>=b*bin_size).nonzero())

        pipi_contacts.append(len(bin_indx))
        if get_metadata:
            for i in range(len(bin_indx)):
                pep_atom=pep_atoms[bin_indx[i,1]]
                hla_atom=hla_atoms[bin_indx[i,0]]

                meta={}
                meta["pipi_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["pipi_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["pipi_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["pipi_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["pipi_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["pipi_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["pipi_bin_low"]=b*bin_size
                meta["pipi_bin_high"]=b*bin_size+bin_size
                meta["pipi_bin_index"]=b

                pipi_metadata.append(meta)

    # Output data frame
    pipi=pd.DataFrame([pipi_contacts], columns=[
            "pipi_"+str(b) for b in range(bin_num)
            ]
        )

    return pipi, pipi_metadata

def atom_catpi(pep_feat, pep_mol, hla_feat, hla_mol, xyz_mol, bin_min=0, bin_max=5, bin_size=0.5, get_metadata=False):
    '''
    Usage:
    $ atom_catpi(*args,**kwargs)

    > pep_feat: rdkit feature object for the peptide
    > pep_mol:  rdkit mol object for the peptide
    > hla_feat: rdkit feature object for the HLA
    > hla_mol:  rdkit mol object for the hla
    > xyz_mol:  pandas dataframe with atomic 
                coordinates

    Keyword arguments:
    > bin_min:  Minimum distance to calculate
                feature in angstroms
    > bin_max:  Maximum distance to calculate
                feature in angstroms
    > bin_size: Size in angstroms of distance 
                bins
    
    Note: bins are determined [bin_min,bind_max)

    Outputs:
    > Tuple containing:
      [0] Dataframe with summed counts of cat-pi
      and pi-cat interactions at a given distance 
      bin (per atom)
      [1] Metadata contianing info concerning
      rings, members, and distances
      
      Dataframe contents:
      ["pi-pi"]

      Metadata fields:
      "catpi_peptide_resName"
      "catpi_peptide_resNum"
      "catpi_peptide_name"
      "catpi_hla_resName"
      "catpi_hla_resNum"
      "catpi_hla_name"
      "catpi_bin_high"
      "catpi_bin_index"
    '''
    
    # Identify cations and atoms with aromatic bonds 
    pep_cat_atoms=[]
    pep_aro_atoms=[]
    hla_cat_atoms=[]
    hla_aro_atoms=[]
    # Get aromatics first
    numatoms_pep=pep_mol.GetNumAtoms()
    for i in range(numatoms_pep):
        a=pep_mol.GetAtomWithIdx(i)
        if a.GetIsAromatic():
            pep_aro_atoms.append(a)
    numatoms_hla=hla_mol.GetNumAtoms()
    for i in range(numatoms_hla):
        a=hla_mol.GetAtomWithIdx(i)
        if a.GetIsAromatic():
            hla_aro_atoms.append(a)

    # Next get positively ionizable atoms:
    for f in pep_feat:
        if f.GetFamily() == "PosIonizable":
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                pep_cat_atoms.append(pep_mol.GetAtomWithIdx(i))
    for f in hla_feat:
        if f.GetFamily() == "PosIonizable":
            atom_indx=f.GetAtomIds()
            for i in atom_indx:
                hla_cat_atoms.append(hla_mol.GetAtomWithIdx(i))

    # Calculate distance matrices
    cat_pi_dist_mat=retrieve_data.get_distance_matrix(pep_cat_atoms, hla_aro_atoms, xyz_mol)
    pi_cat_dist_mat=retrieve_data.get_distance_matrix(pep_aro_atoms, hla_cat_atoms, xyz_mol)

    # Sum peptide-cation/hla-aromatic and hla-cation/peptide-aromatic interactions at each distance bin 
    bin_num=int((bin_max-bin_min)/bin_size)

    # Initialize lists for pairing counts at each bin
    cat_pi=[]
    pi_cat=[]
    catpi_metadata=[]
    for b in range(bin_num):
        # Get atoms with distance values from bin for each pairing:
        # For the index matrix, the first column = index of hla atom, the second column=index of peptide atom
        cat_pi_bin_indx=np.transpose(np.logical_and(cat_pi_dist_mat<b*bin_size+bin_size, cat_pi_dist_mat>=b*bin_size).nonzero())
        pi_cat_bin_indx=np.transpose(np.logical_and(pi_cat_dist_mat<b*bin_size+bin_size, pi_cat_dist_mat>=b*bin_size).nonzero())

        cat_pi.append(len(cat_pi_bin_indx))
        pi_cat.append(len(pi_cat_bin_indx))

        # Gather metadata
        if get_metadata:
            for i in range(len(cat_pi_bin_indx)):
                pep_atom=pep_cat_atoms[cat_pi_bin_indx[i,1]]
                hla_atom=hla_aro_atoms[cat_pi_bin_indx[i,0]]

                meta={}
                meta["catpi_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["catpi_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["catpi_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["catpi_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["catpi_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["catpi_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["catpi_cation"]="peptide"
                meta["catpi_aromatic"]="hla"
                meta["catpi_bin_low"]=b*bin_size
                meta["catpi_bin_high"]=b*bin_size+bin_size
                meta["catpi_bin_index"]=b

                catpi_metadata.append(meta)

            for i in range(len(pi_cat_bin_indx)):
                pep_atom=pep_aro_atoms[pi_cat_bin_indx[i,1]]
                hla_atom=hla_cat_atoms[pi_cat_bin_indx[i,0]]

                meta={}
                meta["catpi_peptide_resName"]=pep_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["catpi_peptide_resNum"]=pep_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["catpi_peptide_name"]=pep_atom.GetPDBResidueInfo().GetName().strip()
                meta["catpi_hla_resName"]=hla_atom.GetPDBResidueInfo().GetResidueName().strip()
                meta["catpi_hla_resNum"]=hla_atom.GetPDBResidueInfo().GetResidueNumber()
                meta["catpi_hla_name"]=hla_atom.GetPDBResidueInfo().GetName().strip()
                meta["catpi_cation"]="hla"
                meta["catpi_aromatic"]="peptide"
                meta["catpi_bin_low"]=b*bin_size
                meta["catpi_bin_high"]=b*bin_size+bin_size
                meta["catpi_bin_index"]=b

                catpi_metadata.append(meta)

    # Output data frame
    catpi=pd.DataFrame([cat_pi
        +pi_cat], columns=[
            "cat-pi_"+str(b) for b in range(bin_num)]+[
            "pi-cat_"+str(b) for b in range(bin_num)
            ]
        )

    return catpi, catpi_metadata