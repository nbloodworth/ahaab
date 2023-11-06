'''
AHAAB scripts
Get HLA Peptide Models

This script takes 3 required inputs: (1) a directory containing Rosetta
silent files with scored HLA-peptide decoys; (2) a directory
containing already extracted .pdb files; (3) The HLA allele of
interest.

The objective of this script is to use Rosetta's extract_pdbs
application to generate .pdb files for the highest scoring decoys
in each FlexPepDock run.

IMPORTANT NAMING CONVENTIONS:
This script assumes that Rosetta silent files are named after an IEDB 
epitope ID. It also assumes that the .pdb files in the supplied existing 
model directory follows the naming convention:

HLA-<gene><allele group><protein>_<IEDB epitope ID>

For example: HLA-A0201_109418.pdb

Usage examples:

    $ python get_hla_peptide_models.py \\
        --silent_dir path/to/rosetta/silent/files \\
        --model_dir path/to/existing/hla-epitope/pdbs \\
        --hla_allele A0101
    
    > Will find all silent files in silent_dir, extract the
      lowest (best) scoring decoy, and save it to model_dir
      with the filename HLA-A0101_<silent_file_name>. By
      convention, <silent_file_name> is assumed to be the
      IEDB epitope ID for the epitope in question.
 '''
# python
import argparse
from pathlib import Path
import subprocess
import os
import shutil

# pandas
import pandas as pd

def get_new_pdbs(**kwargs):
    '''
    Usage:

        $ get_new_pdbs(**kwargs)
    
    Keyword arguments: 
    > silent_dir:   directory containing rosetta .silent files.
    > model_dir:    directory containign existing hla-epitope models.
    > hla_allele:   the hla allele that the supplied silent file decoys
                    are docked with.
    > rosetta_extract_pdbs: Path to the rosetta extract_pdbs application.
    > silent_file_extension: Extension used to identify silent files.

    Outputs:
    Extracts new .pdb files to model_dir 

    '''

    silent_dir=kwargs["silent_dir"]
    model_dir=kwargs["model_dir"]
    hla_allele=kwargs["hla_allele"]
    rosetta_extract_pdbs=kwargs["rosetta_extract_pdbs"]
    silent_file_extension=kwargs["silent_file_extension"]

    # First step: Get a list of new filenames:
    # 1. get a list of new silent files to add:
    silent_files=[x.split(".")[0] for x in os.listdir(silent_dir) if silent_file_extension in x]
    filenames=[f"HLA-{hla_allele}_{epitopeID}.pdb" for epitopeID in silent_files]
    existing_filenames=os.listdir(model_dir)
    filenames_to_create=[f for f in filenames if f not in existing_filenames]

    if len(filenames_to_create)==0:
        print(f"No new HLA-epitope models found!")
        return

    print(f"Found {len(filenames_to_create)} new HLA-epitope models.\nFound {len(existing_filenames)} existing HLA-epitope models.")
    args=[rosetta_extract_pdbs]
    for i,f in enumerate(filenames_to_create):
        # First, get the best scoring decoy from the silent file (extract score data using pandas):
        tmp_sf_name=os.path.join(silent_dir,f.split("_")[-1].replace(".pdb",silent_file_extension))
        with open(tmp_sf_name,"r") as sf:
            lines=sf.readlines()
        found_header=False
        score_data=[]
        for line in lines:
            if "SCORE:" in line:
                if not found_header:
                    cols=line.strip("\n").split()
                    found_header=True
                else:
                    score_data.append(line.strip("\n").split())
        score_data=pd.DataFrame(score_data,columns=cols)
        score_data["reweighted_sc"]=score_data["reweighted_sc"].astype(float)
        # Now get the tag for the models with the best (lowest) reweighted_sc term:
        tag=score_data["description"].loc[score_data["reweighted_sc"]==score_data["reweighted_sc"].min()].values[0]
        # Now extract the pdb:
        tmp_args=args
        tmp_args.extend(["-in:file:silent",tmp_sf_name,"-in:file:tags",tag,"-out:path:all",model_dir,"-mute","all"])
        print(f"Now extracting {f} ({i+1} of {len(filenames_to_create)})")
        subprocess.run(tmp_args)
        # And move it to the existing models directory:
        shutil.move(os.path.join(os.getcwd(),tag+".pdb"),os.path.join(model_dir,f))
        if Path(os.path.join(os.getcwd(),tag+".pdb")).is_file():
            os.remove(os.path.join(os.getcwd(),tag+".pdb"))

    return

def hla_epitope_list_check(**kwargs):
    '''
    Usage:

            $ hla_epitope_list_check(**kwargs)
        
        Keyword arguments: 
        > model_dir:    directory containign existing hla-epitope models.
        > hla_epitope_list: File containing HLA/epitope codes in the format:
        HLA-<gene><allele group><protein>_<IEDB epitope ID>, one per line.

        Outputs:
        "missing_hla_epitope_models.txt": File containing a list of
        HLA/epitope codes without models in model_dir
    '''

    model_dir=kwargs["model_dir"]
    hla_epitope_list=os.path.abspath(kwargs["hla_epitope_list"])
    
    # Make sure file exists
    if not Path(hla_epitope_list).is_file():
        print(f"File {hla_epitope_list} not found. Unable to validate missing HLA/epitope models.")
        return

    # If it does, get the list of files we are looking for:
    with open(hla_epitope_list,"r") as f:
        hla_epitope_data=f.readlines()

    existing_filenames=[x.split(".")[0] for x in os.listdir(model_dir) if ".pdb" in x]
    models_to_create=[]
    for hla_epitope in hla_epitope_data:
        if hla_epitope.strip("\n") not in existing_filenames:
            models_to_create.append(hla_epitope)
    if models_to_create:
        print(f"Found {len(models_to_create)} HLA/epitope combinations without models in {model_dir}")
        with open(os.path.join(os.getcwd(),"missing_hla_epitope_models.txt"),"w") as f:
            for m in models_to_create:
                f.write(m)
    else:
        print("No missing HLA/epitope combinations found")

    return

parser = argparse.ArgumentParser(description="A python script to update HLA-peptide models for featurization in AHAAB.", epilog="If you use AHAAB for your research, please cite the following manuscript:\n", usage=__doc__)

# Mandatory:
parser.add_argument("-s","--silent_dir", nargs=1, required=True, help="Directory containing Rosetta silent files. Uses the default '.silent' extension to identify silent files in supplied directory.")
parser.add_argument("-m", "--model_dir", nargs=1, required=True, help="Directory containing .pdb models of HLA/peptide complexes.")
parser.add_argument("-a","--hla_allele",nargs=1,required=True,help="The HLA of interest in the format <gene><allele group><protein> (for example, A0201 for the allele HLA-A*02:01)")

# Optional:
parser.add_argument("-r", "--rosetta", nargs=1, default=["/dors/meilerlab/apps/rosetta/rosetta-3.13/main"], help="Path to the /main subfolder of Rosetta")
parser.add_argument("-ex","--silent_file_extension", nargs=1, default=[".silent"], help="Extension used to identify Rosetta silent files (default is '.silent')")
parser.add_argument("-l", "--hla_epitope_list", nargs=1, default=False, help="File containing a full list of HLA/epitope combinations needed. Names must be in the format 'HLA-<gene><allele group><protein>_<IEDB epitope ID>', with one per line (for example, 'HLA-A0101_67264'). If passed, the script will check the list of existing .pdb files in --model_dir and compare it with the list of HLA/epitope combinations provided in this file, noting inconsistencies.")

# First step: verify input fidelity:
args=parser.parse_args()
silent_dir=os.path.abspath(args.silent_dir[0])
model_dir=os.path.abspath(args.model_dir[0])
rosetta_extract_pdbs=os.path.join(os.path.abspath(args.rosetta[0]),"source/bin/extract_pdbs.default.linuxgccrelease")

if not Path(silent_dir).is_dir():
    print(f"Silent file containing directory {silent_dir} not found!")
elif not Path(model_dir).is_dir():
    print(f"Directory containing existing HLA-epitope models {model_dir} not found!")
elif not Path(rosetta_extract_pdbs).is_file():
    print(f"Rosetta application extract_pdbs not found!")
else:
    # Extract the new pdbs to the desired location
    get_new_pdbs(
        silent_dir=silent_dir,
        model_dir=model_dir,
        hla_allele=args.hla_allele[0],
        rosetta_extract_pdbs=rosetta_extract_pdbs,
        silent_file_extension=args.silent_file_extension[0]
        )
    # Check existing HLA/epitope models against what is required in the supplied argument --hla_epitope_list
    if args.hla_epitope_list:
        hla_epitope_list_check(
            model_dir=model_dir,
            hla_epitope_list=args.hla_epitope_list[0]
            )