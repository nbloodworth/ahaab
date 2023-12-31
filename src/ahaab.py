'''
Welcome to AHAAB. 

Created by Nathaniel Bloodworth, MD, PhD
Vanderbilt University Medical Center
Department of Medicine
Division of Clinical Pharmacology

Detailed Usage Examples:

=====================Examples====================
Creating feature representations from PDB models:

    To create atom-based features only, use the 
    --featurize flag:

    $ ahaab.py --featurize example.pdb

    To featurize multiple files, you can pass a 
    directory (which will trigger an attempt to 
    assign features to all PDB files) or a list
    of pdb files (which will be pulled from the
    current directory)

    $ ahaab.py --featurize /path/to/pdbs

    OR

    $ ahaab.py --featurize file_list.txt

    You may customize the features AHAAB uses,
    if the features are defined as a function in
    the module features/atom_features.py. Use
    the --feature_list file to provide features
    in the command line or a .csv file containing
    complete feature definitions. Pass the
    --feature_list flag without arguments to see 
    complete documentation and examples:

    $ ahaab.py --featurize file_list.txt \
               --feature_list

    To commit features to the AHAAB data folder
    (which will be used for data standardization
    and model training/prediction), pass the
    flag --commit_features.

Training AHAAB:
    AHAAB works using a two-step classification
    schema to determine pKd values:
        1. Train classifier to discriminate
           binders from non-binders
        2. Train regression model to predict
           pKd values of binders
    
    Therefore, it must be trained twice - once
    to discriminate binders from non-binders,
    and again to predict pKd values. To train the
    classifier (step 1), an AHAAB features file
    and corresponding pKd values must be 
    supplied. The file containing pKd values will
    contain ONLY 0 (non-binders) or 1 (binder),
    with each feature label at row N corresponding
    to the feature vector in that same row N. The
    classifier is trained in a similar fashion;
    however, provided pKd values must be the
    (experimentally determined) pKd values for
    the hla/peptide pair in question.  
    
    Known pKd values must be provided in either 
    a separate file (with indexed pKd values 
    corresponding to indexed feature vectors in 
    the feature file) or in the same file, with a 
    column heading specififed by the user 
    (defaults to a case-invariant "pkd"). Use 
    the --pkd flag to specify the file name where
    pKd values are stored or the column heading in
    the features file that contains pKd values.

    $ ahaab.py --train AHAAB_atom_features.csv \
               --pkd pkd_values.csv \
               OR
               --pkd <column_heading>

    By default, AHAAB trains the classifier(s) on
    all available feature data. Use the --kfold 
    flag to change this behavior. Set --kfold to
    1 in order to train the model on the entire
    featurized dataset (or omit the argument
    entirely).

    $ ahaab.py --train ahaab_atom_features.csv \
               --pkd pkd_values.csv \
               --kfold 5

Predicting pKd values:
Coming soon.

=================List of Modules=================
ahaab/src
├──ahaab.py
├──features/
|   ├──get_features.py
|   └──atom_features.py
├──train/
|   └──train_ahaab.py
├──predict/
|   ├──predict_ahaab.py
|   └──classifiers.py
├──tools/
|   ├──data_loaders.py
|   ├──handle_input.py
|   ├──multitask.py
|   ├──utils.py
|   └──formats.py
└──data/
    ├──manage_data.py
    ├──features/
    └──weights/
'''

# AHAAB module imports
from features.get_features import get_features_atom
from tools.handle_input import handle_featurize_input
from tools.handle_input import handle_training_input
from tools.handle_input import handle_predict_input
from tools.handle_input import check_feature_list
from tools import multitask
from tools import formats
from data import manage_data
from train import train_ahaab
from predict import predict_ahaab

# Python base libraries
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description="AHAAB, an artificial neural network for predicing peptide-MHC-I binding affinity using atom-based features", epilog="If you use AHAAB for your research, please cite the following manuscript:\n", usage=__doc__)

# ================Arguments that dictate module behavior================
# === Featurize ===
# Mandatory
parser.add_argument("-f","--featurize", nargs=1, default=False, help="Creates a file of atom-based features from one or more peptide-MHC-I PDB models")
# Optional:
parser.add_argument("--feature_list", nargs="*", default=["default"], help="A file or command-line string of features to generate. Pass this flag without arguments to view documentation on how to customize featurization")
parser.add_argument("--multitask", action="store_true", default=False, help="Featurizes complexes simultaneously")
parser.add_argument("--get_metadata", action="store_true", default=False, help="Retrieve metadata for individual atom pairings and write to .json file (CAUTION: for many complexes, this file can be quite large)")
parser.add_argument("--commit_features", action="store_true", default=False, help="Commit feature data to ahaab/data for use in making predictions")


# === Make training and testing datasets ===
# Mandatory
parser.add_argument("-t","--train", nargs=1, default=False, help="Create training and testing datasets from vectorized feature data, and train classifier")
# Options
parser.add_argument("--pkd", nargs=1, default=["pkd"], help="Key file with file names and known pKd values, or column heading with pkd values in vectorized features")
parser.add_argument("--kfold", nargs=1, type=int, default=[1], help="Number of training/testing sets to create for cross-validation. Omit or pass a value of 1 to indicate model should be trained on all available data.")
parser.add_argument("--skip_train", action="store_true", default=False, help="Make testing and training datasets only without generating new network weights")
parser.add_argument("--update_weights", action="store_true", default=False, help="Update model weights to ahaab/data for use in making predictions")

# === Making predictions ===
# Mandatory
parser.add_argument("-p","--predict", nargs=1, default=False, help="Predict pKd values from an AHAAB features file")
parser.add_argument("--predict_pdb", nargs=1, default=False, help="Predict pKd values from PDB file(s)")
parser.add_argument("--weights", nargs=1, default=[os.path.abspath(os.path.join(Path(__file__).parents[0], "data", "weights", "AHAAB_atom_classifier.pt"))], help="Location of trained pytorch model and model weights")
parser.add_argument("--reference_features", nargs=1, default=[os.path.abspath(os.path.join(Path(__file__).parents[0], "data", "features", "AHAAB_atom_features.csv"))], help="Location of reference features file for standardizing input data")

# ======================================================================

# Parse arguments
args=parser.parse_args()

# Creating feature representations from PDB models:
if args.featurize:
    file_list=handle_featurize_input(args.featurize[0])
    feature_list=check_feature_list(args.feature_list)
    if not file_list:
        formats.error("--featurize flag passed but no models found to featurize!")
    # Check if multitasking
    elif args.multitask and not feature_list[1]:
        if os.name=="nt":
            formats.error("Multitasking featurization currently only functions on Linux machines")
        else:
            batches=multitask.batch_files(file_list)
            features=multitask.multiprocess_batches(batches,feature_list[0],get_metadata=args.get_metadata)
    # Proceed with featurization only if a valid feature list was created with user input
    elif not feature_list[1]:
        features=get_features_atom(file_list,feature_list[0],get_metadata=args.get_metadata)
    # Commit featurized data to ahaab/data if user requests
    if file_list and args.commit_features:
        manage_data.commmit_features(features)

# Create testing/training datasets from feature data and train the model(s):
elif args.train:
    # Return a list of datasets divided into train/test pairs
    training_input=handle_training_input(
        args.train[0],
        args.kfold[0],
        pkd_values=args.pkd[0])
    if args.skip_train:
        formats.message("Argument --skip_train passed; train/test datasets created, but model(s) not trained.")
    elif training_input:
        # Return pytorch models
        model_filenames=train_ahaab.train_ahaab_atom(
            ahaab_dataset=training_input[0],
            num_features=training_input[1]
            )
        # Commit weights to ahaab/data if user requests
        if args.update_weights:
            if args.k_fold[0]>1:
                formats.warning("User requested to newly trained model weights, but set value for cross-validation. Model weights not comitted.")
            else:
                manage_data.update_weights(model_filenames)

# Make a prediction
elif args.predict:
    feature_data=handle_predict_input(args.predict[0])
    if feature_data:
        predict_ahaab.predict_ahaab_atom(feature_data[0], feature_data[1], args.weights[0], args.reference_features[0])