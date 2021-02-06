# Guide for replication of results for paper "Project-Level Encoding for Neural Source Code Summarization of Subroutines", currently under peer review
## Step 0 - Dataset building
We began with the main java dataset of 2.1m methods aswell the complete 50 million method extended set requested from Le Clair et al{}

The dataset was filtered for duplicates then project context was constructed from the extended set using the scripts in the "builder" folder.

Due to cloud memory constraint, we provide the compiled dataset as well as the scripts used to compile, the intermediary data files can be made available on request for modification purposes. This data can be found at :


## Step 1 - Training
To ensure no recursive errors or edits, create directories nfs>projects and clone this git repository.
Download and unpack all data from the aws link into this directory as well.
Create directory outdir, with 4 subdirectories  **outdir/{models, histories, viz, predictions}**
**Use Requirements.txt to get your python 3.x virtual environment in sync with our setup.** Venv is preferred. Common issues that might arise from updating an existing venv and solutions :
- GPU not recognized: checking the compatibility of your gpu cudnn/cuda or other drivers with the keras and tf versions fixes this.
- Tf unable to allocate tensor: uninstall tensorflow and then update tensorflow-gpu only. Note we have not tested our setup with tf 2.x
- keras "learning rate" error: clean uninstall keras and install keras 2.3.1 {pip upgrade is broken for this dependency so will not work}

To train the most basic project-level context model use the following command :
```
time python3 train.py --model-type=attendgru-pc --batch-size=50 --epochs=10 --datfile=dataset_random.pkl --gpu=0
```
Note: --datfile=dataset_3Drandom.pkl for code2seq and graph2seq models or any custom models that use ast graphs you might wanna test. This is true for --datfile arg for all scripts in this project.

Scripts for firstwords versions for RQ2 table can be found in the firstwords folder and largely follow the same pattern as these scripts {predicts are provided as well}

## Step 2 - Predictions
Training print screen will display the epoch at which the model converges, that is when the validation accuracy is not increase much or just before it starts to decrease and validation loss goes up. Once epoch is identified run the following script and replace file in this example with the trained model epoch and timestamp.

```
python3 predict.py /nfs/projects/projcon/outdir/models/attendgru-pc_E09_random_1608163249.h5 --datfile=dataset_random.pkl --gpu=0
```
predicted comments for all models are provided in the predictions folder.
## Step 3 - Ensemble Predictions
A script to run ensembles using mean predictions from two models can be run with this simple modification after isolating two best performing models files.
```
python3 predict_ensemble.py /nfs/projects/projcon/outdir/models/attendgru_E10_random_1609946700.h5 /nfs/projects/projcon/outdir/models/attendgru-pc_E09_random_1608163249.h5 --datfile=dataset_random.pkl --gpu=0
```
predicted comments for all ensembles are provided in the predictions folder.
## Step 4 - Metrics
Bleu and Rouge scores as well a comparison script to insolate maximum improvement have been provided by the name of bleu.py, rougemetric.py and bleucompare.py all of them can be run with the similar commands
```
 python3 rougemetric.py /nfs/projects/projcon/outdir/predictions/predict-attendgru_E10_random_1609946700-attendgru-pc_E09_random_1608163249.txt 
```
```
 python3 bleu.py /nfs/projects/projcon/outdir/predictions/predict-attendgru_E10_random_1609946700-attendgru-pc_E09_random_1608163249.txt 
```
