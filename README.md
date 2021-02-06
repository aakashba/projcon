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

## Step 2 - Predictions
## scripts for firstwords variations can be found at
## Step 3 - Ensemble Predictions
## Step 4 - Metrics
