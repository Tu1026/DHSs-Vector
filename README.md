# DHSs-Vector

How can we use artificial intelligence to understand and predict the role of DNA regions that regulate gene expression in different cell and tissue types, and how they are linked to human diseases? In this project, we will apply a state-of-the-art deep learning model, DNABERT2, to classify DNA sequences based on their cell and tissue specificity. We will also explore the associations between these sequences and various disease phenotypes, using public databases and computational tools. Our prelimimary goal is to explore the differences between disease and healthy DHSs. In the first step of the project we were able to obtain a linear classifer that predicts healthy and disease DHSs with 72% accuracy with only a subset of data.

## Data
We will be mainly wokring with the data from the Meuleman et al. 2020 paper. The data other metadata information are available at https://www.meuleman.org/research/dhsindex/

Additional filtering setups and data setup will be discussed in the next section.

## Getting Started

### Requirements
- OS: Linux
- Python 3.8 or higher
- Access to a GPU with atleast 16GB of memory is highly recommended


### Environment Setup
You can easily replicate the project enviorment by using conda:
```bash
conda env create -f environment.yml
conda activate DHS-disease
```


### Data setup
We will follow the preprocessing steps that the people at PinelloLab had done in https://github.com/pinellolab/DNA-Diffusion/blob/main/notebooks/master_dataset.ipynb

I have modified their code to make them more modular and easier to use. The code is available in the `file_generate` folder. The main file is `master_dataset.py` and directly running the script will generate "master_dataset.ftr" file in the `data` folder. The file provides most information that will be needed. It will also generate other relevant files such as `DHS_Index_and_Vocabulary_metadata.tsv` and `DHS_Index_and_Vocabulary_hg38_WM20190703.txt` which offers more granular information about the DHSs and their associated metadata.


### Pre-training
Run the code in `explore.ipynb` up to the point that generates `train.csv`, `dev.csv` and `test.csv` files. The simply run the following command to train the model:
```bash
sh utils/fine_tune.sh
```
Log into Wandb and enable the argument --report_to wandb to log the results to the wandb dashboard.

See the example pre-training results here https://wandb.ai/ding-group/DNABERT?workspace=user-tu1026


### Analysis
Finish running the code in `explore.ipynb` and then run the code. All of the visualization and results would be generated in the notebook.




## Results/Contributions
The most important results and contributions of the project so far are the following:
- We have obtained a linear classifier that predicts healthy and disease DHSs with 72% accuracy with only a subset of data.
- A proof of concept that DNABERT2 can be used to classify DHSs based on their cell and tissue specificity.
- We fined-tuned the DNABERT2 model to classify DHSs vocabulary based only on sequenced of the motifs.
- The finetune model was able to utilize the latent space of the model more for DHS specific tasks.
- Setup structure that allows us to quickly scale up the model to use more data and more complex models.

**Check out the rendered-ipynb for better viewing experience here** https://htmlpreview.github.io/?https://github.com/Tu1026/DHSs-Vector/blob/main/explore.html


## Notes

The results are only running on a very small subset of the data due to limited access to HPC this week. The results are not final and the model is not fully optimized. The model is also not fully trained on the data. We would expect better performance and better latnet space separation with more data and more training. Finally I also acknowledge that there are many potential directions still to be explored for this task
