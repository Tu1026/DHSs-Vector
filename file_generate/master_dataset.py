"""
Code used from https://github.com/pinellolab/DNA-Diffusion/blob/main/notebooks/master_dataset.ipynb
This code is used to create the master dataset
"""
# %%
import requests
import gzip
import numpy as np
import pandas as pd
from Bio import SeqIO
import os
from pathlib import Path
import shutil
import subprocess
# %%

subprocess.call("wget https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz", shell=True)
subprocess.call("gzip -d hg38.fa.gz", shell=True)


            


# %% [markdown]
# # Loading genome

# %%
GENOME_PATH = "hg38.fa"

# %%
class DataSource:
    # Sourced from https://github.com/meuleman/SynthSeqs/blob/main/make_data/source.py

    def __init__(self, data, filepath):
        self.raw_data = data
        self.filepath = filepath

    @property
    def data(self):
        return self.raw_data


class ReferenceGenome(DataSource):
    """Object for quickly loading and querying the reference genome."""

    @classmethod
    def from_path(cls, path):
        genome_dict = {record.id: str(record.seq).upper() for record in SeqIO.parse(path, "fasta")}
        return cls(genome_dict, path)

    @classmethod
    def from_dict(cls, data_dict):
        return cls(data_dict, filepath=None)

    @property
    def genome(self):
        return self.data

    def sequence(self, chrom, start, end):
        chrom_sequence = self.genome[chrom]

        assert end < len(chrom_sequence), (
            f"Sequence position bound out of range for chromosome {chrom}. "
            f"{chrom} length {len(chrom_sequence)}, requested position {end}."
        )
        return chrom_sequence[start:end]


genome = ReferenceGenome.from_path(GENOME_PATH)

# %%
subprocess.call("wget https://www.meuleman.org/DHS_Index_and_Vocabulary_metadata.tsv", shell=True)

# !wget https://www.meuleman.org/DHS_Index_and_Vocabulary_metadata.tsv

# Last row is empty
DHS_Index_and_Vocabulary_metadata = pd.read_table('./DHS_Index_and_Vocabulary_metadata.tsv').iloc[:-1]


# %%
# Contains a 733 row (biosample) x 16 (component) peak presence/abscence matrix (not a binary matrix)
# Used later to map component number within metadata dataframe and find proportion for given component

# Downloading basis
basis_array = requests.get("https://zenodo.org/record/3838751/files/2018-06-08NC16_NNDSVD_Basis.npy.gz?download=1")

with open('2018-06-08NC16_NNDSVD_Basis.npy.gz', 'wb') as f:
    f.write(basis_array.content)

# !gzip -d 2018-06-08NC16_NNDSVD_Basis.npy.gz
subprocess.call("gzip -d 2018-06-08NC16_NNDSVD_Basis.npy.gz", shell=True)

# Converting npy file to csv
basis_array = np.load('2018-06-08NC16_NNDSVD_Basis.npy')
np.savetxt("2018-06-08NC16_NNDSVD_Basis.csv", basis_array, delimiter=",")

# Creating nmf_loadings matrix from csv
nmf_loadings = pd.read_csv('2018-06-08NC16_NNDSVD_Basis.csv', header=None)
nmf_loadings.columns = ['C' + str(i) for i in range(1, 17)]


# Joining metadata with component presence matrix
DHS_Index_and_Vocabulary_metadata = pd.concat([DHS_Index_and_Vocabulary_metadata, nmf_loadings], axis=1)

# %%
DHS_Index_and_Vocabulary_metadata.head()

# %%
COMPONENT_COLUMNS = [
    'C1',
    'C2',
    'C3',
    'C4',
    'C5',
    'C6',
    'C7',
    'C8',
    'C9',
    'C10',
    'C11',
    'C12',
    'C13',
    'C14',
    'C15',
    'C16',
]

DHS_Index_and_Vocabulary_metadata['component'] = (
    DHS_Index_and_Vocabulary_metadata[COMPONENT_COLUMNS].idxmax(axis=1).apply(lambda x: int(x[1:]))
)

# %% [markdown]
# # Creating sequence metadata dataframe

# %% [markdown]
# CAUTION NEXT CELL CAN TAKE UPWARDS OF 10 MINUTES TO RUN

# %%
# File loaded from drive available from below link
mixture_array = requests.get("https://zenodo.org/record/3838751/files/2018-06-08NC16_NNDSVD_Mixture.npy.gz?download=1")

# Downloading mixture array that contains 3.5M x 16 matrix of peak presence/absence decomposed into 16 components
with open('2018-06-08NC16_NNDSVD_Mixture.npy.gz', 'wb') as f:
    f.write(mixture_array.content)
# !gzip -d 2018-06-08NC16_NNDSVD_Mixture.npy.gz
subprocess.call("gzip -d 2018-06-08NC16_NNDSVD_Mixture.npy.gz", shell=True)

# Turning npy file into csv
mixture_array = np.load('2018-06-08NC16_NNDSVD_Mixture.npy').T
np.savetxt("2018-06-08NC16_NNDSVD_Mixture.csv", mixture_array, delimiter=",")

# Creating nmf_loadings matrix from csv and renaming columns
nmf_loadings = pd.read_csv('2018-06-08NC16_NNDSVD_Mixture.csv', header=None, names=COMPONENT_COLUMNS)

# %%
# Loading in DHS_Index_and_Vocabulary_metadata that contains the following information:
# seqname, start, end, identifier, mean_signal, numsaples, summit, core_start, core_end, component
subprocess.call("wget https://www.meuleman.org/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz", shell=True)
subprocess.call("gunzip -d DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz", shell=True)

# !wget https://www.meuleman.org/DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz
# !gunzip -d DHS_Index_and_Vocabulary_hg38_WM20190703.txt.gz

# Loading sequence metadata
sequence_metadata = pd.read_table('./DHS_Index_and_Vocabulary_hg38_WM20190703.txt', sep='\t')

# Dropping component column that contains associated tissue rather than component number (We will use the component number from DHS_Index_and_Vocabulary_metadata)
sequence_metadata = sequence_metadata.drop(columns=['component'], axis=1)

# Join metadata with component presence matrix
df = pd.concat([sequence_metadata, nmf_loadings], axis=1, sort=False)

# %%
# Functions used to create sequence column
def sequence_bounds(summit: int, start: int, end: int, length: int):
    """Calculate the sequence coordinates (bounds) for a given DHS.
    https://github.com/meuleman/SynthSeqs/blob/main/make_data/process.py
    """
    half = length // 2

    if (summit - start) < half:
        return start, start + length
    elif (end - summit) < half:
        return end - length, end

    return summit - half, summit + half


def add_sequence_column(df: pd.DataFrame, genome, length: int):
    """
    Query the reference genome for each DHS and add the raw sequences
    to the dataframe.
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe of DHS annotations and NMF loadings.
    genome : ReferenceGenome(DataSource)
        A reference genome object to query for sequences.
    length : int
        Length of a DHS.

    https://github.com/meuleman/SynthSeqs/blob/main/make_data/process.py
    """
    seqs = []
    for rowi, row in df.iterrows():
        l, r = sequence_bounds(row['summit'], row['start'], row['end'], length)
        seq = genome.sequence(row['seqname'], l, r)

        seqs.append(seq)

    df['sequence'] = seqs
    return df


# Recreating some of the columns from our original dataset
df['component'] = df[COMPONENT_COLUMNS].idxmax(axis=1).apply(lambda x: int(x[1:]))
df['proportion'] = df[COMPONENT_COLUMNS].max(axis=1) / df[COMPONENT_COLUMNS].sum(axis=1)
df['total_signal'] = df['mean_signal'] * df['numsamples']
df['proportion'] = df[COMPONENT_COLUMNS].max(axis=1) / df[COMPONENT_COLUMNS].sum(axis=1)
df['dhs_id'] = df[['seqname', 'start', 'end', 'summit']].apply(lambda x: '_'.join(map(str, x)), axis=1)
df['DHS_width'] = df['end'] - df['start']

# Creating sequence column
df = add_sequence_column(df, genome, 200)

# Changing seqname column to chr
df = df.rename(columns={'seqname': 'chr'})

# Reordering and unselecting columns
df = df[
    [
        'dhs_id',
        'chr',
        'start',
        'end',
        'DHS_width',
        'summit',
        'numsamples',
        'total_signal',
        'component',
        'proportion',
        'sequence',
    ]
]
df

# %%
# Downloading binary peak matrix file from https://www.meuleman.org/research/dhsindex/
# https://drive.google.com/uc?export=download&id=1Nel7wWOWhWn40Yv7eaQFwvpMcQHBNtJ2

# Extracting file from zip
# !gzip -d dat_bin_FDR01_hg38.txt.gz

# subprocess.call("gzip -d dat_bin_FDR01_hg38.txt.gz", shell=True)

# Opening file
binary_matrix = pd.read_table('./dat_bin_FDR01_hg38.txt', header=None)

# Collecting names of cells into a list with fromat celltype_encodeID
celltype_encodeID = [
    row['Biosample name'] + "_" + row['DCC Library ID'] for _, row in DHS_Index_and_Vocabulary_metadata.iterrows()
]

# Renaming columns using celltype_encodeID list
binary_matrix.columns = celltype_encodeID

# %%
master_dataset = pd.concat([df, binary_matrix], axis=1, sort=False)

# %%
master_dataset

# %%
# Save as feather file
master_dataset.to_feather('master_dataset.ftr')


