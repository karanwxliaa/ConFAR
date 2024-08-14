# ConFAR: Continual Multi-Modal Facial Expression Analysis Using Causal Rehearsal and Multi-Task Learning

## Overview
This repository contains the code implementation for the paper titled **ConFAR: Continual Multi-Modal Facial Expression Analysis Using Causal Rehearsal and Multi-Task Learning**. The paper proposes a novel architecture for lifelong multi-modal facial expression analysis that integrates Continual Learning (CL), Multi-Task Learning (MTL), and causal reasoning techniques to minimize catastrophic forgetting while learning multiple tasks incrementally.

## Table of Contents
1. [Installation](#installation)
2. [Dataset](#dataset)
3. [Training](#training)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgement](#acknowledgement)



## Installation

### Step 1: Clone the Repository
First, clone the repository from GitHub:

```bash
git clone https://github.com/karanwxliaa/ConFAR.git
```

Navigate into the repository directory:

```bash
cd ConFAR
```

### Step 2: Create a Conda Environment
Ensure you have Conda installed. If not, you can install it from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

Create a new Conda environment with Python 3.9:

```bash
conda create -n confar_env python=3.9
```

Activate the environment:

```bash
conda activate confar_env
```

### Step 3: Install Required Packages
Once the environment is activated, install the required packages using pip:

```bash
pip install -r requirements.txt
```


Make sure to use a Python environment compatible with PyTorch.

## Dataset
The code utilizes multi-modal facial expression datasets, particularly Affwild2, which includes data for Action Unit detection, Arousal-Valence estimation, and Emotion Recognition tasks. The dataset is not included in this repository and needs to be acquired separately.

1. **Affwild2 Dataset**: [Download from the official source](https://ibug.doc.ic.ac.uk/resources/aff-wild2/). Place the dataset files in the `Data/` directory.

## Training
The repository is divided into modules for the different components of the ConFAR framework:

### Continual Learning
To train the ConFAR baseline model 
```bash
python main.py --dataset single --model_type vgg16face --model_name vgg16_pt
```

### Continual Learning
Each CL strategy is implemented in separate scripts and can be trained using the following command:
```bash
python main.py --agent_name = (Naive_Rehearsal / LGR / EWC / SI)
```


## Results
The results, including performance metrics and comparison with baseline models, are stored in the `Results/` directory. The results are organized by the type of learning strategy applied.


## Citation
If you find this work useful in your research, please consider citing:

```bibtex
@article{ConFAR,
  title={ConFAR: Continual Multi-Modal Facial Expression Analysis Using Causal Rehearsal and Multi-Task Learning},
  author={Jaskaran Singh Walia and Nikhil Churamani and Jiaee Cheong and Hatice Gunes},
  journal={[Journal/Conference]},
  year={2024}
}
```

## Acknowledgement
**Funding**: JS. Walia contributed to this work while undertaking a remote visiting studentship at the Department of Computer Science and Technology, University of Cambridge. N. Churamani and H. Gunes are supported by Google under the GIG Funding Scheme. <br> <br>
**Open Access**: For open access purposes, the authors have applied a Creative Commons Attribution (CC BY) licence to any Author-Accepted Manuscript version arising. <br> <br>
**Data Access Statement**: This study involves secondary analyses of the existing datasets, that are described and cited in the text
