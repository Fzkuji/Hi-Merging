# Hi-Merging


## Overview

Large Language Models (LLMs) have demonstrated exceptional capabilities across diverse natural language processing (NLP) tasks. The release of open-source LLMs like LLaMA and Qwen has triggered the development of numerous fine-tuned models tailored for various tasks and languages. 

This project explores an important question: **Is it possible to combine specialized models to create a unified model with multi-task capabilities?**

We introduce **Hierarchical Iterative Merging (Hi-Merging)**, a training-free method for unifying different specialized LLMs into a single model. Specifically, Hi-Merging employs model-wise and layer-wise pruning and scaling, guided by contribution analysis, to mitigate parameter conflicts.

## Getting Started with Hi-Merging

This section provides a step-by-step guide on how to use this repository to fine-tune base models, merge them using our **Hierarchical Iterative Merging (Hi-Merging)** method, and evaluate the results.

### Prerequisites

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Fzkuji/Hi-Merging.git
    ```
2.  **Set up Environment:** Ensure you have Python (e.g., 3.9 - 3.12), PyTorch, and CUDA (e.g., 11.8 or 12.4) installed.
3.  **LLaMA Factory & MergeKit:** This project relies on [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) for fine-tuning and [MergeKit](https://github.com/cg123/mergekit) for the merging backbone. Ensure they are correctly installed and accessible, potentially via submodules or separate installations referenced in your scripts. Update paths in configuration files (e.g., `scripts/run_*.sh`).


### Step 1: Model Fine-tuning

Fine-tune a base Large Language Model (LLM) like Qwen2-7B-Instruct or Llama-3-8B-Instruct on specific tasks and languages using LLaMA Factory.

1.  **Prepare Datasets:** Use the notebooks in `datasets` to generate proper data format for LLaMA-Factory. Then place the generated json files under `LLaMA-Factory/data` and add the declaration in the configuration file `dataset_info.json`.
2.  **Configure Training:** Modify the LLaMA Factory configuration file (e.g., `examples/train_lora/qwen2_lora_sft.yaml` referenced in your scripts) to specify the base model, dataset, LoRA parameters (e.g., rank=8, alpha=16, dropout=0.01), learning rate (e.g., 1.0e-4), output directory, etc.
3.  **Run Fine-tuning:** Execute the fine-tuning script. You might have a dedicated script like `scripts/run_finetuning.sh` or integrate this step into a larger experiment script (like parts of `scripts/run_experiments.sh`).

    ```bash
    llamafactory-cli train examples/train_lora/your_config.yaml
    ```
    Repeat this step for each specialized model you want to create (e.g., one for English MCQA, one for Chinese QA).
4.  **Export Models** If you fine-tuning the LLM with LoRA, you should merge it back to the LLM and save a copy of the model for merging.

    ```bash
    llamafactory-cli export examples/merge_lora/qwen2_lora_sft.yaml
    ```

### Step 2: Model Merging (Hi-Merging)

Combine the fine-tuned models using our Hi-Merging approach, which involves hierarchical pruning and scaling guided by contribution analysis.

1.  **Configure Merging (MergeKit YAML):** Prepare a MergeKit configuration file (e.g., `examples/ties.yml`). This file will list the models to be merged and initial parameters (like weights for scaling, density for pruning, etc.).
    Note that Hi-Merging will *programmatically modify* aspects of this configuration (especially weights/parameters layer-by-layer) based on its analysis. The `ConsensusMethod` is `sum`.
2.  **Run Hi-Merging Script:** Execute the main script that implements Hi-Merging. This script performs the following internally:
    *   **Model-wise Pruning & Scaling:** Applies initial pruning (`p`) and scaling (`s`) to the delta vectors (`θ_finetuned - θ_base`) of each model to reduce noise and overfitting effects.
        *   Use `run_model_wise.sh`.
    *   **Layer-wise Iterative Conflict Resolution:**
        *   Uses **Contribution Analysis** (calculating α and β impacts, Section 3.2.1) to identify conflicts layer by layer.
            *    Use `run_single_contribution.sh` to calculate the contribution before merging.
            *    Use `run_merge_contribution.sh` to calculate the contribution after merging.
        *   Iteratively processes the most conflicted layers, applying layer-specific pruning, scaling, or dropping based on the conflict type.
        *   Leverages MergeKit for the final combination according to the decisions made by the Hi-Merging logic.

            ```bash
            mergekit-yaml ./examples/ties.yml ./output_model/qwen2_lora_sft/merge_en_zh --allow-crimes --cuda
            ```


## Project Structure

Below is an organized description of the project's directory structure, explaining the role and contents of each key folder and file.

### 1. datasets

Contains various medical-related datasets for fine-tuning and evaluation:

- **Chinese Medical Datasets**:
    - `cmedqa2/`: Chinese medical QA dataset
    - `cmexam/`: Chinese medical examination dataset
    - `cmb/`: Base Chinese medical dataset
    - `chinesemedical/`: General dataset for Chinese medical tasks

- **English Medical Datasets**:
    - `healthcaremagic/`: English open-domain Medical QA dataset
    - `medqa/`: Medical multiple-choice QA dataset
    - `medical/`: General English medical tasks dataset

We also find a financial dataset that is suitable for our experimental setup. Although it was not involved in the current experiments, it could be quite appropriate for future domain-specific fine-tuning tasks:
- **Financial Dataset**:
    - `goldentouchstone/`: A financial bilingual evaluation benchmark dataset that follows a similar QA format to our medical datasets, making it compatible with our training pipeline for potential merging.

### 2. models


- `qwen2.ipynb`: Notebook primarily used for testing and determining the optimal prompt configurations for each fine-tuned model based on Qwen2 open-source models.

### 3. scripts

Command-line scripts to automate the merging experiments and analysis:

- `run_model_wise.sh`: Script for model-wise pruning and scaling
- `run_single_contribution.sh`: Script evaluating single-layer contributions and impacts before merging
- `run_merge_contribution.sh`: Script running contribution analysis after merging
