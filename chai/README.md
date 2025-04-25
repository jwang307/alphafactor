## Quick Start to Running Chai-1 for Structure Prediction

### Prerequisites

1. Install dependencies. Chai-1 requires a GPU and memory capacity detailed in the [Chai-1 repository](https://github.com/chaidiscovery/chai-lab?tab=readme-ov-file#installation).
```bash
pip install chai_lab==0.6.1
```
2. Run Chai-1:
 - The comand line interface for chai offers easy structure prediction from a fasta file:
 ```bash
 # simple fold without msa
 chai-lab fold input.fasta output_folder
 # fold with msa and templates
 chai-lab fold --use-msa-server --use-templates-server input.fasta output_folder
 ```
 - For more control over the inference process, check out `run_chai.predict`, which is a wrapper function returning the pdb structure and relevant scores. An example of how to use the script is shown in `run_chai.py` and can be executed with:
 ```bash
 python run_chai.py
 ```
 
