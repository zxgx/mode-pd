### NSCC Env Setup
1. Define environment variables: add following lines to `.bashrc` and then run `source ~/.bashrc`
    ```bash
    export SCRATCH=/scratch/users/nus/yourid
    ```

2. Run follwoing command and upload sif file to `$SCRATCH/images`
    ```bash
    mkdir $SCRATCH/images
    mkdir $SCRATCH/venvs
    ```
3. Setup virtual environment
    ```bash
    # launch interactive shell
    module load singularity
    singularity shell --bind $SCRATCH:$SCRATCH --nv $SCRATCH/images/cuda_12.4.1-cudnn-devel-u22.04.sif
    
    # create environment
    python3 -m venv $SCRATCH/venvs/modepd

    # activate environment
    source $SCRATCH/venvs/modepd/bin/activate

    # install packages
    pip install torch accelerate tqdm datasets transformers huggingface_hub wandb

    # setup wandb (paste in API key)
    wandb login

    # download model
    python modepd/download.py
    ```