# Tasknet for MNIST

This codes the main experimental result (task-aware and end-to-end training).
Basically, we take a pre-trained DNN and pair it with a VAE and create a composite model. The goal is to vary the latent dimension z and see how much we can
compress, but not sacrifice sensing task accuracy.

## Install dependencies

`pip install -r requirements.txt`

## Notation

Composite model: VAE + DNN

TaskNet: DNN weights are fixed, VAE can be trained at-will

End to End Model: DNN, VAE weights are fixed to be trained at will

## Code

task_aware_run_MNIST.sh
    - task-aware training for various z_dim latent sizes
    - calls:
        - tasknet_full.py

task_agnostic_run_MNIST.sh
    - task-agnostic training for various z_dim latent sizes
    - calls:
        - tasknet_full.py

end_to_end_MNIST.sh
    - end to end training, exact same as above but DNN not fixed
    - calls:
        - tasknet_full.py

CompositeModel.py
    - utils that link a VAE + CNN together and evaluate model
