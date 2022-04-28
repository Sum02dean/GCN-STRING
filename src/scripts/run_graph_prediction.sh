#!/bin/bash

/mnt/mnemo5/sum02dean/mambaforge/envs/gcn_env/bin/python gcn.py \
--model_name model_0 \
--output_directory ../outputs \
--batch_size 50 \
--epochs 50 \
--samples = args.num_samples \

# The maximum number of samples is 10834 (balanced +/-)
# The species is e.coli
# The DCA is not included directly but just used to create inter-residues edges