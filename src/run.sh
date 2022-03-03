#!/bin/bash

for DS in "moodle_combine_cla.csv"; do
  for FEA in "combine"; do
    for TREC in 0.60 0.70 0.80 0.85 0.90 0.95 0.99 1.0; do
      for SEED in $(seq 0 1 29); do
        /home/vitor-apolinario/anaconda3/envs/py2/bin/python /home/vitor-apolinario/Desktop/harmless/Vuls_HPC/src/runner.py error_hpcc_feature_ds $FEA $SEED $DS $TREC &
        sleep 5
      done
    done
  done
done
