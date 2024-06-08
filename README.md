# Image Analysis

## Description

Contains software that demonstrates various techniques related to image analysis, signal processing, and compressed sensing.

## How to use the software

First reproduce the development environment using conda by running

`conda env create -f environment.yml`

Then the main code scripts can be run as follows:

`python src/segment.py` runs the segmentation code,

`python src/inverse_problems.py` runs the code for the rest of the problems besides the LGD training pipeline,

`python src/lgd.py -m demo` runs the LGD training script.

In the last case, the option `demo` makes the code load in a pre-trained model checkpoint from epoch 1990 (this was produced on Google Colaboratory),
and runs the last 10 epochs.
This gives a quick demonstration of the code that is feasible on a CPU.
To run the entire training pipeline use option `full` instead,
although this will take much longer if no GPU is present.
