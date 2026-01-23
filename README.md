# Approximate full conformal prediction in RKHS

Please check the notebooks in `test/conformal_prediction/`
to see how prediction regions are computed using kernel regression with ridge regularization.

The experiments done in the paper can be reproduced by
the notebooks in `experiments/`.

`experiments/prediction_region/`
contains the notebook that compares
the length of different prediction regions
produced by different methods (see Figure 4 in the article).

`experiments/thickness/`
contains the notebooks that compare
evolution of upper bounds on the thickness
for: UStableCP, LocStableCP and InfluenceFunctionCP.

To run the script, one must download this repo
and install the required packages in `requirement.txt`.