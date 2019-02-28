# bouligandlevenbergmarquardt

This repository contains implementations of the iterative regularization method described in the paper
[Bouligand-Levenberg-Marquardt iteration for a non-smooth ill-posed problem](https://arxiv.org/abs/1902.10596)
by Christian Clason and Vũ Hữu Nhự.

### Python code

The results in the paper were generated using the provided Python implementation (`BouligandLevenbergMarquardt.py`) (with Python 3.7.2, Numpy 1.17.0, and Scipy 1.3.0) with the following notable differences to allow faster testing:

1. The current code defaults to `N=128`, while the reported results were obtained with `N=512` (see line 24).

2. Plotting is disabled by default but can be enabled by setting `figs = True`  in line 27.

3. Warm starting is used in the semi-smooth Newton method but can be disabled by replacing `F(un,yn)` by `F(un)` in line 140.

To run a representative example (`N=128`, `delta = 1e-4`, `beta=0.005`), run `python3 BouligandLevenbergMarquardt.py`.


### Julia code

We also provide an equivalent [Julia](https://julialang.org) (version 1.1) implementation in the module `BouligandLevenbergMarquardt.jl`. To run the same example, start `julia` in the same directory as the module and enter
```julia
include("./BouligandLevenbergMarquardt.jl")
BouligandLevenbergMarquardt.run_example(128,1e-4,0.005);
```
(Note that the first time this is done, the Julia code will be compiled to native code; subsequent calls to `run_example` (even with changed parameters) will then be much faster.)


### Reference

If you find this code useful, you can cite the paper as

    @article{BouligandLevenbergMarquardt,
        author = {Clason, Christian and Nhu, Vu Huu},
        title = {Bouligand--Levenberg--Marquardt iteration for a non-smooth ill-posed problem},
        year = {2019},
        eprinttype = {arxiv},
        eprint = {1902.10596},
    }


