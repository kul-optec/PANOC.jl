# PANOC.jl

[![Build Status](https://travis-ci.org/kul-forbes/PANOC.jl.svg?branch=master)](https://travis-ci.org/kul-forbes/PANOC.jl)
[![Coverage Status](https://coveralls.io/repos/github/kul-forbes/PANOC.jl/badge.svg?branch=master)](https://coveralls.io/github/kul-forbes/PANOC.jl?branch=master)
[![codecov](https://codecov.io/gh/kul-forbes/PANOC.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/kul-forbes/PANOC.jl)

PANOC is a Newton-type accelerated proximal gradient method for nonsmooth
optimization: this repository contains its generic implementation in Julia.

**Deprecated: an up-to-date implementation of the same algorithm is available as part of [ProximalAlgorithms.jl](https://github.com/kul-forbes/ProximalAlgorithms.jl).**

## Installation

From the Julia REPL, hit `]` to enter the package manager, then

```julia
pkg> add https://github.com/kul-forbes/PANOC.jl
```

## Quick guide

PANOC solves optimization problems of the form

```
minimize f(Ax) + g(x)
```

where `x` is the decision variable, while
* `f` is a smooth function, `g` is a function with easily computable
proximal operator, both of which can be taken from
[ProximalOperators.jl](https://github.com/kul-forbes/ProximalOperators.jl);
* `A` is a linear mapping, e.g. a matrix or an object from linear operator
packages such as
[AbstractOperators.jl](https://github.com/kul-forbes/AbstractOperators.jl),
[LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl),
or [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).

The above problem is solved calling the `panoc` function:

```julia
julia> using PANOC
julia> x_opt, it = panoc(f, A, g, x0)
```

where `x0` is the starting point of the iterations.
This returns the optimal point found, and the number of iterations it took to find it.
The full list of options is described in the docstring, accessible with

```julia
julia> ?panoc
```

## Citing

If you use this package for your publications, please consider including the
following BibTeX entries in the references

```
@inproceedings{stella2017simple,
  author    = {Stella, Lorenzo and Themelis, Andreas and Sopasakis, Pantelis and Patrinos, Panagiotis},
  title     = {A simple and efficient algorithm for nonlinear model predictive control},
  booktitle = {56th IEEE Conference on Decision and Control (CDC)},
  year      = {2017},
  pages     = {1939-1944},
  doi       = {10.1109/CDC.2017.8263933},
  url       = {https://doi.org/10.1109/CDC.2017.8263933}
}
```

```
@misc{stella2018panoc,
  author        = {Stella, Lorenzo},
  title         = {{PANOC}.jl: {N}ewton-type accelerated proximal gradient method in Julia},
  howpublished  = {\url{https://github.com/kul-forbes/PANOC.jl}},
  year          = {2018}
}
```

## References

Stella, Themelis, Sopasakis, Patrinos, [*A simple and efficient algorithm for nonlinear model predictive control*](https://doi.org/10.1109/CDC.2017.8263933), 56th IEEE Conference on Decision and Control (2017).
