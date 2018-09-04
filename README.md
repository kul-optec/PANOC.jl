# PANOC.jl

PANOC is a Newton-type accelerated proximal gradient method for nonsmooth optimization: this repository contains its generic implementation in Julia.

## Installation

From the Julia REPL, hit `]` to enter the package manager, then

```julia
pkg> add https://github.com/kul-forbes/PANOC.jl
```

## Quick guide

```julia
julia> using PANOC
julia> ?panoc
```

## References

Stella, Themelis, Sopasakis, Patrinos, [*A simple and efficient algorithm for nonlinear model predictive control*](https://doi.org/10.1109/CDC.2017.8263933), 56th IEEE Conference on Decision and Control (2017).
