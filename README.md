# PANOC.jl

PANOC is a Newton-type accelerated proximal gradient method for nonsmooth optimization: this repository contains its generic implementation in Julia.

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
[ProximalOperator.jl](https://github.com/kul-forbes/ProximalOperators.jl);
* `A` is a linear mapping, e.g. a matrix or an object from linear operator
packages such as
[AbstractOperators.jl](https://github.com/kul-forbes/AbstractOperators.jl),
[LinearMaps.jl](https://github.com/Jutho/LinearMaps.jl),
or [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).

All available options are described in the docstring, accessible with:

```julia
julia> using PANOC
julia> ?panoc
```

## Examples

## References

Stella, Themelis, Sopasakis, Patrinos, [*A simple and efficient algorithm for nonlinear model predictive control*](https://doi.org/10.1109/CDC.2017.8263933), 56th IEEE Conference on Decision and Control (2017).
