# MLJIteration.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/ablaom/MLJIteration.jl/workflows/CI/badge.svg)](https://github.com/ablaom/MLJIteration.jl/actions)| [![codecov.io](http://codecov.io/github/ablaom/MLJIteration.jl/coverage.svg?branch=master)](http://codecov.io/github/ablaom/MLJIteration.jl?branch=master) |

A package for wrapping iterative models provided by the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine
learning framework in a control strategy.

Builds on the generic iteration control tool
[IterationControl.jl](https://github.com/ablaom/IterationControl.jl).

&#128679;

## Installation

```julia
using Pkg
Pkg.add("MLJIteration")
```

## Usage

```julia
using MLJ
using MLJIteration
```

Do `?IteratedModel` for details.

## Provided Controls

See
[here](https://github.com/ablaom/IterationControl.jl#controls-provided)
for the complete list.
