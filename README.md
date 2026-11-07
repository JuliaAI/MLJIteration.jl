# MLJIteration.jl 

| Linux | Coverage | Documentation |
| :-----------: | :------: | :-------:|
| [![Build status](https://github.com/JuliaAI/MLJIteration.jl/workflows/CI/badge.svg)](https://github.com/JuliaAI/MLJIteration.jl/actions)| [![codecov.io](http://codecov.io/github/JuliaAI/MLJIteration.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaAI/MLJIteration.jl?branch=master) | [![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaAI.github.io/MLJ.jl/dev/controlling_iterative_models/)|


A package for wrapping iterative models provided by the
[MLJ](https://JuliaAI.github.io/MLJ.jl/dev/) machine
learning framework in a control strategy.

Builds on the generic iteration control tool
[IterationControl.jl](https://github.com/JuliaAI/IterationControl.jl).


## Installation

Included as part of
[MLJ installation](https://JuliaAI.github.io/MLJ.jl/dev/#Installation-1). 

Alternatively, for a "minimal" installation:

```julia
using Pkg
Pkg.add("MLJBase")
Pkg.add("MLJIteration")
Pkg.add("MLJModels")     # to load models
```


## Sample usage

Assuming MLJ.jl and EvoTrees.jl are in your package environment:

```julia
using MLJ

X, y = make_moons(1000)
EvoTreeClassifier = @load EvoTreeClassifier verbosity=0
model = EvoTreeClassifier(rng=123, eta=0.1, colsample=0.5)
iterated_model = IteratedModel(
    model,
    resampling=Holdout(fraction_train=0.8),
    measures=log_loss,
    controls=[
        Step(1),
        Patience(2),
        WithLossDo(loss-> @info "holdout loss: $loss"),
        NumberLimit(50),
    ],
    retrain=true,
)

mach = machine(iterated_model, X, y) |> fit!;
traces = report(mach).traces
holdout_losses = traces[3].loss

```

## Documentation

See the [Controlling Iterative
Models](https://JuliaAI.github.io/MLJ.jl/dev/controlling_iterative_models/)
section of the [MLJ
manual](https://JuliaAI.github.io/MLJ.jl/dev/).

