# MLJIteration.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/ablaom/MLJIteration.jl/workflows/CI/badge.svg)](https://github.com/ablaom/MLJIteration.jl/actions)| [![codecov.io](http://codecov.io/github/ablaom/MLJIteration.jl/coverage.svg?branch=master)](http://codecov.io/github/ablaom/MLJIteration.jl?branch=master) |

A package for wrapping iterative models provided by the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine
learning framework in a control strategy.

Builds on the generic iteration control tool
[IterationControl.jl](https://github.com/ablaom/IterationControl.jl).


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

What follows is a draft of documentation to be added to the [MLJ
manual](https://alan-turing-institute.github.io/MLJ.jl/dev/).

---
# Controling Iterative Models

Iterative supervised machine learning models are usually trained until
an out-of-sample estimate of the performance satisfies some stopping
criterion, such as `k` consecutive deteriorations of the performance
(see [`Patience`](@ref) below). A more sophisticated kind of control
might dynamically mutate parameters, such as a learning rate, in
response to the behaviour of these estimates. Some iterative models
will enable limited "under the hood" control through hyper-parameter
choices (with the method and options for doing so varying from model
to model). But often it is up to the user is to arrange control, which
may amount to manually experimenting with the iteration parameter.

In response to this ad hoc state of affairs, MLJ provides a uniform
and feature-rich interface for controlling any iterative model that
exposes its iteration parameter as a hyper-parameter, and which
implements the "warm restart" behaviour described in [Machines](@ref).

As in [Tuning models](@ref), iteration control in MLJ is implemeted as
a model wrapper, which allows composition with other meta-algorithms.
Ordinarily, the wrapped model behaves just like the original model,
but with the training occuring on a subset of the provided data, and
with the iteration parameter automatically determined by the controls
specified in the wrapper.


SIMPLE EXAMPLE

## Controls provided

In the table below, `mach` is the machine being iterated, constructed
by binding the supplied data to the `model` specified in the
`IteratedModel` wrapper, but trained in each iteration on a subset of
the data, according to the value of the `resampling` hyper-parameter
of the wrapper.


control                                              | description                                                                             | can trigger a stop
-----------------------------------------------------|-----------------------------------------------------------------------------------------|--------------------
[`Step`](@ref)`(n=1)`                                | Train model for `n` more iterations                                                     | no
[`TimeLimit`](@ref)`(t=0.5)`                         | Stop after `t` hours                                                                    | yes
[`NumberLimit`](@ref)`(n=100)`                       | Stop after `n` applications of the control                                              | yes
[`NotANumber`](@ref)`()`                             | Stop when `NaN` encountered                                                             | yes
[`Threshold`](@ref)`(value=0.0)`                     | Stop when `loss < value`                                                                | yes
[`GL`](@ref)`(alpha=2.0)`                            | ★ Stop after "GeneralizationLossDo" exceeds `alpha`                                      | yes
[`Patience`](@ref)`(n=5)`                            | ★ Stop after `n` consecutive loss increases                                              | yes
[`PQ`](@ref)`(alpha=0.75, k=5)`                      | ★ Stop after "Progress-modified GL" exceeds `alpha`                                      | yes
[`Info`](@ref)`(f=identity)`                         | Log to `Info` the value of `f(mach)`, where `mach` is current machine                   | no
[`Warn`](@ref)`(predicate; f="")`                    | Log to `Warn` the value of `f` or `f(mach)` if `predicate(mach)` holds                  | no
[`Error`](@ref)`(predicate; f="")`                   | Log to `Error` the value of `f` or `f(mach)` if `predicate(mach)` holds and then stop   | yes
[`Callback`](@ref)`(f=_->nothing)`                   | Call `f(mach)`                                                                          | yes
`WithNumberDo`](@ref)`(f=n->@info(n))`               | Call `f(n + 1)` where `n` is number of previous calls                                   | yes
[`WithIterationsDo`](@ref)`(f=x->@info("loss: $x"))` | Call `f(i)`, where `i` is number of iterations                                          | yes
[`WithLossDo`](@ref)`(f=x->@info(x))`                | Call `f(loss)` where `loss` is the current loss                                         | yes
[`WithTrainingLossesDo`](@ref)`(f=v->@info(v))`      | Call `f(v)` where `v` is the current batch of training losses                           | yes
[`Save`](@ref)`(filename="machine.jlso")`            | Save current machine to `machine1.jlso`, `machine2.jslo`, etc (or similar)              | yes

> Table 1. Atomic controls. Some advanced options omitted.

★ For more these controls see [Prechelt, Lutz
 (1998)](https://link.springer.com/chapter/10.1007%2F3-540-49430-8_3):
 "Early Stopping - But When?", in *Neural Networks: Tricks of the
 Trade*, ed. G. Orr, Springer.

**Stopping option.** All the following controls trigger a stop if the
provided function `f` returns `true` and `stop_if_true=true` is
specified in the constructor: `Callback`, `WithNumberDo`,
`WithLossDo`, `WithTrainingLossesDo`.

There are also three control wrappers to modify a control's behavior:

wrapper                                            | description
---------------------------------------------------|-------------------------------------------------------------------------
`IterationControl.skip(control, predicate=1)`      | Apply `control` every `predicate` applications of the control wrapper (can also be a function; see doc-string)
`IterationControl.debug(control)`                  | Apply `control` but also log its state to `Info` (at any `verbosity` level)
`IterationControl.composite(controls...)`          | Apply each `control` in `controls` in sequence; mostly for under-the-hood use

> Table 2. Wrapped controls

```@docs
Step
TimeLimit
NumberLimit
NotANumber
Threshold
GL
PQ
Info
Warn
Error
Callback
WithNumberDo
WithIterationsDo
WithLossDo
WIthTrainingLossesDo
Save
```

## API

```@docs
IteratedModel
```


