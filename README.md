# MLJIteration.jl

| Linux | Coverage |
| :-----------: | :------: |
| [![Build status](https://github.com/ablaom/MLJIteration.jl/workflows/CI/badge.svg)](https://github.com/ablaom/MLJIteration.jl/actions)| [![codecov.io](http://codecov.io/github/ablaom/MLJIteration.jl/coverage.svg?branch=master)](http://codecov.io/github/ablaom/MLJIteration.jl?branch=master) |

A package for wrapping iterative models provided by the
[MLJ](https://alan-turing-institute.github.io/MLJ.jl/dev/) machine
learning framework in a control strategy.

Builds on the generic iteration control tool
[IterationControl.jl](https://github.com/ablaom/IterationControl.jl).

Not registered  and under construction.


## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/ablaom/MLJIteration.jl")
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


## Basic use

As in [Tuning models](@ref), iteration control in MLJ is implemeted as
a model wrapper, which allows composition with other meta-algorithms.
Ordinarily, the wrapped model behaves just like the original model,
but with the training occuring on a subset of the provided data (to
allow computation of an out-of-sample loss) and with the iteration
parameter automatically determined by the controls specified in the
wrapper.

By setting `retrain=true` one can ask that the wrapped model retrain
on *all* supplied data, after learning the appropriate number of
itertations from the controlled training phase:

```@example gree
using MLJ
using MLJIteration

X, y = make_moons(1000, rng=123)
EvoTreeClassifier = @load EvoTreeClassifier verbosity=0

iterated_model = IteratedModel(model=EvoTreeClassifier(rng=123, η=0.005),
                               resampling=Holdout(rng=123),
                               measures=log_loss,
                               iteration_parameter=:nrounds,
                               controls=[Step(5),
                                         Patience(2),
                                         NumberLimit(100)],
                               retrain=true)

mach = machine(iterated_model, X, y) |> fit!;
```

As detailed under [`IteratedModel`](@ref) below, the specified
`controls` are repeatedly applied in sequence to a *training machine*,
constructed under the hood, until one of the controls triggers a
stop. Here `Step(5)` means "Compute 5 more iterations" (in this case
starting from none); `Patience(2)` means "Stop at the end of the
control cycle if there have been 2 consecutive drops in the log loss";
and `NumberLimit(100)` is a safeguard ensuring a stop after 100
control cycles (500 iterations). See [Controls provided](@ref) below
for a complete list.

Because iteration is implemented as a wrapper, the "self-iterating"
model can be evaluated using cross-validation, say, and the number of
iterations on each fold will generally be different:

```julia
e = evaluate!(mach, resampling=CV(nfolds=3), measure=log_loss, verbosity=0);
map(e.report_per_fold) do r
    r.n_iterations
end
```

Alternatively, one might wrap the self-iterating model in a tuning
strategy, using `TunedModel`; see [Tuning Models](@ref).


## Controls provided

In the table below, `mach` is the *training machine* being iterated, constructed
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

## Custom controls

### Example 1 - Iterating from a list of iteration parameter values

Below we define a control, `IterateFromList(list)`, to train, on the
each application of the control, until the iteration count reaches
the next value on a user-specified list, triggering a stop when the
list is exhausted.

In the code, `wrapper` is an object that wraps the training machine
(see above), which is accessed by `wrapper.machine`, but which also
contains other information, such as the current performance
evaluation object, `wrapper.evaluation`, used here. See more under
[`The training machine wrapper`](@ref) below.

```julia
struct
    IterateFromList list::Vector{<:Int} # list of iteration parameter values
    IterateFromList(v) = new(unique(sort(v)))
end

function MLJIteration.update!(control::IterateFromList, m, verbosity)
    Δi = control.list[1]
    verbosity > 1 && @info "Training $Δi more iterations. "
    MLJIteration.train!(m, Δi)
    return (index = 2, )
end

function MLJIteration.update!(control::IterateFromList, m, verbosity, state)
    index = state.positioin_in_list
    Δi = control.list[i] - m.n_iterations
    verbosity > 1 && @info "Training $Δi more iterations. "
    MLJIteration.train!(m, Δi)
    return (index = index + 1, )
end
```

The first `update` method will be called the first time the control is
applied, returning an initialized "state", which is passed to the
second `update` method, which is called on subsequent control
applications (and which returns an updated "state"). In this example
the two definitions can actually be combined into the one:

```julia
function MLJIteration.update!(control::IterateFromList,
                              m,
                              verbosity,
                              state=(index = 1, ))
    index = state.index
    Δi = control.list[index] - m.n_iterations
    verbosity > 1 && @info "Training $Δi more iterations. "
    MLJIteration.train!(m, Δi)
    return (index = index + 1, )
end
```

We also need to implement a `done` method to say when the control is
to trigger a stop:

```julia
MLJIteration.done(control::IterateFromList, state) =
    state.index > length(control.list)
```

### Example 2 - Basing a stop on multiple measures

When one specifies a vector `measures=...` in the `IteratedModel`
constructor, only the *first* measure is used to define the "loss"
used by stopping controls provided by MLJIteration.jl. The following
example defines a new control to trigger a stop when *all* of the
measures satisfy the [`NumberSinceBest`](@ref) criterion.

For simplicity we are assuming all the measures have the `:loss`
orientation (lower is better).

```julia
struct MultiSinceBest
    n::Integer
end

function MLJIteration.update(control::MultiSinceBest, wrapper, verbosity)
    e = wrapper.evaluation  # current evaluation object
    measures = e.measure
    best_losses = e.measurement
    numbers_since_best = zeros(Int, length(measures))
    return (measures = measures,
            best_losses = best_losses,
            numbers_since_best = numbers_since_best)
end

function MLJIteration.update(control::MultiSinceBest, wrapper, verbosity, state)
    measures, best_losses, numbers_since_best = state
    e = wrapper.evaluation  # current evaluation object
    losses = e.measurement
    for k in eachindex(measures)
        if losses[k] < best_losses[k]
            best_losses[k] = losses[k]
            number_since_best[k] = 0
        else
            numbers_since_best[k] += 1
        end
    end
    return (measures = measures,
            best_losses = best_losses,
            numbers_since_best = numbers_since_best)
end
```

### Example 3 - Cyclic learning rates


```julia
struct CylicLearningStep{F<:AbstractFloat}
        n::Int           # number of cycles of learning rate mutations
        stepsize::Int    # twice this is the cycle period
        min_lr::F        # lower learning rate
        max_lr::F        # upper learning rate
    learning_rate_parameter::Union{Symbol,Expr}
end

```








## API

```@docs
IteratedModel
```
