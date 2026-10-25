const IterationResamplingTypes =
    Union{Holdout,InSample,Nothing,MLJBase.TrainTestPairs}


## TYPES AND CONSTRUCTOR

mutable struct DeterministicIteratedModel{M<:Deterministic,L} <: MLJBase.Deterministic
    model::M
    controls
    resampling # resampling strategy
    measure
    weights::Union{Nothing,Vector{<:Real}}
    class_weights::Union{Nothing,Dict{Any,<:Real}}
    operation
    retrain::Bool
    check_measure::Bool
    iteration_parameter::Union{Nothing,Symbol,Expr}
    cache::Bool
    logger::L
    compact_traces::Bool
    warm_restart_extends_traces::Bool
end

mutable struct ProbabilisticIteratedModel{M<:Probabilistic,L} <: MLJBase.Probabilistic
    model::M
    controls
    resampling # resampling strategy
    measure
    weights::Union{Nothing,AbstractVector{<:Real}}
    class_weights::Union{Nothing,Dict{Any,<:Real}}
    operation
    retrain::Bool
    check_measure::Bool
    iteration_parameter::Union{Nothing,Symbol,Expr}
    cache::Bool
    logger::L
    compact_traces::Bool
    warm_restart_extends_traces::Bool
end

const ERR_MISSING_TRAINING_CONTROL =
    ArgumentError("At least one control must be a training control "*
                  "(have type `$TrainingControl`) or be a "*
                  "custom control that calls IterationControl.train!. ")

const ERR_TOO_MANY_ARGUMENTS =
    ArgumentError("At most one non-keyword argument allowed. ")
const EitherIteratedModel{M,L} =
    Union{DeterministicIteratedModel{M,L},ProbabilisticIteratedModel{M,L}}
const ERR_NOT_SUPERVISED =
    ArgumentError("Only `Deterministic` and `Probabilistic` "*
                  "model types supported.")
const ERR_NEED_MEASURE =
    ArgumentError("Unable to deduce a default measure for specified model. "*
                  "You must specify `measure=...`. ")
const ERR_NEED_PARAMETER =
    ArgumentError("Unable to determine the name of your model's iteration "*
                  "parameter. Please specify `iteration_parameter=...`. This "*
                  "must be a `Symbol` or, in the case of a nested parameter, "*
                  "an `Expr` (as in `booster.nrounds`). ")
const ERR_MODEL_UNSPECIFIED = ArgumentError(
"Expecting atomic model as argument, or as keyword argument `model=...`, "*
    "but neither detected. ")

const WARN_POOR_RESAMPLING_CHOICE =
    "Training could be very slow unless "*
    "`resampling` is `Holdout(...)`, `nothing`, or "*
    "a vector of the form `[(train, test),]`, where `train` and `test` "*
    "are valid row indices for the data, as in "*
    "`resampling = [(1:100, 101:150),]`. "

const WARN_POOR_CHOICE_OF_PAIRS =
    "Training could be very slow unless you limit the number of `(train, test)` pairs "*
    "to one, as in resampling = [(1:100, 101:150),]. Alternatively, "*
    "use a `Holdout` resampling strategy. "

err_bad_iteration_parameter(p) =
    ArgumentError("Model to be iterated does not have :($p) as an iteration parameter. ")

"""
    IteratedModel(
        model;
        controls=MLJIteration.DEFAULT_CONTROLS,
        resampling=Holdout(),
        measure=nothing,
        retrain=false,
        advanced_options...,
    )

Wrap the iterative, supervised `model` in the specified iteration `controls`. Here `model`
should support iteration, which is true if `iteration_parameter(model)` is different from
`nothing`.

The wrapped model behaves much like `model` except that the iteration hyperparameter is
"learned" rather than fixed ahead of time by the user. Additionally, unless
`retrain=true`, some of the data presented for training will have been used exclusively to
calculate an out-of-sample loss for stopping criteria, and so not directly used to infer
other learned parameters used in subsequent prediction.

The controls will include a `Step(k)` control, specifying the number, `k`, of iterations
between "control cycles", and one or more stopping criterion, such as `Patience(6)`, which
can signal that the current control cycle will be the last. Other assorted callback
controls are available. For more on the training logic, see "Training" under "Extended
help" below.

```julia-repl
using MLJBase, MLJIteration, StatisticalMeasures # or just `using MLJ`
using EvoTrees
using Dates

X, y = make_moons(); # a table and a vector
model = EvoTreeClassifier(seed=123)
controls = [Step(2), Threshold(0.3), InvalidValue(), TimeLimit(Second(10))]
iterated_model = IteratedModel(
    model;
    controls,
    resampling=Holdout(fraction_train=0.8),
    measure=brier_loss,
)
mach = machine(iterated_model, X, y)

julia> fit!(mach)
[ Info: Training machine(ProbabilisticIteratedModel(model = EvoTreeClassifier(loss = mlogloss, …), …), …).
[ Info: Training machine(ProbabilisticIteratedModel(model = EvoTreeClassifier(loss = mlogloss, …), …), …).
[ Info: No iteration parameter specified. Using `iteration_parameter=:(nrounds)`.
[ Info: final loss: 0.27815571427345276
[ Info: Stop triggered by Threshold(0.3) stopping criterion.
[ Info: Total of 4 iterations.

julia> report(mach).n_iterations
4

julia> traces = report(mach).traces
4-element Vector{Any}:
 ControlTrace{<:Step}, ending (new_iterations = 4, new_steps = 2)
 ControlTrace{<:Threshold}, ending (loss = 0.27815571427345276, excess = -0.02184428572654723)
 ControlTrace{<:InvalidValue}, ending (encountered = nothing,)
 ControlTrace{<:TimeLimit}, ending (time = DateTime("2026-10-23T17:19:04.844"),)

julia> traces[2] |> pretty
┌────────────┬────────────┐
│ loss       │ excess     │
│ Float64    │ Float64    │
│ Continuous │ Continuous │
├────────────┼────────────┤
│ 0.339146   │ 0.0391458  │
│ 0.278156   │ -0.0218443 │
└────────────┴────────────┘
```

Available controls: $CONTROLS_LIST.

# Extended help

Note there is a generally a difference between the total number of iterations and the
number of control cycles, because each control cycle begins with `k` iterations of the
wrapped model, when `controls = [`Step(k)`, ...]`. In control names (e.g, `NumberLimit`
and `WithNumberDo`) and in documentation, "number" means "number of steps", not number of
iterations.

# Options

- `controls=$DEFAULT_CONTROLS`: A vector of controls, which are summarized at the
  "[Controls-provided](https://juliaai.github.io/MLJ.jl/dev/controlling_iterative_models/#Controls-provided)"
  section of the MLJ manual, but query individual doc-strings for details and advanced
  options. For creating your own controls, refer to the "[Custom
  controls](https://juliaai.github.io/MLJ.jl/dev/controlling_iterative_models/#Custom-controls)"
  section of the MLJ manual.

- `resampling=Holdout(fraction_train=0.7)`: The default resampling holds back 30% of data
  for computing an out-of-sample estimate of performance (the "loss") for loss-based
  controls such as `WithLossDo`. Specify `resampling=InSample()` to use all data for
  training and to compute a training loss as a proxy for the out-of-sample loss. If
  `supports_training_losses(model) == true`, you can specify `resampling=nothing` to
  instead use the training loss reported by the model and avoid the extra
  computation. Otherwise, `resampling` must have type `Holdout` or be a vector with one
  element of the form `(train_indices, test_indices)`.

- `measure=nothing`: StatisticalMeasures.jl compatible measure for estimating model
  performance (the "loss", but the orientation is immaterial - i.e., this could be a
  score). Inferred by default. Ignored if `resampling=nothing`.

- `retrain=false`: If `retrain=true` or `resampling=nothing`, `iterated_model` behaves
  exactly like the original `model` but with the iteration parameter automatically
  selected ("learned"). That is, the model is retrained on *all* available data, using the
  same number of iterations, once controlled iteration has stopped. This is typically
  desired if wrapping the iterated model further, or when inserting in a pipeline or other
  composite model. If `retrain=false` (default) and `resampling isa Holdout`, then
  `iterated_model` behaves like the original model trained on a subset of the provided
  data.

- `compact_traces=false`: Whether to record full traces of each control or to retain only
  the most recent state; traces appear in the `IteratedModel` machine report.

- `warm_restart_extends_traces=false`: If `true`, then in warm restarts traces are
  extended rather than created afresh. This option is only supported if the control types
  remain unchanged on the new call to `fit!`. Traces appear in the `IteratedModel` machine
  report.

- `weights=nothing`: per-observation weights to be passed to `measure` where supported; if
  unspecified, these are understood to be uniform.

- `class_weights=nothing`: class-weights to be passed to `measure` where supported; if
  unspecified, these are understood to be uniform.

- `operation=nothing`: Operation, such as `predict` or `predict_mode`, for computing
  target values, or proxy target values, for consumption by `measure`; automatically
  inferred by default.

- `check_measure=true`: Specify `false` to override checks on `measure` for compatibility
  with the training data.

- `iteration_parameter=nothing`: A symbol, such as `:epochs`, naming the iteration
  parameter of `model`; inferred by default. Note that the actual value of the iteration
  parameter in the supplied `model` is ignored; only the value of an internal clone is
  mutated during training the wrapped model.

- `cache=true`: Whether or not model-specific representations of data are cached in
  between iteration parameter increments; specify `cache=false` to prioritize memory over
  speed.

- `logger=default_logger()`: a logger for externally reporting model performance
  evaluations, such as an `MLJFlow.Logger` instance. On startup,
  `default_logger()=nothing`; use `default_logger(logger)` to set a global logger.


# Training

Training an instance `iterated_model` of `IteratedModel` on some `data` (by binding to a
machine and calling `fit!`, for example) performs the following actions:

- The the `data` is split into *train* and *test* sets, according to the specified
  `resampling` strategy. If `resampling isa Holdout` (the default) then train and test are
  disjoint, if `resampling == InSample()`, then train and test both point to teh entire
  dataset provided. See under "resampling" above.

- A clone of the wrapped model, `model` is bound to the train data in an internal machine,
  `train_mach`. This machine is the object to which controls are applied. For example,
  `Callback(fitted_params |> print)` will print the value of `fitted_params(train_mach)`.

- The iteration parameter of the clone is set to `0`.

- The specified `controls` are repeatedly applied to `train_mach` in sequence, until one
  of the controls triggers a stop. Loss-based controls (eg, `Patience()`, `GL()`,
  `Threshold(0.001)`) use a loss based on the test data, applying `measure` to predictions
  on the test inputs paired with the test target values. (Specifically, these predictions
  are those returned by `operation(train_mach)`.)

- Once a stop has been triggered, a clone of `model` is bound to all `data` in a machine
  called `production_machine` below, unless `retrain == false` (true by default) or
  `resampling === nothing`, in which case `production_machine` coincides with `train_mach`.


# Prediction

Calling `predict(mach, Xnew)` in the example above returns `predict(production_machine,
Xnew)`. Similar similar statements hold for `predict_mean`, `predict_mode`,
`predict_median`.


# Controls that mutate parameters

A control is permitted to mutate the fields (hyper-parameters) of
`train_mach.model` (the clone of `model`). For example, to mutate a
learning rate one might use the control

    Callback(mach -> mach.model.eta = 1.05*mach.model.eta)

However, unless `model` supports warm restarts with respect to changes
in that parameter, this will trigger retraining of `train_mach` from
scratch, with a different training outcome, which is not recommended.


# Warm restarts

In the following example, the second `fit!` call will not restart training of the internal
`train_mach`, assuming `model` supports warm restarts:

```julia
iterated_model = IteratedModel(
    model,
    controls = [Step(1), NumberLimit(100)],
)
mach = machine(iterated_model, X, y)
fit!(mach) # train for 100 iterations
iterated_model.controls = [Step(1), NumberLimit(50)],
fit!(mach) # train for an *extra* 50 iterations
```

More generally, if `iterated_model` is mutated and `fit!(mach)` is called again, then a
warm restart is attempted if the only parameters to change are `model` or `controls` or
both.

Specifically, `train_mach.model` is mutated to match the current value of
`iterated_model.model` and the iteration parameter of the latter is updated to the last
value used in the preceding `fit!(mach)` call. Then repeated application of the (updated)
controls begin anew.

"""
function IteratedModel(
    args...;
    model=nothing,
    control=DEFAULT_CONTROLS,
    controls=control,
    resampling=MLJBase.Holdout(),
    measures=nothing,
    measure=measures,
    weights=nothing,
    class_weights=nothing,
    operation=nothing,
    retrain=false,
    check_measure=true,
    iteration_parameter=nothing,
    cache=true,
    logger=MLJBase.default_logger(),
    compact_traces=false,
    warm_restart_extends_traces=false,
    )

    length(args) < 2 || throw(ArgumentError("At most one non-keyword argument allowed. "))
    if length(args) === 1
        atom = first(args)
        model === nothing ||
            @warn "Using `model=$atom`. Ignoring specification `model=$model`. "
    else
        model === nothing && throw(ERR_MODEL_UNSPECIFIED)
        atom = model
    end

    options =  (
        atom,
        controls,
        resampling,
        measure,
        weights,
        class_weights,
        operation,
        retrain,
        check_measure,
        iteration_parameter,
        cache,
        logger,
        compact_traces,
        warm_restart_extends_traces,
    )

    if atom isa Deterministic
        iterated_model = DeterministicIteratedModel(options...)
    elseif atom isa Probabilistic
        iterated_model = ProbabilisticIteratedModel(options...)
    else
        throw(ERR_NOT_SUPERVISED)
    end

    message = clean!(iterated_model)
    isempty(message) || @warn message

    return iterated_model

end

function MLJBase.clean!(iterated_model::EitherIteratedModel)
    message = ""
    measure = iterated_model.measure
    if measure === nothing &&
        iterated_model.resampling !== nothing
        measure = MLJBase.default_measure(iterated_model.model)
        measure === nothing && throw(ERR_NEED_MEASURE)
    end
    iter = deepcopy(iterated_model.iteration_parameter)
    if iter === nothing
        iter = iteration_parameter(iterated_model.model)
        iter === nothing && throw(ERR_NEED_PARAMETER)
    end
    try
        MLJBase.recursive_getproperty(iterated_model.model,
                                      iter)
    catch
        throw(err_bad_iteration_parameter(iter))
    end

    resampling = iterated_model.resampling

    if !(resampling isa IterationResamplingTypes)
        message *= WARN_POOR_RESAMPLING_CHOICE
    end

    if resampling isa MLJBase.TrainTestPairs && length(resampling) !== 1
        message *= WARN_POOR_CHOICE_OF_PAIRS
    end

    training_control_candidates = filter(iterated_model.controls) do c
        c isa TrainingControl || !(c isa Control)
    end

    if isempty(training_control_candidates)
        throw(ERR_MISSING_TRAINING_CONTROL)
    end

    return message
end
