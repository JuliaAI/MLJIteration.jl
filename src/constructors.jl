const IterationResamplingTypes =
    Union{Holdout,InSample,Nothing,MLJBase.TrainTestPairs}


## TYPES AND CONSTRUCTOR

mutable struct DeterministicIteratedModel{M<:Deterministic} <: MLJBase.Deterministic
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
end

mutable struct ProbabilisticIteratedModel{M<:Probabilistic} <: MLJBase.Probabilistic
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
end

const ERR_MISSING_TRAINING_CONTROL =
    ArgumentError("At least one control must be a training control "*
                  "(have type `$TrainingControl`) or be a "*
                  "custom control that calls IterationControl.train!. ")

const ERR_TOO_MANY_ARGUMENTS =
    ArgumentError("At most one non-keyword argument allowed. ")
const EitherIteratedModel{M} =
    Union{DeterministicIteratedModel{M},ProbabilisticIteratedModel{M}}
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
    IteratedModel(model;
        controls=...,
        resampling=Holdout(),
        measure=nothing,
        retrain=false,
        advanced_options...,
    )

Wrap the specified supervised `model` in the specified iteration `controls`. Here `model`
should support iteration, which is true if (`iteration_parameter(model)` is different from
`nothing`.

Available controls: $CONTROLS_LIST.

!!! important

    To make out-of-sample losses available to the controls, the wrapped `model` is only
    trained on part of the data, as iteration proceeds. The user may want to force
    retraining on all data after controlled iteration has finished by specifying
    `retrain=true`. See also "Training", and the `retrain` option, under "Extended help"
    below.

# Extended help

# Options

- `controls=$CONTROLS_DEFAULT`: Controls are summarized at
  [https://JuliaAI.github.io/MLJ.jl/dev/getting_started/](https://JuliaAI.github.io/MLJ.jl/dev/controlling_iterative_models/)
  but query individual doc-strings for details and advanced options. For creating your own
  controls, refer to the documentation just cited.

- `resampling=Holdout(fraction_train=0.7)`: The default resampling holds back 30% of data
  for computing an out-of-sample estimate of performance (the "loss") for controls such
  as `WithLossDo` and stopping criterion; specify `resampling=nothing` if all data is to
  be used for controlled iteration, with each out-of-sample loss replaced by the most
  recent training loss, assuming this is made available by the model
  (`supports_training_losses(model) == true`). If the model does not provide report a
  training loss, you can use `resampling=InSample()` instead, with an additional
  performance cost.  Otherwise, `resampling` must have type `Holdout` or be a vector with
  one element of the form `(train_indices, test_indices)`.

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


# Training

Training an instance `iterated_model` of `IteratedModel` on some `data` (by binding to a
machine and calling `fit!`, for example) performs the following actions:

- Assuming `resampling !== nothing`, the `data` is split into *train* and *test* sets,
  according to the specified `resampling` strategy.

- A clone of the wrapped model, `model` is bound to the train data in an internal machine,
  `train_mach`. If `resampling === nothing`, all data is used instead. This machine is the
  object to which controls are applied. For example, `Callback(fitted_params |> print)`
  will print the value of `fitted_params(train_mach)`.

- The iteration parameter of the clone is set to `0`.

- The specified `controls` are repeatedly applied to `train_mach` in sequence, until one
  of the controls triggers a stop. Loss-based controls (eg, `Patience()`, `GL()`,
  `Threshold(0.001)`) use an out-of-sample loss, obtained by applying `measure` to
  predictions and the test target values. (Specifically, these predictions are those
  returned by `operation(train_mach)`.)  If `resampling === nothing` then the most recent
  training loss is used instead. Some controls require *both* out-of-sample and training
  losses (eg, `PQ()`).

- Once a stop has been triggered, a clone of `model` is bound to all `data` in a machine
  called `mach_production` below, unless `retrain == false` (true by default) or
  `resampling === nothing`, in which case `mach_production` coincides with `train_mach`.


# Prediction

Calling `predict(mach, Xnew)` in the example above returns `predict(mach_production,
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

Specifically, `train_mach.model` is mutated to match the current value
of `iterated_model.model` and the iteration parameter of the latter is
updated to the last value used in the preceding `fit!(mach)` call. Then
repeated application of the (updated) controls begin anew.

"""
function IteratedModel(args...;
                       model=nothing,
                       control=CONTROLS_DEFAULT,
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
                       cache=true)

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
