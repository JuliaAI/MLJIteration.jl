const ERR_MISSING_TRAINING_CONTROL =
    ArgumentError("At least one control must be a training control "*
                  "(ie, be on this list: $TRAINING_CONTROLS) or be a "*
                  "custom control that calls IterationControl.train!. ")


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

const EitherIteratedModel{M} =
    Union{DeterministicIteratedModel{M},ProbabilisticIteratedModel{M}}

const ERR_NO_MODEL =
    ArgumentError("You need to specify model=... ")
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

"""
    IteratedModel(model=nothing,
                  controls=$CONTROLS_DEFAULT,
                  retrain=false,
                  resampling=Holdout(),
                  measure=nothing,
                  weights=nothing,
                  class_weights=nothing,
                  operation=predict,
                  verbosity=1,
                  check_measure=true,
                  iteration_parameter=nothing,
                  cache=true)

Wrap the specified `model <: Supervised` in the specified iteration
`controls`. Training a machine bound to the wrapper iterates a
corresonding machine bound to `model`. Here `model` should support
iteration.

To list all controls, do `MLJIteration.CONTROLS`. Controls are
summarized at
[https://alan-turing-institute.github.io/MLJ.jl/dev/getting_started/](https://alan-turing-institute.github.io/MLJ.jl/dev/controlling_iterative_models/)
but query individual doc-strings for details and advanced options. For
creating your own controls, refer to the documentation just cited.

To make out-of-sample losses available to the controls, the machine
bound to `model` is only trained on part of the data, as iteration
proceeds.  See details on training below. Specify `retrain=true`
to ensure the model is retrained on *all* available data, using the
same number of iterations, once controlled iteration has stopped.

Specify `resampling=nothing` if all data is to be used for controlled
iteration, with each out-of-sample loss replaced by the most recent
training loss, assuming this is made available by the model
(`supports_training_losses(model) == true`). Otherwise, `resampling`
must have type `Holdout` (eg, `Holdout(fraction_train=0.8, rng=123)`).

Assuming `retrain=true` or `resampling=nothing`,
`iterated_model` behaves exactly like the original `model` but with
the iteration parameter automatically selected. If
`retrain=false` (default) and `resampling` is not `nothing`, then
`iterated_model` behaves like the original model trained on a subset
of the provided data.

Controlled iteration can be continued with new `fit!` calls (warm
restart) by mutating a control, or by mutating the iteration parameter
of `model`, which is otherwise ignored.


### Training

Given an instance `iterated_model` of `IteratedModel`, calling
`fit!(mach)` on a machine `mach = machine(iterated_model, data...)`
performs the following actions:

- Assuming `resampling !== nothing`, the `data` is split into *train* and
  *test* sets, according to the specified `resampling` strategy, which
  must have type `Holdout`.

- A clone of the wrapped model, `iterated_model.model`, is bound to
  the train data in an internal machine, `train_mach`. If `resampling
  === nothing`, all data is used instead. This machine is the object
  to which controls are applied. For example, `Callback(fitted_params
  |> print)` will print the value of `fitted_params(train_mach)`.

- The iteration parameter of the clone is set to `0`.

- The specified `controls` are repeatedly applied to `train_mach` in
  sequence, until one of the controls triggers a stop. Loss-based
  controls (eg, `Patience()`, `GL()`, `Threshold(0.001)`) use an
  out-of-sample loss, obtained by applying `measure` to predictions
  and the test target values. (Specifically, these predictions are
  those returned by `operation(train_mach)`.)  If `resampling ===
  nothing` then the most recent training loss is used instead. Some
  controls require *both* out-of-sample and training losses (eg,
  `PQ()`).

- Once a stop has been triggered, a clone of `model` is bound to all
  `data` in a machine called `mach_production` below, unless
  `retrain == false` or `resampling === nothing`, in which case
  `mach_production` coincides with `train_mach`.


### Prediction

Calling `predict(mach, Xnew)` returns `predict(mach_production,
Xnew)`. Similar similar statements hold for `predict_mean`,
`predict_mode`, `predict_median`.


### Controls

A control is permitted to mutate the fields (hyper-parameters) of
`train_mach.model` (the clone of `model`). For example, to mutate a
learning rate one might use the control

    Callback(mach -> mach.model.eta = 1.05*mach.model.eta)

However, unless `model` supports warm restarts with respect to changes
in that parameter, this will trigger retraining of `train_mach` from
scratch, with a different training outcome, which is not recommended.


### Warm restarts

If `iterated_model` is mutated and `fit!(mach)` is called again, then
a warm restart is attempted if the only parameters to change are
`model` or `controls` or both.

Specifically, `train_mach.model` is mutated to match the current value
of `iterated_model.model` and the iteration parameter of the latter is
updated to the last value used in the preceding `fit!(mach)` call. Then
repeated application of the (updated) controls begin anew.

"""
function IteratedModel(; model=nothing,
                       control=CONTROLS_DEFAULT,
                       controls=control,
                       resampling=MLJBase.Holdout(),
                       measures=nothing,
                       measure=measures,
                       weights=nothing,
                       class_weights=nothing,
                       operation=predict,
                       retrain=false,
                       check_measure=true,
                       iteration_parameter=nothing,
                       cache=true)

    model == nothing && throw(ERR_NO_MODEL)

    if model isa Deterministic
        iterated_model = DeterministicIteratedModel(model,
                                                    controls,
                                                    resampling,
                                                    measure,
                                                    weights,
                                                    class_weights,
                                                    operation,
                                                    retrain,
                                                    check_measure,
                                                    iteration_parameter,
                                                    cache)
    elseif model isa Probabilistic
        iterated_model = ProbabilisticIteratedModel(model,
                                                    controls,
                                                    resampling,
                                                    measure,
                                                    weights,
                                                    class_weights,
                                                    operation,
                                                    retrain,
                                                    check_measure,
                                                    iteration_parameter,
                                                    cache)
    else
        throw(ERR_NOT_SUPERVISED)
    end

    message = clean!(iterated_model)
    isempty(message) || @info message

    return iterated_model

end

function MLJBase.clean!(iterated_model::EitherIteratedModel)
    message = ""
    if iterated_model.measure === nothing &&
        iterated_model.resampling !== nothing
        iterated_model.measure = MLJBase.default_measure(iterated_model.model)
        if iterated_model.measure === nothing
            throw(ERR_NEED_MEASURE)
        else
            message *= "No measure specified. "*
            "Setting measure=$(iterated_model.measure). "
        end
    end
    iterated_model.iteration_parameter === nothing &&
        iteration_parameter(iterated_model.model) === nothing &&
        throw(ERR_NEED_PARAMETER)

    if iterated_model.resampling isa Holdout &&
        iterated_model.resampling.shuffle
        message *= "The use of sample-shuffling in `Holdout` "*
            "will significantly slow training as "*
            "each increment of the iteration parameter "*
            "will force iteration from scratch (cold restart). "
    end

    training_control_candidates = filter(iterated_model.controls) do c
        c isa TrainingControl || !(c isa Control)
    end
    if isempty(training_control_candidates)
        throw(ERR_MISSING_TRAINING_CONTROL)
    end

    return message
end
