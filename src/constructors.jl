const IterationResamplingTypes =
    Union{Holdout,Nothing,MLJBase.TrainTestPairs}


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
function IteratedModel(args...;
                       model=nothing,
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

    length(args) < 2 || throw(ArgumentError("At most one non-keyword argument allowed. "))
    if length(args) === 1
        atom = first(args)
        model === nothing ||
            @warn "Using `model=$atom`. Ignoring specification `model=$model`. "
    else
        model === nothing && throw(ERR_MODEL_UNSPECIFIED)
        atom = model
    end

    if atom isa Deterministic
        iterated_model = DeterministicIteratedModel(atom,
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
    elseif atom isa Probabilistic
        iterated_model = ProbabilisticIteratedModel(atom,
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
