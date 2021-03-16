## TYPES AND CONSTRUCTOR

mutable struct DeterministicIteratedModel{M<:Deterministic} <: MLJBase.Deterministic
    model::M
    controls
    resampling # resampling strategy
    measure
    weights::Union{Nothing,Vector{<:Real}}
    class_weights::Union{Nothing,Dict{Any,<:Real}}
    operation
    final_train::Bool
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
    final_train::Bool
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
                  controls=[Step(10), Patience(5), NumberLimit(50), NotANumber()],
                  final_train=false,
                  resampling=Holdout(),
                  measure=nothing,
                  weights=nothing,
                  class_weights=nothing,
                  class_weights=nothing,
                  operation=predict,
                  verbosity=1,
                  check_measure=true,
                  iteration_parameter=nothing,
                  cache=true)

Wrap the specified `model <: Supervised` in the specified iteration
`controls`. Training a machine bound to the wrapper iterates a
corresonding machine bound to `model`. Here `model` should support
iteration. Controls are discussed further below.

To make out-of-sample losses available to the controls, the machine
bound to `model` is only trained on part of the data, as iteration
proceeds.  See details on training below. Specify `final_train=true`
to ensure the model is retrained on *all* available data, using the
same number of iterations, once controlled iteration has stopped.

Specify `resampling=nothing` if all data is to be used for controlled
iteration, with each out-of-sample loss replaced by the most recent
training loss, assuming this is made available by the model
(`supports_training_losses(model) == true`). Otherwise, `resampling`
must have type `Holdout`.

Assuming `final_train=true` or `resampling=nothing`,
`iterated_model` behaves exactly like the original `model` but with
the iteration parameter automatically selected. If
`final_train=false` (default) and `resampling` is not `nothing`, then
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
  `final_train == false` or `resampling === nothing`, in which case
  `mach_production` coincides with `train_mach`.


### Prediction

Calling `predict(mach, Xnew)` returns `predict(mach_production,
Xnew)`. Similar similar statements hold for `predict_mean`,
`predict_mode`, `predict_median`.


### Controls

Every set of controls should include a `Step(...)`, usually the first control.

For a summary of all available controls, see
[https://github.com/ablaom/IterationControl.jl#controls-provided](https://github.com/ablaom/IterationControl.jl#controls-provided). Basic
controls for getting started are summarized below. Query the
doc-strings for details and advanced options.

control                          | description
---------------------------------|----------------------------------
`Step(n=1)`                      | Train model for `n` iterations
`Info(f=identity)`               | Log to `Info` the value of `f(train_mach)`
`Callback(f=_->nothing)`         | Call `f(train_mach)`
`TimeLimit(t=0.5)`               | Stop after `t` hours
`NumberLimit(n=100)`             | Stop after `n` control cycles
`WithLossDo(f=x->@info(x))`      | Call `f(loss)` where `loss` is current loss
`NotANumber()`                   | Stop when `NaN` encountered
`Threshold(value=0.0)`           | Stop when `loss < value`
`Patience(n=5)`                  | Stop after `n` consecutive loss increases
`Save(filename="machine.jlso")`  | Save `train_mach` machine to file
`MLJIteration.skip(control, p=1)`| Apply `control` but only every `p` cycles.

A control is permitted to mutate the fields (hyper-parameters) of
`train_mach.model` (the clone of `model`). For example, to mutate a
learning rate one might use the control

    Callback(m -> m.model.eta = 1.05*m.model.eta)

However, unless `model` supports warm restarts with respect to changes
in that parameter, this will trigger retraining of `train_mach` from
scratch, with a different training outcome, which is not recommended.

See the MLJ documentation for user-defined controls.


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
                       final_train=false,
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
                                                    final_train,
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
                                                    final_train,
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
    if iterated_model.measure === nothing
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

    return message
end
