# We will be wrapping MLJ machines/resampling machines
# (cf. MLJBase/src/resampling.jl). It is for the wrapped type that we
# will be overloading the methods of `IterationControl`.

const ERR_EVALUATION =
    ArgumentError("There are no evaluation objects if `resampling=nothing`. ")

mlj_model(mach::Machine) = mach.model
mlj_model(mach::Machine{<:Resampler}) = mach.model.model

# training machine wrappper (constructor resets mlj model iterations
# to zero):
struct ICModel{M}
    mach::M
    iteration_parameter::Union{Symbol,Expr}
    verbosity::Integer
    n::Base.RefValue{Int64} # num calls to train!
    i::Base.RefValue{Int64} # num iterations
    Δi::Base.RefValue{Int64}
    function ICModel(mach::M, iteration_parameter, verbosity) where M
        model = mlj_model(mach)
        rset!(model, iteration_parameter, 0)
        return new{M}(mach,
                      iteration_parameter,
                      verbosity,
                      Ref(0),
                      Ref(0),
                      Ref(0))
    end
end

mlj_model(ic_model::ICModel) = mlj_model(ic_model.mach)

# overloading `train!`:
function IterationControl.train!(m::ICModel, n::Int)

    # increment training call counter:
    m.n[] = m.n[] + 1

    model = mlj_model(m)

    # update iteration parameter value stored on ic_model:
    m.i[] = rget(model, m.iteration_parameter) + n

    # update the iteration parameter in the mlj model:
    rset!(model, m.iteration_parameter, m.i[])

    fit!(m.mach, verbosity=m.verbosity - 2)
    m.Δi[] = n
end

# overloading `expose`- for `resampling === nothing`:
IterationControl.expose(ic_model::ICModel) = ic_model.mach

# overloading `expose`- for `resampling isa Holdout` or
# other resampling strategy:
IterationControl.expose(ic_model::ICModel{<:Machine{<:Resampler}}) =
    MLJBase.fitted_params(ic_model.mach).machine

# overloading `loss` - for `resampling === nothing`:
function IterationControl.loss(m::ICModel)
    losses = training_losses(IterationControl.expose(m))
    losses isa Nothing && return nothing
    return last(losses)
end

# overloading `loss` - for `resampling isa Holdout`:
IterationControl.loss(m::ICModel{<:Machine{<:Resampler}}) =
    evaluate(m.mach).measurement |> first

# overloading `training_loss`:
function IterationControl.training_losses(m::ICModel)
    mach = IterationControl.expose(m)
    losses = training_losses(mach)
    losses isa Nothing && return nothing
    s = length(losses)
    return view(losses, (s - m.Δi[] + 1):s)
end

_evaluation(ic_model) = throw(ERR_EVALUATION)
_evaluation(ic_model::ICModel{<:Machine{<:Resampler}}) =
    MLJBase.evaluate(ic_model.mach)

# interface for user-defined controls:
function Base.getproperty(m::ICModel, property::Symbol)
    property === :machine && return IterationControl.expose(m)
    property === :model && return mlj_model(m)
    property === :n_cycles && return m.n[]
    property === :n_iterations && return m.i[]
    property === :Δiterations && return m.Δi[]
    property === :loss && return IterationControl.loss(m)
    property === :training_losses && return IterationControl.training_losses(m)
    property === :evaluation && return _evaluation(m)
    return getfield(m, property)
end
Base.propertynames(::ICModel) = (:machine,
                                 :model,
                                 :n_cycles,
                                 :n_iterations,
                                 :Δiterations,
                                 :loss,
                                 :training_losses,
                                 :evaluation)
