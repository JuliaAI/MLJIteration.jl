# We will be wrapping MLJ resampling machines
# (cf. MLJBase/src/resampling.jl). It is for the wrapped type that we
# will be overloading the methods of `IterationControl`.

const ERR_TRAINING_LOSSES =
    ArgumentError("Attempt to inspect training losses for "*
                  "a model that doesn't report them. ")

# wrappper:
struct ICModel{M}
    machine::M
    iter::Union{Symbol,Expr}
    verbosity::Integer
    last_Δn::Base.RefValue{Int64}
    ICModel(m::M, i, v) where M = new{M}(m, i, v, Ref(0))
end

mlj_model(ic_model) = ic_model.machine.model
mlj_model(ic_model::ICModel{<:Machine{<:Resampler}}) =
    ic_model.machine.model.model

# overloading `train!`:
function IterationControl.train!(m::ICModel, n::Int)
    model = mlj_model(m)

    # increase the iteration parameter by `n`:
    rset!(model, m.iter, rget(model, m.iter) + n)

    fit!(m.machine, verbosity=m.verbosity - 1)
    m.last_Δn[] = n
end

# overloading `expose`- for `resampling === nothing`:
IterationControl.expose(ic_model::ICModel) = ic_model.machine

# overloading `expose`- for `resampling isa Holdout`:
IterationControl.expose(ic_model::ICModel{<:Machine{<:Resampler}}) =
    MLJBase.fitted_params(ic_model.machine).machine

# overloading `loss` - for `resampling === nothing`:
function IterationControl.loss(m::ICModel)
    losses = training_losses(IterationControl.expose(m))
    losses === nothing && throw(ERR_TRAINING_LOSSES)
    return last(losses)
end

# overloading `loss` - for `resampling isa Holdout`:
IterationControl.loss(m::ICModel{<:Machine{<:Resampler}}) =
    evaluate(m.machine).measurement |> first

# overloading `training_loss`:
function IterationControl.training_losses(m::ICModel)
    mach = IterationControl.expose(m)
    losses = training_losses(mach)
    losses === nothing && throw(ERR_TRAINING_LOSSES)
    s = length(losses)
    return view(losses, (s - m.last_Δn[] + 1):s)
end
