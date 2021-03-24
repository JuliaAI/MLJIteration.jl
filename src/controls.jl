# # ITERATIONS

struct WithIterationsDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithIterationsDo(f::Function;
     stop_if_true=false,
     stop_message=nothing) = WithIterationsDo(f, stop_if_true, stop_message)
WithIterationsDo(; f=n->@info(n), kwargs...) = WithIterationsDo(f, kwargs...)

IterationControl.@create_docs(
    WithIterationsDo,
    header="WithIterationsDo(f=n->@info(n), stop_if_true=false, "*
    "stop_message=nothing)",
    example="WithIterationsDo(i->put!(my_channel, i))",
    body="Call `f(i)`, where "*
    "`i` is the current number of model iterations "*
    "(generally more than "*
    "the number of control cycles). "*
    "If `stop_if_true` is `true`, then trigger an early stop "*
    "if the value returned by `f` is `true`, logging the "*
    "`stop_message` if specified. ")

function IterationControl.update!(c::WithIterationsDo,
                                  ic_model,
                                  verbosity, state...)
    i = ic_model.n_iterations
    r = c.f(i)
    done = (c.stop_if_true && r isa Bool && r) ? true : false
    return (done = done, i = i)
end

IterationControl.done(c::WithIterationsDo, state) = state.done

function IterationControl.takedown(c::WithIterationsDo, verbosity, state)
    if state.done
        message = c.stop_message === nothing ?
            "Stop triggered by a `WithIterationsDo` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return (done = true, log = message)
    else
        return (done = false, log = "")
    end
end


# # WithEvaluationDo

struct WithEvaluationDo{F<:Function}
    f::F
    stop_if_true::Bool
    stop_message::Union{String,Nothing}
end

# constructor:
WithEvaluationDo(f::Function;
     stop_if_true=false,
     stop_message=nothing) = WithEvaluationDo(f, stop_if_true, stop_message)
WithEvaluationDo(; f=e->@info(e.measuresment),
                 kwargs...) = WithEvaluationDo(f, kwargs...)

IterationControl.@create_docs(
    WithEvaluationDo,
    header="WithEvaluationDo"*
    "(f=e->@info(e.measurement), "*
    "stop_if_true=false, "*
    "stop_message=nothing)",
    example="WithEvaluationDo(e->put!(my_evaluations, e))",
    body="Call `f(e)`, where "*
    "`e` is the latest performance evaluation, as returned by "*
    "`evaluate!(train_mach, resampling=..., ...)`. Not valid if "*
    "`resampling=nothing`.\n\n"*
    "If `stop_if_true` is `true`, then trigger an early stop "*
    "if the value returned by `f` is `true`, logging the "*
    "`stop_message` if specified. ")

function IterationControl.update!(c::WithEvaluationDo,
                                  ic_model,
                                  verbosity, state...)
    e = ic_model.evaluation
    r = c.f(e)
    done = (c.stop_if_true && r isa Bool && r) ? true : false
    return (done = done, )
end

IterationControl.done(c::WithEvaluationDo, state) = state.done

function IterationControl.takedown(c::WithEvaluationDo, verbosity, state)
    if state.done
        message = c.stop_message === nothing ?
            "Stop triggered by a `WithEvaluationDo` control. " :
            c.stop_message
        verbosity > 0 && @info message
        return (done = true, log = message)
    else
        return (done = false, log = "")
    end
end


# # Save

struct Save{K}
    filename::String
    kwargs::K
end

# constructor:
Save(filename="machine.jlso"; kwargs...) =
    Save(filename, kwargs)

IterationControl.@create_docs(Save,
             header="Save(filename=\"machine.jlso\"; kwargs...)",
             example="Save(\"run3/machine.jlso\", compression=:gzip)",
             body="Save the current state of the machine being iterated to "*
             "disk, using the provided `filename`, decorated with a number, "*
             "as in \"run3/machine_42.jlso\". The specified `kwargs` "*
             "are passed to the model-specific serializer "*
             "(JLSO for most Julia models).\n\n"*
             "For more on what is meant by \"the machine being iterated\", "*
             "see [`IteratedModel`](@ref).")

function IterationControl.update!(c::Save,
                                  ic_model,
                                  verbosity,
                                  state=(filenumber=0, ))
    filenumber = state.filenumber + 1
    root, suffix = splitext(c.filename)
    filename = string(root, filenumber, suffix)
    train_mach = IterationControl.expose(ic_model)
    verbosity > 0 && @info "Saving \"$filename\". "
    MLJBase.save(filename, train_mach, c.kwargs...)
    return (filenumber=filenumber, )
end


# # CYCLE LEARNING RATE

struct CycleLearningRate{F<:AbstractFloat}
    stepsize::Int
    lower::F
    upper::F
    learning_rate_parameter::Union{Symbol,Expr}
end
CycleLearningRate(; stepsize=4,
                  lower::T=0.001,
                  upper::T=0.05,
                  learning_rate_parameter=:(optimiser.η)) where T<:Real =
                      CycleLearningRate(stepsize,
                                        float(lower),
                                        float(upper),
                                        learning_rate_parameter)

IterationControl.@create_docs(
    CycleLearningRate,
    header="CycleLearningRate(stepsize=4, lower=0.001, "*
    "upper=0.05, learning_rate_parameter=:(optimiser.η))",
    example="CycleLearningRate(learning_rate_parameter=:learning_rate)",
    body="Mutate the specified `learning_rate_parameter` "*
    "(defaulting to the field appropriate for MLJFlux models) "*
    "to the next value "*
    "in a triangular cyclic learning rate policy, using the specified "*
    "`upper` and `lower` learning rate bounds. "*
    "See [L. N. Smith (2019)](https://ieeexplore.ieee.org/document/7926641): "*
    "\"Cyclical Learning Rates for Training "*
    "Neural Networks,\" 2017 IEEE Winter Conference on Applications "*
    "of Computer Vision (WACV), Santa Rosa, CA, USA, pp. 464-472.\n\n"*
    "Here `stepsize` is half the period of the mutation cycle, measured "*
    "in iterations of the underlying model. "*
    "For example, `stepsize=4` means the learning rate undergoes one "*
    "cycle of mutation every 8 epochs in an MLJFlux model.\n\n"*
    "**Note.** Since \"one iteration\" is the same as \"one epoch\" "*
    "in MLJFlux models, this means learning rate updates can be applied "*
    "once per epoch, at most, "*
    "rather than \"continuously\" throughout an epoch as in "*
    "the cited paper. ")

# return one cycle of learning rate values:
function one_cycle(c::CycleLearningRate)
    rise = range(c.lower, c.upper, length=c.stepsize + 1)
    fall = reverse(rise)
    return vcat(rise[1:end - 1], fall[1:end - 1])
end

function IterationControl.update!(control::CycleLearningRate,
                                  wrapper,
                                  verbosity,
                                  state = (n = 0,))
    n = state.n
    rates = n == 0 ? one_cycle(control) : state.learning_rates
    index = mod(n, length(rates)) + 1
    r = rates[index]
    verbosity > 1 && @info "learning rate: $r"
    MLJBase.recursive_setproperty!(wrapper.model,
                                   control.learning_rate_parameter,
                                   r)
    return (n = n + 1, learning_rates = rates)
end
