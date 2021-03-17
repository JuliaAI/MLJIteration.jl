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
    root, suffix = splitext(cfilename)
    filename = string(root, filenumber, suffix)
    train_mach = IterationControl.expose(ic_model)
    verbosity > 0 && @info "Saving \"$filename\". "
    MLJBase.save(filename, train_mach, c.kwargs...)
    return (filenumber=filenumber, )
end

