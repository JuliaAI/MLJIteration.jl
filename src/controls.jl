# # SIMPLE CONTROLS

const EXTRACTOR_GIVEN_STR = Dict(
    "iterations"    => :(ic_model.n_iterations),
    "evaluation"    => :(ic_model.evaluation),
    "fitted_params" => :(fitted_params(ic_model.machine)),
    "report"        => :(report(ic_model.machine)),
    "machine"       => :(ic_model.machine),
    "model"         => :(ic_model.machine.model))

# maps "fitted_params" to ":WithFittedParamsDo":
_control_name(str) = string("With",
                            join(uppercasefirst.(split(str, "_"))),
                            "Do") |> Symbol

const NAME_GIVEN_STR =
    Dict([str=>_control_name(str) for str in keys(EXTRACTOR_GIVEN_STR)]...)

const DOC_GIVEN_STR = Dict(
    "iterations" =>
    "`x` is the current number of model iterations "*
    "(generally more than the number of control cycles)",

    "evaluation" =>
    "`x` is the latest performance evaluation, as returned by "*
    "`evaluate!(train_mach, resampling=..., ...)`. Not valid if "*
    "`resampling=nothing`",

    "fitted_params" =>
    "`x = fitted_params(mach)` is the fitted parameters "*
    "of the training machine, `mach`, in its current state",

    "report"       =>
    "`x = report(mach)` is the report associated with the training "*
    "machine, `mach`,  in its current state",

    "machine"      =>
    "`x` is the training machine in its current state",

    "model" =>
    "`x` is the model associated with the "*
    "training machine; `f` may mutate `x`, as in "*
    "`f(x) = (x.learning_rate *= 0.9)`")

for str in keys(EXTRACTOR_GIVEN_STR) # eg, "fitted_params"
    sym = Symbol(str)                # eg, :fitted_params
    C = NAME_GIVEN_STR[str]          # eg, :WithFittedParamsDo
    C_str = string(C)
    doc = DOC_GIVEN_STR[str]         # eg, "`x` is the training machine..."
    extractor = EXTRACTOR_GIVEN_STR[str]

    quote
        struct $C{F<:Function}
            f::F
            stop_if_true::Bool
            stop_message::Union{String,Nothing}
        end

        # constructor:
        $C(f::Function;
           stop_if_true=false,
           stop_message=nothing) = $C(f, stop_if_true, stop_message)
        $C(; f=x->@info("$($str): $x"), kwargs...) =
            $C(f, kwargs...)

        IterationControl.@create_docs(
        $C,
        header="$($C_str)(f=x->@info(\"$($str): \$x\"), "*
            "stop_if_true=false, "*
            "stop_message=nothing)",
        example="$($C_str)(x->put!(my_channel, x))",
        body="Call `f(x)`, where $($doc). "*
            "If `stop_if_true` is `true`, then trigger an early stop "*
            "if the value returned by `f` is `true`, logging the "*
            "`stop_message` if specified. ")

        function IterationControl.update!(c::$C,
                                          ic_model,
                                          verbosity,
                                          n,
                                          state...)
            x = $extractor
            r = c.f(x)
            done = (c.stop_if_true && r isa Bool && r) ? true : false
            return (done = done, $sym = x)
        end

        IterationControl.done(c::$C, state) = state.done

        function IterationControl.takedown(c::$C, verbosity, state)
            if state.done
                message = c.stop_message === nothing ?
                    "Stop triggered by a `$($C_str)` control. " :
                    c.stop_message
                verbosity > 0 && @info message
                return (done = true, log = message)
            else
                return (done = false, log = "")
            end
        end
    end |> eval
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
                                  ncycles,
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


# # SAVE CONTROL
struct Save{F<:Function}
    filename::String
    method::F
end

Save(filename; method=serialize) =
    Save(filename, method)

Save(;filename="machine.jls", method=serialize) = 
    Save(filename, method)

IterationControl.@create_docs(Save,
             header="Save(filename=\"machine.jls\")",
             example="Save(\"run3/machine.jls\")",
             body="Save the current state of the machine being iterated to "*
             "disk, using the provided `filename`, decorated with a number, "*
             "as in \"run3/machine42.jls\". The default behaviour uses "*
             "the Serialization module but this can be changed by setting "*
             "the `method=save_fn(::String, ::Any)` argument where `save_fn` "*
             "is any serialization method. "*
             "For more on what is meant by \"the machine being iterated\", "*
             "see [`IteratedModel`](@ref).")

function IterationControl.update!(c::Save,
                                  ic_model,
                                  verbosity,
                                  n,
                                  state=(filenumber=0, ))
    filenumber = state.filenumber + 1
    root, suffix = splitext(c.filename)
    filename = string(root, filenumber, suffix)
    train_mach = IterationControl.expose(ic_model)
    verbosity > 0 && @info "Saving \"$filename\". "
    strain_mach = MLJBase.serializable(train_mach)
    c.method(filename, strain_mach)
    return (filenumber=filenumber, )
end
