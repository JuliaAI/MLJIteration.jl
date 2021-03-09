module DummyModel

export DummyIterativeModel, make_dummy

using Random
using Statistics
import StableRNGs.LehmerRNG
using CategoricalArrays
import Base.==

using MLJModelInterface
const MMI = MLJModelInterface


# AN OBJECT THAT CONSUMES NUMBERS AND GUESSES WHAT'S NEXT BY AVERAGING

mutable struct Guesser
    avg::Float64 # average so far
    n::Integer # number of observations so far
end
Guesser() = Guesser(0.0, 0)

==(g1::Guesser, g2::Guesser) = g1.avg == g2.avg && g1.n == g2.n

# consume a new number `r`:
function train!(guesser::Guesser, r)
    guesser.avg = (guesser.n*guesser.avg + r)/(guesser.n + 1)
    guesser.n = guesser.n + 1
end


# THE DUMMY MLJ MODEL

# Suppose `X` is a vector with `Finite` elscitype and `y` is a vector of
# `Continuous` elscitype. `DummyModel` predicts `y` given `X` by
# randomly sampling from the training data, without replacement, `n`
# times, and averaging the results for each class of `X`.


mutable struct DummyIterativeModel <: Deterministic
    n::Int
    rng::LehmerRNG
end
DummyIterativeModel(; n=10, rng=LehmerRNG(123)) = DummyIterativeModel(n, rng)

# core fitting function:
function train!(guesser_given_class,
                n,
                verbosity,
                training_losses,
                rng,
                global_avg_given_class,
                X,
                y)
    loss = isempty(training_losses) ? Inf : training_losses[end]
    for _ in 1:n
        i = rand(rng, eachindex(X))
        class = X[i]
        r = y[i]
        guesser = guesser_given_class[class]
        train!(guesser, r)
        global_avg = global_avg_given_class[class]
        loss = min(abs(guesser.avg - global_avg)/global_avg,
                   loss)
        verbosity < 2 || @info "training loss: $loss"
        push!(training_losses, loss)
    end
end

function MMI.fit(model::DummyIterativeModel, verbosity, X, y)

    training_losses = Float64[]

    # cheat to synthesize training losses:
    global_avg_given_class = Dict(c => mean(y[X .== c]) for c in levels(X))

    # intiate guessers, one per class in `X`:
    guesser_given_class = Dict(class => Guesser() for class in levels(X))

    loss = Inf

    rng = copy(model.rng)

    train!(guesser_given_class,
                model.n,
                verbosity,
                training_losses,
                rng,
                global_avg_given_class,
                X,
                y)

    fitresult = guesser_given_class
    report = (training_losses=training_losses, )
    cache = (rng, deepcopy(model), training_losses, global_avg_given_class)

    return fitresult, cache, report

end

function MMI.update(model::DummyIterativeModel,
                    verbosity,
                    fitresult,
                    cache,
                    X,
                    y)

    rng, old_model, training_losses, global_avg_given_class = cache
    guesser_given_class = fitresult

    Δn = model.n - old_model.n

    # warm restart only for increase in iteration parameter:
    Δn >= 0 || return fit(model, verbosity, X, y)

    train!(guesser_given_class,
           Δn,
           verbosity,
           training_losses,
           rng,
           global_avg_given_class,
           X,
           y)

    fitresult = guesser_given_class
    report = (training_losses=training_losses, )
    cache = (rng, deepcopy(model), training_losses, global_avg_given_class)

    return fitresult, cache, report

end

MMI.predict(::DummyIterativeModel, fitresult, Xnew) =
    [fitresult[c].avg for c in Xnew]

MMI.iteration_parameter(::Type{<:DummyIterativeModel}) = :n
MMI.training_losses(::DummyIterativeModel, report) = report.training_losses

MMI.supports_training_losses(::Type{<:DummyIterativeModel}) = true
MMI.input_scitype(::Type{<:DummyIterativeModel}) =
    AbstractVector{<:MMI.ScientificTypes.Finite}
MMI.target_scitype(::Type{<:DummyIterativeModel}) =
    AbstractVector{<:MMI.ScientificTypes.Continuous}


# # FOR SYTHESIZING DATA FOR USE WITH DUMMY MODEL

function make_dummy(; N=10, rng=LehmerRNG(123))
    X = categorical(vcat(fill('a', N), fill('b', N), fill('c', N)))
    y = vcat(rand(rng, N), 10*rand(rng, N), 100*rand(rng, N))
    shuffled = randperm(rng, 3N)
    return X[shuffled], y[shuffled]
end

end


# # TEMPORARY TEST AREA

using .DummyModel

using MLJBase

X, y = make_dummy(N=1000)

# train in stages:
model = DummyIterativeModel(n=4)
mach = machine(model, X, y) |> fit!
model.n = 9
fit!(mach)
fp_stages = fitted_params(mach)

# train in one hit:
model = DummyIterativeModel(n=9)
mach = machine(model, X, y) |> fit!
fp_onehit = fitted_params(mach)

@assert fp_onehit == fp_stages

@assert training_losses(mach) == report(mach).training_losses
@assert length(training_losses(mach)) == 9
