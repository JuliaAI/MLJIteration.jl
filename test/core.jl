module TestCore

using Test
using MLJIteration
using IterationControl
using MLJBase
using MLJModelInterface
using StatisticalMeasures
using ..DummyModel

X, y = make_dummy(N=20)
model = DummyIterativeModel(n=0)

@testset "integration: resampling=nothing" begin

    model = DummyIterativeModel(n=0)
    controls=[Step(2), Threshold(0.01), TimeLimit(0.005)]

    # using IterationControl.jl directly:
    mach = machine(deepcopy(model), X, y);
    function IterationControl.train!(mach::Machine{<:DummyIterativeModel},
                                     n::Int)
        mlj_model = mach.model
        mlj_model.n = mlj_model.n + n
        fit!(mach, verbosity=0)
    end
    IterationControl.loss(mach::Machine{<:DummyIterativeModel}) =
        last(training_losses(mach))
    IterationControl.train!(mach, controls..., verbosity=0)
    losses1 = report(mach).training_losses
    yhat1 = predict(mach, X)

    # using IteratedModel wrapper:
    imodel = IteratedModel(model=model,
                           controls=controls,
                           resampling=nothing)
    mach = machine(imodel, X, y)
    @test_logs((:info, r"Training"),
               (:info, MLJIteration.info_unspecified_iteration_parameter(:n)),
               (:info, r"final loss"),
               (:info, r"final training loss"),
               (:info, r"Stop"),
               (:info, r"Total"),
               fit!(mach, verbosity=1))
    imodel.resampling = nothing
    @test_logs fit!(mach, verbosity=0)

    losses2 = report(mach).model_report.training_losses
    yhat2 = predict(mach, X);
    @test imodel.model == DummyIterativeModel(n=0) # hygeine check

    # compare:
    @test losses1 ≈ losses2
    @test yhat1 ≈ yhat2

    # report:
    r = report(mach)
    @test r.model_report.training_losses ≈ losses1
    @test :controls in keys(r)
    i = r.n_iterations
    @test i == 38

    # warm restart when changing controls:
    noise(n) = fill((:info, r"43"), n)
    imodel.controls = [Step(1),
                       WithIterationsDo(i-> i>39,
                                        stop_if_true=true),
                       Info(x->"43")]
    @test_logs((:info, r"Updating"),
               noise(2)...,
               (:info, r""),
               (:info, r""),
               (:info, r""),
               (:info, r""),
               fit!(mach))
    @test report(mach).n_iterations == i + 2

    # warm restart when changing iteration parameter (trains one more
    # iteration because stopping control comes after `Step(...)`):
    imodel.model.n = 1
    @test_logs((:info, r"Updating"),
               noise(1)...,
               (:info, r""),
               (:info, r""),
               (:info, r""),
               (:info, r""),
               fit!(mach))
    @test report(mach).n_iterations == i + 3

    # cold restart when changing anything else:
    imodel.check_measure=false
    imodel.controls = [Step(1),
                       WithIterationsDo(i-> i>4,
                                        stop_if_true=true),
                       Info(x->"43")]
    @test_logs((:info, r"Updating"),
               (:info, MLJIteration.info_unspecified_iteration_parameter(:n)),
               noise(5)...,
               (:info, r""),
               (:info, r""),
               (:info, r""),
               (:info, r""),
               fit!(mach))
    @test report(mach).n_iterations == 5
end

@testset "integration: resampling=Holdout()" begin

    controls=[Step(2), Patience(4), TimeLimit(0.001)]

    # using IterationControl.jl directly:
    mach = machine(deepcopy(model), X, y);
    train, test = partition(eachindex(y), 0.7)
    function IterationControl.train!(mach::Machine{<:DummyIterativeModel},
                                     n::Int)
        mlj_model = mach.model
        mlj_model.n = mlj_model.n + n
        fit!(mach, rows=train, verbosity=0)
    end
    function IterationControl.loss(mach::Machine{<:DummyIterativeModel})
        yhat = predict(mach, rows=test)
        return mae(yhat, y[test]) |> mean
    end
    IterationControl.train!(mach, controls...; verbosity=0)
    losses1 = report(mach).training_losses;
    yhat1 = predict(mach, X[test]);
    niters = mach.model.n
    @test niters == length(losses1)

    # using IteratedModel wrapper:
    imodel = IteratedModel(model=model,
                           resampling=Holdout(fraction_train=0.7),
                           controls=controls)
    mach = machine(imodel, X, y)
    @test_logs((:info, r"Training"),
               (:info, MLJIteration.info_unspecified_iteration_parameter(:n)),
               (:info, MLJIteration.info_unspecified_measure(l2)),
               (:info, r"final loss"),
               (:info, r"final train"),
               (:info, r"Stop"),
               (:info, r"Total"),
               fit!(mach, verbosity=1))
    imodel.measure = mae
    @test_logs fit!(mach, verbosity=0)
    losses2 = report(mach).model_report.training_losses;
    yhat2 = predict(mach, X[test]);

    # compare:
    @test losses1 ≈ losses2
    @test yhat1 ≈ yhat2

    # now repeat wrapper train with retrain=true:
    imodel = IteratedModel(model=model,
                           resampling=Holdout(fraction_train=0.7),
                           controls=controls,
                           retrain=true,
                           measure=mae)
    mach = machine(imodel, X, y)
    fit!(mach, verbosity=0)
    losses3 = report(mach).model_report.training_losses
    yhat3 = predict(mach, X[test])
    @test fitted_params(mach).machine.model.n == niters

    # to be compared with atomic model trained on all data for same
    # number of iterations:
    model2 = DummyIterativeModel(n=niters)
    mach2 = machine(model2, X, y)
    fit!(mach2, verbosity=0)
    losses4 = report(mach2).training_losses
    yhat4 = predict(mach2, X[test])

    # check report:
    r = report(mach2)

    # compare:
    @test losses3 ≈ losses4
    @test yhat3 ≈ yhat4
end

@testset "integration: cyclic learning rates" begin

    X, y = make_dummy(N=100)
    ss = 3
    model = DummyIterativeModel()

    # constant learning rate:
    controls = [CycleLearningRate(stepsize=ss,
                                  lower=1,
                                  upper=1,
                                  learning_rate_parameter=:learning_rate),
                Step(1),
                NumberLimit(5*ss)]

    imodel = IteratedModel(model=model,
                           resampling=Holdout(fraction_train=0.7),
                           controls=controls,
                           measure=mae)

    mach = machine(imodel, X, y)
    fit!(mach, verbosity=0)
    yhat1 = predict(mach, X)

    # no cycling:
    imodel.controls = [Step(1), NumberLimit(5*ss)]
    fit!(mach, force=true, verbosity=0)
    yhat2 = predict(mach, X)

    # with real cycling:
    controls = [CycleLearningRate(stepsize=ss,
                                  lower=0.5,
                                  upper=1.5,
                                  learning_rate_parameter=:learning_rate),
                Step(1),
                NumberLimit(5*ss)]
    imodel.controls = controls
    fit!(mach, force=true, verbosity=0)
    yhat3 = predict(mach, X)

    # compare:
    @test yhat1 ≈ yhat2
    @test !(yhat3 ≈ yhat2)

end

@testset "issue #40" begin
    # change trait to an incorrect value:
    MLJModelInterface.iteration_parameter(::Type{<:DummyIterativeModel}) = :junk
    @test iteration_parameter(model) == :junk

    # user specifies correct value:
    iterated_model = IteratedModel(model, measure=mae, iteration_parameter=:n)
    mach = machine(iterated_model, X, y)

    # and model still runs:
    fit!(mach, verbosity=0)

    # change trait back:
    MLJModelInterface.iteration_parameter(::Type{<:DummyIterativeModel}) = :n
    @test iteration_parameter(model) == :n
end

# define a supervised model with ephemeral `fitresult`, but which overcomes this by
# overloading `save`/`restore`:
thing = []
mutable struct EphemeralRegressor <: Deterministic
    n::Int # dummy iteration parameter
end
EphemeralRegressor(; n=1) = EphemeralRegressor(n)
function MLJBase.fit(::EphemeralRegressor, verbosity, X, y)
    # if I serialize/deserialized `thing` then `id` below changes:
    id = objectid(thing)
    fitresult = (thing, id, mean(y))
    return fitresult, nothing, NamedTuple()
end
function MLJBase.predict(::EphemeralRegressor, fitresult, X)
    thing, id, μ = fitresult
    return id == objectid(thing) ? fill(μ, nrows(X)) :
        throw(ErrorException("dead fitresult"))
end
MLJBase.iteration_parameter(::EphemeralRegressor) = :n
function MLJBase.save(::EphemeralRegressor, fitresult)
    thing, _, μ = fitresult
    return (thing, μ)
end
function MLJBase.restore(::EphemeralRegressor, serialized_fitresult)
    thing, μ = serialized_fitresult
    id = objectid(thing)
    return (thing, id, μ)
end

@testset "save and restore" begin
    #https://github.com/alan-turing-institute/MLJ.jl/issues/1099
    X, y = (; x = rand(10)), fill(42.0, 3)
    controls = [Step(1), NumberLimit(2)]
    imodel = IteratedModel(
        EphemeralRegressor(42);
        measure=l2,
        resampling=Holdout(),
        controls,
    )
    mach = machine(imodel, X, y)
    fit!(mach, verbosity=0)
    io = IOBuffer()
    MLJBase.save(io, mach)
    seekstart(io)
    mach2 = machine(io)
    close(io)
    @test MLJBase.predict(mach2, (; x = rand(2))) ≈ fill(42.0, 2)
end

end
true
