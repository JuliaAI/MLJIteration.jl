module TestCore

using Test
using MLJIteration
using IterationControl
using MLJBase
using ..DummyModel

X, y = make_dummy(N=20)
model = DummyIterativeModel(n=0)

@testset "integration: resampling=nothing" begin

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
                           resampling=nothing,
                           controls=controls,
                           measure=rms)
    mach = machine(imodel, X, y)
    fit!(mach, verbosity=0)
    losses2 = report(mach).model_report.training_losses
    yhat2 = predict(mach, X)

    # compare:
    @test losses1 ≈ losses2
    @test yhat1 ≈ yhat2

end

@testset "integration: resampling=Holdout()" begin

    X, y = make_dummy(N=100)
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
        mlj_model = mach.model
        yhat = predict(mach, rows=test)
        return mae(yhat, y[test]) |> mean
    end
    IterationControl.train!(mach, controls...; verbosity=0)
    losses1 = report(mach).training_losses
    yhat1 = predict(mach, X[test])
    niters = mach.model.n
    @test niters == length(losses1)

    # using IteratedModel wrapper:
    imodel = IteratedModel(model=model,
                           resampling=Holdout(fraction_train=0.7),
                           controls=controls,
                           measure=mae)
    mach = machine(imodel, X, y)
    fit!(mach, verbosity=0)
    losses2 = report(mach).model_report.training_losses
    yhat2 = predict(mach, X[test])

    # compare:
    @test losses1 ≈ losses2
    @test yhat1 ≈ yhat2

    # now repeat wrapper train with final_train=true:
    imodel = IteratedModel(model=model,
                           resampling=Holdout(fraction_train=0.7),
                           controls=controls,
                           final_train=true,
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

    # compare:
    @test losses3 ≈ losses4
    @test yhat3 ≈ yhat4
end

end

true
