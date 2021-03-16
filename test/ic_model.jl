module TestICModel

using Test
using MLJIteration
using MLJBase
using IterationControl
using ..DummyModel

X, y = make_dummy(N=20)

@testset "IterationControl overloadings" begin

    model=DummyIterativeModel(n=10)

    # evaluate after 10 iterations by hand:
    mach = machine(model, X, y)
    e = evaluate!(mach,
                 resampling=Holdout(fraction_train=0.5),
                  measure=l2).measurement[1]

    losses = training_losses(mach)

    # evaluate after 10 iterations using IterationControl, and in two stages:
    resampling_machine =
        machine(Resampler(model=model,
                          resampling=Holdout(fraction_train=0.5),
                          measure=l2),
                X,
                y)

    ic_model = MLJIteration.ICModel(resampling_machine, :n, 0)
    IterationControl.train!(ic_model, 4)
    IterationControl.train!(ic_model, 6)

    # compare:
    @test IterationControl.loss(ic_model) ≈ e

    # test `Δi` is remembered:
    @test ic_model.Δi[] == 6

    # test training_losses:
    @test IterationControl.training_losses(ic_model) == losses[end - 5:end]

    # test expose:
    @test IterationControl.expose(ic_model) ==
        fitted_params(ic_model.mach).machine
    @test IterationControl.expose(MLJIteration.ICModel(mach, :n, 0)) == mach

    # test methods in case applying when `resampling == nothing`:
    model=DummyIterativeModel(n=0)
    mach = machine(model, X, y)
    ic_model2 = MLJIteration.ICModel(mach, :n, 0)
    IterationControl.train!(ic_model2, 10)
    losses2 = training_losses(mach)
    @test losses2 ≈ IterationControl.training_losses(ic_model2)
    @test losses2[end] == IterationControl.loss(ic_model2)
end

mutable struct FooBar <: MLJBase.Deterministic
    n::Int
end
iteration_parameter(::Type{<:FooBar}) = :n
MLJBase.fit(::FooBar, ::Any, data...) = nothing, nothing, nothing
MLJBase.predict(::FooBar, ::Any, Xnew) = ones(length(Xnew))

@testset "IterationControl.training_losses: error for unsupported models" begin
    resampling_machine =
        machine(Resampler(model=FooBar(10),
                          resampling=Holdout(fraction_train=0.5),
                          measure=l2),
                X,
                y)
    fit!(resampling_machine, verbosity=0)
    ic_model = MLJIteration.ICModel(resampling_machine, :n, 0)
    @test_throws(MLJIteration.ERR_TRAINING_LOSSES,
                 IterationControl.training_losses(ic_model))

    mach = machine(FooBar(10), X, y)
    fit!(mach, verbosity=0)
    ic_model = MLJIteration.ICModel(mach, :n, 0)
    @test_throws(MLJIteration.ERR_TRAINING_LOSSES,
                 IterationControl.training_losses(ic_model))
    @test_throws(MLJIteration.ERR_TRAINING_LOSSES,
                 IterationControl.loss(ic_model))
end

@testset "ICModel interface for users" begin
    model=DummyIterativeModel(n=0)
    resampling_machine =
        machine(Resampler(model=model,
                          resampling=Holdout(fraction_train=0.5),
                          measure=l2),
                X,
                y)

    ic_model = MLJIteration.ICModel(resampling_machine, :n, 0)
    IterationControl.train!(ic_model, 4)
    IterationControl.train!(ic_model, 6)
    @test ic_model.machine == fitted_params(resampling_machine).machine
    @test ic_model.machine.model == DummyIterativeModel(n=10)
    @test ic_model.n_cycles == 2
    @test ic_model.n_iterations == 10
    @test ic_model.Δiterations == 6
    @test ic_model.loss == IterationControl.loss(ic_model)
    @test ic_model.training_losses == IterationControl.training_losses(ic_model)
end

end

true
