module TestConstructors

using MLJIteration
using MLJBase
using ..DummyModel
using Test
using StatisticalMeasures

struct Foo <: MLJBase.Unsupervised end
struct Bar <: MLJBase.Deterministic end
struct FooBar <: MLJBase.Deterministic end

@testset "constructors" begin
    model = DummyIterativeModel()
    @test_throws MLJIteration.ERR_TOO_MANY_ARGUMENTS IteratedModel(1, 2)
    @test_throws MLJIteration.ERR_MODEL_UNSPECIFIED IteratedModel()
    @test_throws MLJIteration.ERR_NOT_SUPERVISED IteratedModel(model=Foo())
    @test_throws MLJIteration.ERR_NOT_SUPERVISED IteratedModel(model=Int)
    @test_throws MLJIteration.ERR_NEED_MEASURE IteratedModel(model=Bar())
    @test_throws MLJIteration.ERR_NEED_PARAMETER IteratedModel(model=Bar(),
                                                               measure=rms)
    iterated_model = @test_logs(IteratedModel(model=model))
    @test iterated_model.measure === nothing
    @test iterated_model.iteration_parameter === nothing
    iterated_model = @test_logs(
    IteratedModel(model=model, measure=mae, iteration_parameter=:n)
    )
    @test iterated_model.measure == mae
    @test_logs IteratedModel(model, measure=mae, iteration_parameter=:n)

    @test_logs IteratedModel(model=model, resampling=nothing, iteration_parameter=:n)

    @test_logs((:warn, MLJIteration.WARN_POOR_RESAMPLING_CHOICE),
               IteratedModel(model=model,
                             resampling=CV(),
                             measure=rms))
    @test_logs((:warn, MLJIteration.WARN_POOR_CHOICE_OF_PAIRS),
               IteratedModel(model=model,
                             resampling=[([1, 2], [3, 4]),
                                         ([3, 4], [1, 2])],
                             measure=rms))
    @test_logs IteratedModel(model=model,
                             resampling=[([1, 2], [3, 4]),],
                             measure=rms,
                             iteration_parameter=:n)

    @test_throws(MLJIteration.ERR_MISSING_TRAINING_CONTROL,
                 IteratedModel(model=model,
                               resampling=nothing,
                               controls=[Patience(), InvalidValue()]))

    @test_throws(MLJIteration.err_bad_iteration_parameter(:goo),
                 IteratedModel(model=model,
                               measure=mae,
                               iteration_parameter=:goo))
end


end

true
