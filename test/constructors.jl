module TestConstructors

using MLJIteration
using MLJBase
using ..DummyModel
using Test

struct Foo <: MLJBase.Unsupervised end
struct Bar <: MLJBase.Deterministic end

@testset "constructors" begin
    model = DummyIterativeModel()
    @test_throws MLJIteration.ERR_NO_MODEL IteratedModel()
    @test_throws MLJIteration.ERR_NOT_SUPERVISED IteratedModel(model=Foo())
    @test_throws MLJIteration.ERR_NOT_SUPERVISED IteratedModel(model=Int)
    @test_throws MLJIteration.ERR_NEED_MEASURE IteratedModel(model=Bar())
    @test_throws MLJIteration.ERR_NEED_PARAMETER IteratedModel(model=Bar(),
                                                                measure=rms)
    iterated_model = @test_logs((:info, r"No measure"),
               IteratedModel(model=model))
    @test iterated_model.measure == RootMeanSquaredError()
    @test_logs IteratedModel(model=model, measure=mae)

    @test_logs IteratedModel(model=model, resampling=nothing)

    @test_logs((:info, r"`resampling` must be"),
               IteratedModel(model=model,
                             resampling=CV(),
                             measure=rms))
    @test_logs((:info, r"A `resampling` vector may contain"),
               IteratedModel(model=model,
                             resampling=[([1, 2], [3, 4]),
                                         ([3, 4], [1, 2])],
                             measure=rms))
    @test_logs IteratedModel(model=model,
                             resampling=[([1, 2], [3, 4]),],
                             measure=rms)

    @test_throws(MLJIteration.ERR_MISSING_TRAINING_CONTROL,
                 IteratedModel(model=model,
                               resampling=nothing,
                               controls=[Patience(), InvalidValue()]))
end

end

true
