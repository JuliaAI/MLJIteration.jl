module TestUtilities

using MLJIteration
using MLJBase
using ..DummyModel
using Test

@testset "get and set iteration paramter" begin
    model = DummyIterativeModel()
    X, y = make_dummy(N=2)
    mach = machine(model, X, y)
    iter = MLJBase.iteration_parameter(model)
    @test MLJIteration.rget(model, iter) == 10
    @test MLJIteration.rget(mach, iter) == 10
    MLJIteration.rset!(model, iter, 42)
    @test MLJIteration.rget(model, iter) == 42
    @test MLJIteration.rget(mach, iter) == 42
    MLJIteration.rset!(mach, iter, 17)
    @test MLJIteration.rget(model, iter) == 17
    @test MLJIteration.rget(mach, iter) == 17
end

end

true
