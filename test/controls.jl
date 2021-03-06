module TestControls

using MLJIteration
using MLJBase
using Test
using ..DummyModel
using IterationControl
const IC = IterationControl

X, y = make_dummy(N=8);

@testset "WithIterationsDo" begin
    v = Int[]
    f(n) = (push!(v, n); n > 3)
    c = WithIterationsDo(f)
    m = MLJIteration.ICModel(machine(DummyIterativeModel(n=0), X, y), :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 1)
    @test !state.done
    @test v == [2, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 2, state)
    @test !state.done
    @test v == [2, 4]
    @test IC.takedown(c, 0, state) == (done = false, log="")

    v = Int[]
    f2(n) = (push!(v, n); n > 3)
    c = WithIterationsDo(f2, stop_if_true=true)
    m = MLJIteration.ICModel(machine(DummyIterativeModel(n=0), X, y), :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 1)
    @test !state.done
    @test v == [2, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 2, state)
    @test state.done
    @test v == [2, 4]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="Stop triggered by a `WithIterationsDo` control. ")

    v = Int[]
    f3(n) = (push!(v, n); n > 3)
    c = WithIterationsDo(f3, stop_if_true=true, stop_message="foo")
    m = MLJIteration.ICModel(machine(DummyIterativeModel(n=0), X, y), :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 1)
    @test !state.done
    @test v == [2, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 2, state)
    @test state.done
    @test v == [2, 4]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="foo")
end

resampler = MLJBase.Resampler(model=DummyIterativeModel(n=0),
                              resampling=Holdout(),
                              measure=[MLJBase.mae, MLJBase.rms])

@testset "WithEvaluationDo" begin
    v = Any[]
    f(e) = (push!(v, e.measurement); e.measurement[1] < 9.0)
    c = WithEvaluationDo(f)
    resampling_machine = machine(deepcopy(resampler), X, y)
    m = MLJIteration.ICModel(resampling_machine, :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 1)
    @test !state.done
    @test length(v) == 1
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 2, state)
    @test !state.done
    @test length(v) == 2
    @test IC.takedown(c, 0, state) == (done = false, log="")

    v = Any[]
    f2(e) = (push!(v, e.measurement[1]); e.measurement[1][1] < 9.0)
    c = WithEvaluationDo(f2, stop_if_true=true)
    resampling_machine = machine(deepcopy(resampler), X, y)
    m = MLJIteration.ICModel(resampling_machine, :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 1)
    @test !state.done
    @test length(v) == 1
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 2, state)
    @test state.done
    @test length(v) == 2
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="Stop triggered by a `WithEvaluationDo` control. ")

    v = Any[]
    f3(e) = (push!(v, e.measurement[1]); e.measurement[1][1] < 9.0)
    c = WithEvaluationDo(f3, stop_if_true=true, stop_message="foo")
    resampling_machine = machine(deepcopy(resampler), X, y)
    m = MLJIteration.ICModel(resampling_machine, :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 1)
    @test !state.done
    @test length(v) == 1
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, 2, state)
    @test state.done
    @test length(v) == 2
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="foo")
end

@testset "CycleLearningRate" begin
    c = CycleLearningRate(learning_rate_parameter=:learning_rate,
                          stepsize = 2,
                          lower = 0,
                          upper = 2)
    @test MLJIteration.one_cycle(c) == Float64[0, 1, 2, 1]

    c = CycleLearningRate(learning_rate_parameter=:learning_rate,
                          stepsize = 1,
                          lower = 0.5,
                          upper = 1.5)
    model = DummyIterativeModel(n=0, learning_rate=42)
    m = MLJIteration.ICModel(machine(model, X, y), :n, 0)
    state = @test_logs (:info, r"learning rate") IC.update!(c, m, 2, 1)
    state = @test_logs IC.update!(c, m, 1, 1)
    @test state == (n = 1, learning_rates = [0.5, 1.5])
    @test model.learning_rate == 0.5
    state = IC.update!(c, m, 0, 2, state)
    @test model.learning_rate == 1.5
    state = IC.update!(c, m, 0, 3, state)
    @test model.learning_rate == 0.5
end

end
true
