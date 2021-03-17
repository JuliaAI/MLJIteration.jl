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
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [2, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test !state.done
    @test v == [2, 4]
    @test IC.takedown(c, 0, state) == (done = false, log="")

    v = Int[]
    f(n) = (push!(v, n); n > 3)
    c = WithIterationsDo(f, stop_if_true=true)
    m = MLJIteration.ICModel(machine(DummyIterativeModel(n=0), X, y), :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [2, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test v == [2, 4]
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="Stop triggered by a `WithIterationsDo` control. ")

    v = Int[]
    f(n) = (push!(v, n); n > 3)
    c = WithIterationsDo(f, stop_if_true=true, stop_message="foo")
    m = MLJIteration.ICModel(machine(DummyIterativeModel(n=0), X, y), :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test v == [2, ]
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
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
    state = IC.update!(c, m, 0)
    @test !state.done
    @test length(v) == 1
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test !state.done
    @test length(v) == 2
    @test IC.takedown(c, 0, state) == (done = false, log="")

    v = Any[]
    f(e) = (push!(v, e.measurement[1]); e.measurement[1][1] < 9.0)
    c = WithEvaluationDo(f, stop_if_true=true)
    resampling_machine = machine(deepcopy(resampler), X, y)
    m = MLJIteration.ICModel(resampling_machine, :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test length(v) == 1
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test length(v) == 2
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="Stop triggered by a `WithEvaluationDo` control. ")

    v = Any[]
    f(e) = (push!(v, e.measurement[1]); e.measurement[1][1] < 9.0)
    c = WithEvaluationDo(f, stop_if_true=true, stop_message="foo")
    resampling_machine = machine(deepcopy(resampler), X, y)
    m = MLJIteration.ICModel(resampling_machine, :n, 0)
    IC.train!(m, 2)
    state = IC.update!(c, m, 0)
    @test !state.done
    @test length(v) == 1
    IC.train!(m, 2)
    state = IC.update!(c, m, 0, state)
    @test state.done
    @test length(v) == 2
    @test IC.takedown(c, 0, state) ==
        (done = true,
         log="foo")
end

@testset "Save" begin
    c = Save("serialization_test.jlso")
    m = MLJIteration.ICModel(machine(DummyIterativeModel(n=0), X, y), :n, 0)
    IC.train!(m, 2)
    state = @test_logs((:info, "Saving \"serialization_test1.jlso\". "),
                       IC.update!(c, m, 1))
    @test state.filenumber == 1
    IC.train!(m, 3)
    state = IC.update!(c, m, 0, state)
    @test state.filenumber == 2
    yhat = predict(IC.expose(m), X);

    deserialized_mach = MLJBase.machine("serialization_test2.jlso")
    yhat2 = predict(deserialized_mach, X)
    @test yhat2 ≈ yhat

    train_mach = machine(DummyIterativeModel(n=5), X, y)
    fit!(train_mach, verbosity=0)
    @test yhat ≈ predict(train_mach, X)
end


end


true
