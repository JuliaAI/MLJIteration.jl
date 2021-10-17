module TestControls

using MLJIteration
using MLJBase
using Test
using ..DummyModel
using IterationControl
const IC = IterationControl

const X, y = make_dummy(N=8);

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

# No need to re-test all logic for other simple controls, as
# defined by same code using metaprogramming. Some integration tests
# should suffice.

# functions accessing the training machine:
const EXT_GIVEN_STR = Dict(
    "fitted_params" => fitted_params,
    "report"        => report,
    "machine"       => identity,
    "model"         => m->m.model)

# some operation that converts the thing being extracted into a
# number, for comparisons:
const PROJECTION_GIVEN_STR = Dict(
    "fitted_params" => fp -> fp.fitresult['c'].avg,
    "report"        => rep -> first(first(rep)),
    "machine"       => mach -> first(first(report(mach))),
    "model"         => model -> model.learning_rate)

const N = 20

@testset "integration tests for some simple controls" begin
    for str in keys(EXT_GIVEN_STR)
        C = MLJIteration.NAME_GIVEN_STR[str]
        quote
            # by hand:
            trace_by_hand = []
            for n in 1:N
                local model = DummyIterativeModel(n=n)
                local mach = machine(model, X, y)
                fit!(mach, verbosity=0)
                t = mach |> EXT_GIVEN_STR[$str] |> PROJECTION_GIVEN_STR[$str]
                push!(trace_by_hand, t)
            end

            # with iteration control:
            trace = []
            model = DummyIterativeModel()
            c = $C(x -> push!(trace, PROJECTION_GIVEN_STR[$str](x)))
            controls = [Step(1), c, NumberLimit(N)]
            iterated_model = IteratedModel(model=model,
                                           resampling=nothing,
                                           measure=mae,
                                           controls=controls,
                                           retrain=false)
            mach = machine(iterated_model, X, y)
            fit!(mach, verbosity=0)

            # compare:
            @test trace == trace_by_hand
        end |> eval
    end
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
