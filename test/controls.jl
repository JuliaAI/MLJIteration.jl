module TestControls

using MLJIteration
using MLJBase
using Test
using ..DummyModel
using JLSO
using Serialization
using IterationControl
using StatisticalMeasures

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
                              measure=[mae, rms])

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
                t = PROJECTION_GIVEN_STR[$str](mach |> EXT_GIVEN_STR[$str])
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


jlso_save(filename, mach) = JLSO.save(filename, :machine => mach)
function jlso_machine(filename)
    mach = JLSO.load(filename)[:machine]
    MLJBase.restore!(mach)
    return mach
end

@testset "Save" begin
    # Test constructors
    filename = "serialization_test.jls"
    c_ = Save(filename)
    c = Save(filename=filename)
    @test c == c_
    # Test control for Serialization `serialize` and JLSO `save`
    for (save_fn,load_fn) in (serialize => MLJBase.machine, jlso_save => jlso_machine)
        c = Save(filename, method=save_fn)
        m = machine(DummyIterativeModel(n=2), X, y)
        fit!(m, verbosity=0)
        state = @test_logs((:info, "Saving \"serialization_test1.jls\". "),
        IterationControl.update!(c, m, 2, 1))
        @test state.filenumber == 1
        m.model.n = 5
        fit!(m, verbosity=0)
        state = IterationControl.update!(c, m, 0, 2, state)
        @test state.filenumber == 2
        yhat = predict(IterationControl.expose(m), X);

        deserialized_mach = load_fn("serialization_test2.jls")
        yhat2 = predict(deserialized_mach, X)
        @test yhat2 ≈ yhat

        train_mach = machine(DummyIterativeModel(n=5), X, y)
        fit!(train_mach, verbosity=0)
        @test yhat ≈ predict(train_mach, X)

        rm("serialization_test1.jls")
        rm("serialization_test2.jls")
    end
end

end
true
