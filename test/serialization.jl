module TestControls

include("_dummy_model.jl")

using Test
using Serialization
using MLJBase
using MLJIteration
using IterationControl
using .DummyModel


@testset "Save" begin
    X, y = make_dummy(N=8);
    @test_throws Exception Save(filename="myfile.jlso")
    c = Save("serialization_test.jlso")
    m = machine(DummyIterativeModel(n=2), X, y)
    fit!(m, verbosity=0)
    state = @test_logs((:info, "Saving \"serialization_test1.jlso\". "),
    IterationControl.update!(c, m, 2, 1))
    @test state.filenumber == 1
    m.model.n = 5
    fit!(m, verbosity=0)
    state = IterationControl.update!(c, m, 0, 2, state)
    @test state.filenumber == 2
    yhat = predict(IterationControl.expose(m), X);

    deserialized_mach = MLJBase.machine("serialization_test2.jlso")
    yhat2 = predict(deserialized_mach, X)
    @test yhat2 ≈ yhat

    train_mach = machine(DummyIterativeModel(n=5), X, y)
    fit!(train_mach, verbosity=0)
    @test yhat ≈ predict(train_mach, X)

    rm("serialization_test1.jlso")
    rm("serialization_test2.jlso")
end

end

true
