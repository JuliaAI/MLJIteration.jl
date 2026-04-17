module TestLogger

using Test
using MLJIteration
using MLJBase
using StatisticalMeasures
using ..DummyModel

X, y = make_dummy(N=20)

# A minimal logger that records each evaluation's first measurement into a buffer.
struct DummyLogger
    buffer::IOBuffer
end

MLJBase.log_evaluation(logger::DummyLogger, performance_evaluation) =
    write(logger.buffer, performance_evaluation.measurement[1])

@testset "explicit logger with Holdout" begin
    buffer = IOBuffer()
    logger = DummyLogger(buffer)

    model = DummyIterativeModel(n=0)
    controls = [Step(2), NumberLimit(5)]

    imodel = IteratedModel(
        model=model,
        resampling=Holdout(fraction_train=0.7),
        controls=controls,
        measure=l2,
        logger=logger,
    )
    mach = machine(imodel, X, y)
    fit!(mach, verbosity=0)

    # Each control cycle triggers one evaluate! call, which should log once.
    # With Step(2) and NumberLimit(5), we get 5 control cycles.
    seekstart(buffer)
    logged_values = Float64[]
    while !eof(buffer)
        push!(logged_values, read(buffer, Float64))
    end

    @test length(logged_values) == 5
    close(buffer)
end

@testset "logger=nothing produces no logging" begin
    model = DummyIterativeModel(n=0)
    controls = [Step(2), NumberLimit(3)]

    imodel = IteratedModel(
        model=model,
        resampling=Holdout(fraction_train=0.7),
        controls=controls,
        measure=l2,
        logger=nothing,
    )
    mach = machine(imodel, X, y)
    # Should run without error; log_evaluation(::Nothing, ...) is a no-op.
    fit!(mach, verbosity=0)
    @test true
end

@testset "default_logger integration" begin
    buffer = IOBuffer()
    logger = DummyLogger(buffer)
    default_logger(logger)

    model = DummyIterativeModel(n=0)
    controls = [Step(1), NumberLimit(3)]

    # No explicit logger; should pick up the global default.
    imodel = IteratedModel(
        model=model,
        resampling=Holdout(fraction_train=0.7),
        controls=controls,
        measure=l2,
    )
    mach = machine(imodel, X, y)
    fit!(mach, verbosity=0)

    seekstart(buffer)
    logged_values = Float64[]
    while !eof(buffer)
        push!(logged_values, read(buffer, Float64))
    end

    @test length(logged_values) == 3

    # Reset global default.
    default_logger(nothing)
    close(buffer)
end

@testset "logger not invoked when resampling=nothing" begin
    buffer = IOBuffer()
    logger = DummyLogger(buffer)

    model = DummyIterativeModel(n=0)
    controls = [Step(2), NumberLimit(3)]

    imodel = IteratedModel(
        model=model,
        resampling=nothing,
        controls=controls,
        logger=logger,
    )
    mach = machine(imodel, X, y)
    fit!(mach, verbosity=0)

    # No Resampler means no evaluate! call, so nothing should be logged.
    @test position(buffer) == 0
    close(buffer)
end

end

true
