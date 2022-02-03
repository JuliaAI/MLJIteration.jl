using Test

include("_dummy_model.jl")
using .DummyModel

@testset "utilities" begin
    include("utilities.jl")
end

@testset "ic_model" begin
    include("ic_model.jl")
end

@testset "controls" begin
    include("controls.jl")
end

@testset "constructors" begin
    include("constructors.jl")
end

@testset "core" begin
    include("core.jl")
end

@testset "traits" begin
    include("traits.jl")
end

@testset "serialization" begin
    include("serialization.jl")
end
