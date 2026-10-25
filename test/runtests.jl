using Test


include("_dummy_model.jl")
using .DummyModel

test_files = [
    "utilities.jl",
    "ic_model.jl",
    "controls.jl",
    "constructors.jl",
    "core.jl",
    "traits.jl",
    "logger.jl",
]

files = isempty(ARGS) ? test_files : ARGS

for file in files
    quote
        @testset $file begin
            include($file)
        end
    end |> eval
end
