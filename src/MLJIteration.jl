module MLJIteration

using MLJBase
using IterationControl
import IterationControl: debug, skip, composite
import IterationControl: update!, done, takedown, train!

export IteratedModel

const CONTROLS = vcat(IterationControl.CONTROLS,
                      [:WithIterationsDo,
                       :WithEvaluationDo,
                       :Save,
                       :CycleLearningRate])

# export all control types:
for control in CONTROLS
    eval(:(export $control))
end

const CONTROLS_DEFAULT = [Step(10),
                          Patience(5),
                          GL(),
                          TimeLimit(0.03), # about 2 mins
                          NotANumber()]

include("utilities.jl")
include("constructors.jl")
include("traits.jl")
include("ic_model.jl")
include("controls.jl")
include("core.jl")


end # module
