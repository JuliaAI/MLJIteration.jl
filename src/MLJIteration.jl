module MLJIteration

using Serialization
using MLJBase
using IterationControl
import IterationControl: debug, skip, composite
import IterationControl: update!, done, takedown, train!

export IteratedModel, Save

const CONTROLS = vcat(IterationControl.CONTROLS,
                      [:WithIterationsDo,
                       :WithEvaluationDo,
                       :WithFittedParamsDo,
                       :WithReportDo,
                       :WithMachineDo,
                       :WithModelDo,
                       :CycleLearningRate])

const TRAINING_CONTROLS = [:Step, ]

# export all control types:
for control in CONTROLS
    eval(:(export $control))
end

const CONTROLS_DEFAULT = [Step(10),
                          Patience(5),
                          GL(),
                          TimeLimit(0.03), # about 2 mins
                          InvalidValue()]

include("utilities.jl")
include("controls.jl")

const Control = Union{[@eval($c) for c in CONTROLS]...}
const TrainingControl = Union{[@eval($c) for c in TRAINING_CONTROLS]...}

include("constructors.jl")
include("traits.jl")
include("ic_model.jl")
include("core.jl")
include("serialization.jl")



end # module
