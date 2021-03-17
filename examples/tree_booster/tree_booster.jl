# # Basic demonstration of IteratedModel wrapper for a tree booster

using Pkg
Pkg.activate(@__DIR__)
Pkg.instantiate()

using MLJ
using MLJIteration
using Dates

using Plots
pyplot(size = (600, 300*(sqrt(5) - 1)))

using Statistics
using Random
Random.seed!(123)

MLJ.color_off()

X, y = make_moons(1000, rng=123)


# Import an model type:

EvoTreeClassifier = @load EvoTreeClassifier verbosity=0

model = EvoTreeClassifier(rng=123, η=0.005)

iterations = Int[0]
losses = Float64[0]

function update_plot(loss)
    push!(losses, loss)
    plot(iterations[2:end],
         losses[2:end],
         xlim=[1,300],
         ylim=[0,0.5]) |> display
end

imodel = IteratedModel(model=model,
                       resampling=Holdout(rng=123),
                       measures=[brier_loss, log_loss],
                       iteration_parameter=:nrounds,
                       controls=[Step(2),
                                 WithIterationsDo(i->push!(iterations, i)),
                                 WithLossDo(update_plot),
                                 GL(200),
                                 TimeLimit(Second(30))])

mach = machine(imodel, X, y) |> fit!


using Literate #src
Literate.markdown(@__FILE__, @__DIR__, execute=false) #src
Literate.notebook(@__FILE__, @__DIR__, execute=true) #src
