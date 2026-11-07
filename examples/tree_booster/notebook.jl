# # Wrapping a Tree Booster in Iteration Controls

# > A demonstration of wrapping a gradient tree boosting model in
# > [MLJ](https://juliaml.ai)'s `IteratedModel` model wrapper

using MLJBase, MLJIteration, StatisticalMeasures # or just `using MLJ`
import IterationControl
using Dates
import DataFrames.DataFrame
import EvoTrees.EvoTreeClassifier
using Plots
using Statistics
using Random
Random.seed!(123)

# Load some synthetic training data:

X, y = make_moons(1000, rng=123); # a table and a vector

# Instantiate a model:

model = EvoTreeClassifier(rng=123, colsample=0.5)


# Define a callback for plotting:

iterations = Int[0]
losses = Float64[0]
plt = plot(; xlim=[1,100], ylim=[0,0.35], title="Out of sample loss", legend=false)


function update_plot()
    plot!(
        iterations[2:end],
        losses[2:end],
    ) |> display
    sleep(0.5) # for demo only
end;

# Wrap the booster in iteration controls:

imodel = IteratedModel(
    model,
    resampling=Holdout(fraction_train=0.7, rng=123),
    measures=[brier_loss, log_loss],
    iteration_parameter=:nrounds,
    controls=[
        Step(2),
        WithIterationsDo(i->(push!(iterations, i); i)),
        WithLossDo(l->(push!(losses, l); l)),
        IterationControl.skip(Callback(_->update_plot()), predicate=10),
        NumberSinceBest(3),
        GL(0.5),
        TimeLimit(Second(30)),
    ],
)

#-

mach = machine(imodel, X, y) |> fit!
savefig(plt, "out_of_sample_loss.svg");

# ![](out_of_sample_loss.svg)

step, with_iterationss_do, with_loss_do, callback, number_since_best, gl, time_limit =
    report(mach).traces


# We can get loss for every control cycle from the `WithLossDo` or `GL` trace:

with_loss_do

#-

with_loss_do.loss

#-

DataFrame(gl) |> println

