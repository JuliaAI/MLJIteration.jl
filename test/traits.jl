module TestTraits

using MLJBase
using Test
using MLJIteration
using ..DummyModel
using StatisticalMeasures

model = DummyIterativeModel()
imodel = IteratedModel(model=model, measure=mae)

@test is_wrapper(imodel)
@test !MLJBase.caches_data_by_default(imodel)
@test !supports_weights(imodel)
@test !supports_class_weights(imodel)
@test load_path(imodel) == "MLJIteration.IteratedModel"
@test package_name(imodel) == "MLJIteration"
@test package_uuid(imodel) == "614be32b-d00c-4edb-bd02-1eb411ab5e55"
@test package_url(imodel) == "https://github.com/JuliaAI/MLJIteration.jl"
@test package_license(imodel) == "MIT"
@test is_pure_julia(imodel) == is_pure_julia(model)
@test input_scitype(imodel) == input_scitype(model)
@test output_scitype(imodel) == output_scitype(model)
@test target_scitype(imodel) == target_scitype(model)
@test constructor(imodel) == IteratedModel
@test reports_feature_importances(imodel) == reports_feature_importances(model)

end

true
