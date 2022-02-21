MLJBase.is_wrapper(::Type{<:EitherIteratedModel}) = true
MLJBase.caches_data_by_default(::Type{<:EitherIteratedModel}) = false
MLJBase.load_path(::Type{<:DeterministicIteratedModel}) =
    "MLJIteration.DeterministicIteratedModel"
MLJBase.load_path(::Type{<:ProbabilisticIteratedModel}) =
    "MLJIteration.ProbabilisticIteratedModel"
MLJBase.package_name(::Type{<:EitherIteratedModel}) = "MLJIteration"
MLJBase.package_uuid(::Type{<:EitherIteratedModel}) =
    "614be32b-d00c-4edb-bd02-1eb411ab5e55"
MLJBase.package_url(::Type{<:EitherIteratedModel}) =
    "https://github.com/JuliaAI/MLJIteration.jl"
MLJBase.package_license(::Type{<:EitherIteratedModel}) = "MIT"

# inherited traits:
for trait in [:supports_weights,
              :supports_class_weights,
              :is_pure_julia,
              :input_scitype,
              :output_scitype,
              :target_scitype]
    quote
        # needed because traits are not always deducable from
        # the type (eg, `target_scitype` and `Pipeline` models):
        MLJBase.$trait(imodel::EitherIteratedModel) = $trait(imodel.model)
    end |> eval
    for T in [:DeterministicIteratedModel, :ProbabilisticIteratedModel]
        quote
            # try to get trait at level of types ("failure" here just
            # means falling back to `Unknown`):
            MLJBase.$trait(::Type{<:$T{M}}) where M = MLJBase.$trait(M)
        end |> eval
    end
end

