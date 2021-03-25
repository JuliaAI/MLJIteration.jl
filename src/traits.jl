MLJBase.is_wrapper(::Type{<:EitherIteratedModel}) = true
MLJBase.caches_data_by_default(::Type{<:EitherIteratedModel}) = false
MLJBase.supports_weights(::Type{<:EitherIteratedModel{M}}) where M =
    MLJBase.supports_weights(M)
MLJBase.supports_class_weights(::Type{<:EitherIteratedModel{M}}) where M =
    MLJBase.supports_class_weights(M)
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
MLJBase.is_pure_julia(::Type{<:EitherIteratedModel{M}}) where {T,M} =
    MLJBase.is_pure_julia(M)
MLJBase.input_scitype(::Type{<:EitherIteratedModel{M}}) where {T,M} =
    MLJBase.input_scitype(M)
MLJBase.output_scitype(::Type{<:EitherIteratedModel{M}}) where {T,M} =
    MLJBase.output_scitype(M)
MLJBase.target_scitype(::Type{<:EitherIteratedModel{M}}) where {T,M} =
    MLJBase.target_scitype(M)
