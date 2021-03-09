## GETTING AND STTING ITERATION PARAMETER

"""
    rget(model, field::Union{Symbol,Expr})
    rget(mach, field::Union{Symbol,Expr})

Returns the value of the possibly nested parameter, `field`, from
`model` or `mach.model`.

"""
rget(model, field) =
    MLJBase.recursive_getproperty(model, field)
rget(mach::Machine, field) = rget(mach.model, field)

"""
    rset!(model, field::Union{Symbol,Expr}, i)
    rset!(mach, field::Union{Symbol,Expr}, i)

Change the value of the possibly nested parameter of `model` or `mach.model`
to `i`.

"""
function rset!(model, field, i)
    MLJBase.recursive_setproperty!(model, field, i)
    return nothing
end
rset!(mach::Machine, field, i) = rset!(mach.model, field, i)
