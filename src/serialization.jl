struct Save
    filename::String
    method
end

Save(filename="machine.jlso"; method=serialize) =
    Save(filename, method)


IterationControl.@create_docs(Save,
             header="Save(filename=\"machine.jlso\")",
             example="Save(\"run3/machine.jlso\", compression=:gzip)",
             body="Save the current state of the machine being iterated to "*
             "disk, using the provided `filename`, decorated with a number, "*
             "as in \"run3/machine_42.jlso\". The specified `kwargs` "*
             "are passed to the model-specific serializer "*
             "(JLSO for most Julia models).\n\n"*
             "For more on what is meant by \"the machine being iterated\", "*
             "see [`IteratedModel`](@ref).")

function IterationControl.update!(c::Save,
                                  ic_model,
                                  verbosity,
                                  n,
                                  state=(filenumber=0, ))
    filenumber = state.filenumber + 1
    root, suffix = splitext(c.filename)
    filename = string(root, filenumber, suffix)
    train_mach = IterationControl.expose(ic_model)
    verbosity > 0 && @info "Saving \"$filename\". "
    strain_mach = MLJBase.serializable(train_mach)
    c.method(filename, strain_mach)
    return (filenumber=filenumber, )
end
