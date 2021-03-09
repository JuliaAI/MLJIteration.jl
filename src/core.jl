function MLJBase.fit(iterated_model::EitherIteratedModel, verbosity, data...)

    model = iterated_model.model

     # get name of iteration parameter:
     _iter = MLJBase.iteration_parameter(model)
    iter = _iter === nothing ? iterated_model.iteration_parameter : _iter

    # instantiate `train_mach`:
    mach = if iterated_model.resampling === nothing
        machine(model, data...; cache=iterated_model.cache)
    else
        resampler = MLJBase.Resampler(model=deepcopy(model),
                                  resampling=iterated_model.resampling,
                                  measure=iterated_model.measure,
                                  weights=iterated_model.weights,
                                  class_weights=iterated_model.class_weights,
                                  operation=iterated_model.operation,
                                  check_measure=iterated_model.check_measure,
                                  cache=iterated_model.cache)
        machine(resampler, data..., cache=false)
    end

    # instantiate the object to be iterated using IterationControl.jl:
    ic_model = ICModel(mach, iter, verbosity)

    # train with controls:
    creport = IterationControl.train!(ic_model,
                                      iterated_model.controls...,
                                      verbosity=verbosity)

    # retrain on all data if necessary:
    train_mach = IterationControl.expose(ic_model)
    production_mach = if iterated_model.final_train &&
        iterated_model.resampling !== nothing
        clone = deepcopy(model)
        # set iteration parameter to value at end of controlled
        # training:
        n_iterations = rget(train_mach.model, iter)
        rset!(clone, iter, n_iterations)
        production_mach = machine(clone,
                                  data...;
                                  cache=iterated_model.cache)
        verbosity < 1 ||
            @info "Retraining on all provided data. "*
            "To suppress, specify `final_train=false`. "
        fit!(production_mach, verbosity=verbosity - 1)
    else
        train_mach
    end

    verbosity < 1 || @info "Total of $(rget(production_mach.model, iter)) "*
        "iterations. "

    fitresult = production_mach
    cache = ic_model
    report = (model_report=MLJBase.report(production_mach),
              controls=creport,
              niterations=rget(train_mach, iter))
    return fitresult, cache, report

end

MLJBase.fitted_params(::EitherIteratedModel, fitresult) = (machine=fitresult,)

MLJBase.predict(::EitherIteratedModel, fitresult, Xnew) =
    predict(fitresult, Xnew)

MLJBase.transform(::EitherIteratedModel, fitresult, Xnew) =
    transform(fitresult, Xnew)
