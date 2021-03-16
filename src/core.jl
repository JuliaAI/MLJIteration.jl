# helper for `fit` and `update`:
function production_mach(iterated_model,
                         ic_model,
                         iteration_param,
                         verbosity,
                         data...)
    train_mach = IterationControl.expose(ic_model)
    if iterated_model.final_train &&
        iterated_model.resampling !== nothing
        clone = deepcopy(iterated_model.model)
        # set iteration parameter to value at end of controlled
        # training:
        n_iterations = rget(train_mach.model, iteration_param)
        rset!(clone, iteration_param, n_iterations)
        prod_mach = machine(clone,
                            data...;
                            cache=iterated_model.cache)
        verbosity < 1 ||
            @info "Retraining on all provided data. "*
            "To suppress, specify `final_train=false`. "
        fit!(prod_mach, verbosity=verbosity - 1)
    else
        train_mach
    end
end

function MLJBase.fit(iterated_model::EitherIteratedModel, verbosity, data...)

    model = iterated_model.model

     # get name of iteration parameter:
     _iter = MLJBase.iteration_parameter(model)
    iteration_param = _iter === nothing ?
        iterated_model.iteration_param : _iter

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
    ic_model = ICModel(mach, iteration_param, verbosity)

    # train with controls:
    creport = IterationControl.train!(ic_model,
                                      iterated_model.controls...,
                                      verbosity=verbosity)

    # retrains on all data if necessary:
    prod_mach = production_mach(iterated_model,
                                ic_model,
                                iteration_param,
                                verbosity,
                                data...)

    verbosity < 1 ||
        @info "Total of $(rget(prod_mach.model, iteration_param)) "*
        "iterations. "

    fitresult = prod_mach
    cache = ic_model
    report = (model_report=MLJBase.report(prod_mach),
              controls=creport,
              niterations=rget(prod_mach, iteration_param))
    return fitresult, cache, report

end

MLJBase.fitted_params(::EitherIteratedModel, fitresult) = (machine=fitresult,)

MLJBase.predict(::EitherIteratedModel, fitresult, Xnew) =
    predict(fitresult, Xnew)

MLJBase.transform(::EitherIteratedModel, fitresult, Xnew) =
    transform(fitresult, Xnew)
