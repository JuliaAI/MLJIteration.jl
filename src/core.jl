# # HELPERS FOR `FIT` AND `UPDATE`

# either retrains new machine on all data or returns the machine
# trained using controls (embedded in `ic_model`):
function production_mach(iterated_model,
                         ic_model,
                         verbosity,
                         data...)
    train_mach = IterationControl.expose(ic_model)
    if iterated_model.final_train &&
        iterated_model.resampling !== nothing
        iteration_param = ic_model.iteration_parameter
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

function finish(prod_mach, ic_model, iterated_model, creport, verbosity)

    iteration_param = ic_model.iteration_parameter

    verbosity < 1 ||
        @info "Total of $(rget(prod_mach.model, iteration_param)) "*
        "iterations. "

    fitresult = prod_mach
    cache = (ic_model=ic_model,
             iterated_model=deepcopy(iterated_model))

    report = (model_report=MLJBase.report(prod_mach),
              controls=creport,
              n_iterations=rget(prod_mach, iteration_param))
    return fitresult, cache, report
end


# # IMPLEMENTATION OF MLJ MODEL INTERFACE

function MLJBase.fit(iterated_model::EitherIteratedModel, verbosity, data...)

    model = deepcopy(iterated_model.model)

     # get name of iteration parameter:
     _iter = MLJBase.iteration_parameter(model)
    iteration_param = _iter === nothing ?
        iterated_model.iteration_param : _iter

    # instantiate `train_mach`:
    mach = if iterated_model.resampling === nothing
        machine(model, data...; cache=iterated_model.cache)
    else
        resampler = MLJBase.Resampler(model=model,
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
                                verbosity,
                                data...)

    return finish(prod_mach, ic_model, iterated_model, creport, verbosity)

end

function MLJBase.update(iterated_model::EitherIteratedModel,
                        verbosity,
                        old_fitresult,
                        old_cache,
                        data...)

    ic_model, old_iterated_model = old_cache

    # cold restart if anything but `controls` or `model` altered:
    if !MLJBase.is_same_except(iterated_model,
                              old_iterated_model,
                              :model, :controls)
        return fit(iterated_model, verbosity, data...)
    end

    # otherwise, continue training with existing `ic_model`:
    # train with controls:
    creport = IterationControl.train!(ic_model,
                                      iterated_model.controls...,
                                      verbosity=verbosity)

    # retrains on all data if necessary:
    prod_mach = production_mach(iterated_model,
                                ic_model,
                                verbosity,
                                data...)

    return finish(prod_mach, ic_model, iterated_model, creport, verbosity)

end

MLJBase.fitted_params(::EitherIteratedModel, fitresult) = (machine=fitresult,)

MLJBase.predict(::EitherIteratedModel, fitresult, Xnew) =
    predict(fitresult, Xnew)

MLJBase.transform(::EitherIteratedModel, fitresult, Xnew) =
    transform(fitresult, Xnew)
