from external_libs_imports import *


def create_pipeline(clf_name, clf_info, debug=False):
    steps = []
    # Create classifier and set its params
    clf = clf_info['classifier']
    pl = Pipeline(steps=[('classifier', clf)])
    pl.set_params(**{f"classifier__{k}": v for k, v in clf.get_params().items()})

    # Get and set steps for this classifier
    for step_name, step_instance in clf_info['steps'].items():
        prfx_params = {f"{step_name}__{k}": v for k, v in step_instance.get_params().items()}
        pl.set_params(**prfx_params)
        steps.append((step_name, step_instance))

    # Add steps to pipeline
    pl.steps.extend(steps)

    # Store search params
    search_params = {f"classifier__{k}": v for k, v in clf_info['search_params'].items()}
    for step_name, step_instance in clf_info['steps'].items():
        search_params.update({f"{step_name}__{k}": v for k, v in step_instance.get_params().items()})


    if debug:
        print( f"Created pipeline for {clf_name} with steps {clf_info['steps'].keys()} and search params {search_params}")
    return pl, search_params
