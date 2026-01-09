def conformal_region_maker(
    X_train, X_test, y_seen, y_notseen, z_, method, method_name, params_fit
):

    if (method_name == "fcp") or (method_name == "fcp_krr"):
        predictive_region_maker = method.region(
            X_train, y_seen, X_test, y_notseen, params_fit
        )["region"]
    elif method_name == "scp":
        predictive_region_maker = method.region(
            X_train,
            y_seen,
            X_test,
            params_fit,
        )["region"]
    else:
        predictive_region_maker = method.region(
            X_train, y_seen, X_test, z_, params_fit
        )["up"]["region"]

    return predictive_region_maker


def method_maker(method_name, method, method_params, model):
    if method_name in ["fcp", "fcp_krr"]:
        res = method.fcp(
            model, method_params["non_conformity_maker"], method_params["score_params"]
        )
    elif method_name == "scp":
        res = method.scp(
            model,
            method_params["non_conformity_maker"],
            method_params["score_params"],
            method_params["proper_train_size"],
        )
    else:
        res = method.afcp(
            model, method_params["non_conformity_maker"], method_params["score_params"]
        )
    return res
