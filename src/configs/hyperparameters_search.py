from typing import Any

from ray import tune


def check_hs_args(args: dict[str, Any]) -> dict[str, Any]:
    # TODO checks
    return args


def preprocess_search_args(args: dict[str, Any]) -> dict[str, Any]:
    for hs_k, hs_v in args["search"].items():
        if hs_v["type"] == "uniform":
            args["search"][hs_k] = tune.uniform(hs_v["min"], hs_v["max"])
        else:
            args["search"][hs_k] = tune.choice(hs_v["values"])

    return args
