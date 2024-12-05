from typing import Any

import wandb
from src.logging.wandb_access import get_wandb_api_key, get_wandb_entity


# TODO implement
def log_eval_segmentation_results(results, args: dict[str: Any]) -> None:
    raise NotImplementedError("Segmentation evaluation logging not implemented yet")

    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    wandb.init()

    wandb.config.update(args)

    metrics = {
        # todo
    }

    # Finish the W&B run
    wandb.finish()


def log_eval_detection_results(results, args: dict[str, Any]) -> None:
    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)

    wandb.init()

    wandb.config.update(args)

    scalar_metrics = {
        "Speed/Preprocess": results.speed["preprocess"],
        "Speed/Inference": results.speed["inference"],
        "Speed/Loss": results.speed["loss"],
        "Speed/Postprocess": results.speed["postprocess"],

        "Metrics/Precision(B)": results.results_dict["metrics/precision(B)"],
        "Metrics/Recall(B)": results.results_dict["metrics/recall(B)"],
        "Metrics/mAP50(B)": results.results_dict["metrics/mAP50(B)"],
        "Metrics/mAP50-95(B)": results.results_dict["metrics/mAP50-95(B)"],
        "Metrics/Fitness": results.results_dict["fitness"],
    }
    wandb.log(scalar_metrics)

    curves = {
        "Precision-Recall(B)": {
            "x": results.curves_results[0][0],
            "y": results.curves_results[0][1][0],
            "x_label": "Recall",
            "y_label": "Precision",
        },
        "F1-Confidence(B)": {
            "x": results.curves_results[1][0],
            "y": results.curves_results[1][1][0],
            "x_label": "Confidence",
            "y_label": "F1",
        },
        "Precision-Confidence(B)": {
            "x": results.curves_results[2][0],
            "y": results.curves_results[2][1][0],
            "x_label": "Confidence",
            "y_label": "Precision",
        },
        "Recall-Confidence(B)": {
            "x": results.curves_results[3][0],
            "y": results.curves_results[3][1][0],
            "x_label": "Confidence",
            "y_label": "Recall",
        },
    }

    for curve in curves.keys():
        data = [[x, y] for (x, y) in zip(curves[curve]["x"], curves[curve]["y"])]
        table = wandb.Table(
            data=data,
            columns=[curves[curve]["x_label"],
                     curves[curve]["y_label"]]
        )

        wandb.log({
            f"Plots/{curve}": wandb.plot.line(
                table=table,
                x=curves[curve]["x_label"],
                y=curves[curve]["y_label"],
                title=curve,
            )
        })

    # Finish the W&B run
    wandb.finish()
