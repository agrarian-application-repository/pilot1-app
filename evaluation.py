from ultralytics import YOLO

import wandb
from src.configs.evaluate import check_eval_args
from src.configs.utils import parse_config_file, read_yaml_config
from src.logging.evaluation import log_eval_metrics
from src.logging.wandb import get_wandb_api_key, get_wandb_entity


def main():
    # Parse the config file from command line
    config_file_path = parse_config_file()
    # Read YAML config file and transform it into a dict
    eval_args = read_yaml_config(config_file_path)

    # Check arguments validity
    eval_args = check_eval_args(
        eval_args
    )  # TODO argument checks - for correct or YOLO checks

    print("PERFORMING EVALUATION WITH THE FOLLOWING ARGUMENTS:")
    print(eval_args, "\n")

    # Load the model
    model_checkpoint = eval_args.pop("model")
    model = YOLO(model_checkpoint)

    # Evaluate the model
    results = model.val(**eval_args)
    results_dict = results.__dict__

    # wandb_api_key = get_wandb_api_key()
    # wandb.login(key=wandb_api_key)

    # wandb_entity = get_wandb_entity()
    # wandb.init(entity=wandb_entity)
    wandb.init()

    eval_args["model"] = model_checkpoint  # re-insert before logging
    wandb.config.update(eval_args)

    results_scalar = {
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
    wandb.log(results_scalar)

    results_dict_scalar = {
        "Metrics/AP": results_dict["box"].ap.item(),
        "Metrics/AP50": results_dict["box"].ap50.item(),
        "Metrics/F1": results_dict["box"].f1.item(),
        "Metrics/Precision": results_dict["box"].p.item(),
        "Metrics/Recall": results_dict["box"].r.item(),
        "Metrics/mAP": results_dict["box"].map,
        "Metrics/mAP50": results_dict["box"].map50,
        "Metrics/mAP75": results_dict["box"].map75,
        "Metrics/Mean Precision": results_dict["box"].mp,
        "Metrics/Mean Recall": results_dict["box"].mr,
    }
    wandb.log(results_dict_scalar)

    results_curve = {
        "Plots/Precision-Recall(B)": {
            "x": results.curves_results[0][0],
            "y": results.curves_results[0][1][0],
            "x_label": "Recall",
            "y_label": "Precision",
        },
        "Plots/F1-Confidence(B)": {
            "x": results.curves_results[1][0],
            "y": results.curves_results[1][1][0],
            "x_label": "Confidence",
            "y_label": "F1",
        },
        "Plots/Precision-Confidence(B)": {
            "x": results.curves_results[2][0],
            "y": results.curves_results[2][1][0],
            "x_label": "Confidence",
            "y_label": "Precision",
        },
        "Plots/Recall-Confidence(B)": {
            "x": results.curves_results[3][0],
            "y": results.curves_results[3][1][0],
            "x_label": "Confidence",
            "y_label": "Recall",
        },
    }

    for k in results_curve.keys():
        # print(k)
        # print(results_curve[k])
        data = [[x, y] for (x, y) in zip(results_curve[k]["x"], results_curve[k]["y"])]
        table = wandb.Table(
            data=data,
            columns=[results_curve[k]["x_label"], results_curve[k]["y_label"]],
        )
        wandb.log(
            {
                f"Table/{k}": wandb.plot.line(
                    table,
                    results_curve[k]["x_label"],
                    results_curve[k]["y_label"],
                    title=k,
                )
            }
        )

    results_dict_curve = {
        "Plots/all_ap": {
            "x": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
            "y": results_dict["box"].all_ap[0],
            "x_label": "Confidence",
            "y_label": "AP",
        },
        "Plots/f1_curve": {
            "x": results_dict["box"].px,
            "y": results_dict["box"].f1_curve[0],
            "x_label": "Confidence",
            "y_label": "F1",
        },
        "Plots/p_curve": {
            "x": results_dict["box"].px,
            "y": results_dict["box"].p_curve[0],
            "x_label": "Confidence",
            "y_label": "Precision",
        },
        "Plots/r_curve": {
            "x": results_dict["box"].px,
            "y": results_dict["box"].r_curve[0],
            "x_label": "Confidence",
            "y_label": "Recall",
        },
        "Plots/prec_values": {
            "x": results_dict["box"].px,
            "y": results_dict["box"].prec_values[0],
            "x_label": "Recall",
            "y_label": "Precision",
        },
        "Plots/Precision-Recall": {
            "x": results_dict["box"].curves_results[0][0],
            "y": results_dict["box"].curves_results[0][1][0],
            "x_label": "Recall",
            "y_label": "Precision",
        },
        "Plots/F1-Confidence": {
            "x": results_dict["box"].curves_results[1][0],
            "y": results_dict["box"].curves_results[1][1][0],
            "x_label": "Confidence",
            "y_label": "F1",
        },
        "Plots/Precision-Confidence": {
            "x": results_dict["box"].curves_results[2][0],
            "y": results_dict["box"].curves_results[2][1][0],
            "x_label": "Confidence",
            "y_label": "Precision",
        },
        "Plots/Recall-Confidence": {
            "x": results_dict["box"].curves_results[3][0],
            "y": results_dict["box"].curves_results[3][1][0],
            "x_label": "Confidence",
            "y_label": "Recall",
        },
    }

    for k in results_dict_curve.keys():
        # print(k)
        # print(results_dict_curve[k])
        data = [
            [x, y]
            for (x, y) in zip(results_dict_curve[k]["x"], results_dict_curve[k]["y"])
        ]
        table = wandb.Table(
            data=data,
            columns=[
                results_dict_curve[k]["x_label"],
                results_dict_curve[k]["y_label"],
            ],
        )
        wandb.log(
            {
                f"Table/{k}": wandb.plot.line(
                    table,
                    results_dict_curve[k]["x_label"],
                    results_dict_curve[k]["y_label"],
                    title=k,
                )
            }
        )

    # results_artifact = {
    #    "Confusion Matrix": results.confusion_matrix,
    #    "Normalized Confusion Matrix": None,
    # }

    # Finish the W&B run
    wandb.finish()


if __name__ == "__main__":
    main()
