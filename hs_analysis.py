# https://docs.ray.io/en/latest/tune/examples/tune_analyze_results.html

from argparse import ArgumentParser
from ray import tune
from ray.tune import ResultGrid, Result
import json


def main(experiment_path: str):
    print(f"Loading results from {experiment_path}...")

    restored_tuner = tune.Tuner.restore(experiment_path, trainable="_tune")
    result_grid: ResultGrid = restored_tuner.get_results()

    if result_grid.errors:
        print("One or more trials failed!")
    else:
        print("No errors!")

    #for i, result in enumerate(result_grid):
    #    print(f"Trial #{i}: Configuration: {result.config}, Last Reported Metrics: {result.metrics}")

    # Iterate over results
    for i, result in enumerate(result_grid):
        if result.error:
            print(f"Trial #{i} had an error: ", result.error)
            continue

        print(
            f"Trial #{i} finished successfully with a mean accuracy metric of: ",
            result.metrics["metrics/mAP50-95(B)"]
        )

    results_df = result_grid.get_dataframe()
    filtered_results = results_df[["training_iteration", "metrics/mAP50-95(B)", "metrics/mAP50(B)", "metrics/precision(B)", "metrics/recall(B)"]]
    filtered_results = filtered_results.sort_values(by="metrics/mAP50-95(B)", ascending=False)
    print(filtered_results)
    filtered_results.to_csv(experiment_path + "/metrics.csv")

    # Get the result with the maximum test set `mean_accuracy`
    best_result: Result = result_grid.get_best_result(metric="metrics/mAP50-95(B)", mode="max")
    best_result_config = best_result.config

    print(f"Best result config: {best_result_config}")

    with open(experiment_path + "/best_config.json", "w") as f:
        json.dump(best_result_config, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    args = parser.parse_args()
    main(args.path)

# python hs_analysis.py --path /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/experiments_agrarian_hs/hs_search_1280_720_m
# python hs_analysis.py --path /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/experiments_agrarian_hs/hs_search_1920_1080_m
# python hs_analysis.py --path /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/experiments_agrarian_hs/hs_search_1280_720_x
# python hs_analysis.py --path /davinci-1/home/msarti/projects/AGRARIAN_PROJECT/AGRARIAN/experiments_agrarian_hs/hs_search_1920_1080_x