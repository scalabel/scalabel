from fsdet.evaluation.evaluator import DatasetEvaluator


class ScalabelEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name): # initial needed variables
        self._dataset_name = dataset_name

    def reset(self): # reset predictions
        self._predictions = []

    def process(self, inputs, outputs): # prepare predictions for evaluation
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}
            if "instances" in output:
                prediction["instances"] = output["instances"]
            self._predictions.append(prediction)

    def evaluate(self): # evaluate predictions
        # TODO: call evaluation function from scalabel
        results = evaluate_predictions(self._predictions)
        return {
            "AP": results["AP"],
            "AP50": results["AP50"],
            "AP75": results["AP75"],
        }
