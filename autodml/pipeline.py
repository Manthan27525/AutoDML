from autodml.preprocessing import Preprocessor
from autodml.modeling import ModelTrainer
from autodml.evaluation import Evaluator
from autodml.optimization import ModelOptimizer

from utils.logger import get_logger
from utils.exception import AutoDMLError

from config.models import Models

logger = get_logger(__name__)


class AutoDMLPipeline:
    def __init__(self, target, df):
        self.df = df
        self.target = target

    def run(self):
        try:
            logger.info("Running AUTODML Pipeline.")
            preprocessor = Preprocessor(df=self.df, target_column=self.target)
            x_train, x_test, y_train, y_test = preprocessor.process()
            problem = preprocessor.problem_type
            trainer = ModelTrainer(
                x_train=x_train,
                x_test=x_test,
                y_test=y_test,
                y_train=y_train,
                problem_type=problem,
            )
            model = trainer.get_model()
            optimizer = ModelOptimizer(
                model_name=model, task_type=problem, x_train=x_train, y_train=y_train
            )
            _, param = optimizer.optimize()
            evaluator = Evaluator(
                task_type=problem,
                model=model,
                param=param,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            results = evaluator.evaluate()

            models = Models.get_models()
            best_model = models[preprocessor.problem_type][model]
            best_params = optimizer.best_params

            print(results)
            logger.info("AUTODML Pipeline Finished.")
            return best_model, best_params, results

        except Exception as e:
            logger.error(str(e))
            raise AutoDMLError(
                message="Error Occurred WHile Running AUTODML Pipeline", details=str(e)
            )


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("temp/Corona.csv", skipfooter=30000, encoding="latin-1")
    target = "Sentiment"

    autodml = AutoDMLPipeline(df=df, target=target)
    model, parameters, results = autodml.run()

    print(model)
    print(parameters)
    print(results)
