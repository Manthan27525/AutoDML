from autodml.preprocessing import Preprocessor
from autodml.modeling import ModelTrainer
from autodml.evaluation import Evaluator
from autodml.optimization import ModelOptimizer
from autodml.data_analysis import DataAnalyzer
from utils.logger import get_logger
from utils.exception import AutoDMLError
from autodml.registry import ModelRegistry

Models = ModelRegistry()
logger = get_logger(__name__)


class AutoDMLPipeline:
    def __init__(self, target, df):
        self.df = df
        self.target = target

    def run(self):
        try:
            logger.info("Running AUTODML Pipeline.")
            analyzer = DataAnalyzer(df=self.df, target=self.target)
            analysis = analyzer.generate_report()
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

            models = Models.get_models(task_type=problem)
            best_model = models[model]
            best_params = optimizer.best_params

            print(results)
            logger.info("AUTODML Pipeline Finished.")
            return best_model, best_params, results, analysis

        except Exception as e:
            logger.error(str(e))
            raise AutoDMLError(
                message="Error Occurred WHile Running AUTODML Pipeline", details=str(e)
            )


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("temp/spam.csv")
    target = "Category"

    autodml = AutoDMLPipeline(df=df, target=target)
    model, parameters, results, analysis = autodml.run()

    print(model)
    print(parameters)
    print(results)
    print(analysis)
