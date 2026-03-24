from autodml.preprocessing import Preprocessor
from autodml.modeling import ModelTrainer
from autodml.evaluation import Evaluator
from autodml.optimization import ModelOptimizer
from autodml.data_analysis import DataAnalyzer
from autodml.utils.logger import get_logger
from autodml.utils.exception import AutoDMLError
from autodml.registry import ModelRegistry
from autodml.data_visualization import DataVisualizer
import pickle
from autodml.nlp.nltk_setup import download_nltk_data

Models = ModelRegistry()
logger = get_logger(__name__)


class AutoDMLPipeline:
    def __init__(self, target, df):
        download_nltk_data()
        self.df = df
        self.target = target
        self.preprocessor = None
        self.best_model_obj = None
        self.input_features = None
        self.results = None
        self.analysis = None

    def run(self):
        try:
            logger.info("Running AUTODML Pipeline.")
            analyzer = DataAnalyzer(df=self.df, target=self.target)
            analysis = analyzer.generate_report()

            self.preprocessor = Preprocessor(df=self.df, target_column=self.target)
            x_train, x_test, y_train, y_test, meta = self.preprocessor.process()

            problem = self.preprocessor.problem_type
            self.input_features = self.preprocessor.input_features
            trainer = ModelTrainer(
                x_train=x_train,
                x_test=x_test,
                y_test=y_test,
                y_train=y_train,
                problem_type=problem,
            )

            model_name = trainer.get_model()
            optimizer = ModelOptimizer(
                model_name=model_name,
                task_type=problem,
                x_train=x_train,
                y_train=y_train,
            )
            _, param = optimizer.optimize()

            evaluator = Evaluator(
                task_type=problem,
                model=model_name,
                param=param,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
            )
            results = evaluator.evaluate()

            with open("data/model/model.pkl", "rb") as f:
                self.best_model_obj = pickle.load(f)

            visualizer = DataVisualizer(
                model=self.best_model_obj,
                feature_names=meta["inputs"],
                df=self.df,
                target=self.target,
            )

            plots = visualizer.generate_all_visuals()
            reports = visualizer.generate_pdf_report(plots=plots)

            logger.info("AUTODML Pipeline Finished.")
            self.results = results
            self.analysis = analysis
            self.visualizations = reports

            return (
                model_name,
                param,
                results,
                analysis,
                self.visualizations,
                meta,
                self.best_model_obj,
            )

        except Exception as e:
            logger.error(str(e))
            raise AutoDMLError(
                message="Error Occurred While Running AUTODML Pipeline", details=str(e)
            )

    def predict(self, input_data):
        logger.info("Initiating Prediction.")
        try:
            if self.best_model_obj is None or self.preprocessor is None:
                raise AutoDMLError(
                    message="Prediction Error",
                    details="Pipeline must be 'run' before calling predict.",
                )

            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data.copy()

            processed_input = self.preprocessor.prediction_preprocessor(input_df)

            predictions = self.best_model_obj.predict(processed_input)

            if self.preprocessor.problem_type == "Classification":
                target_encoder = self.preprocessor.encoders.get("_TARGET_")
                if target_encoder:
                    predictions = target_encoder.inverse_transform(predictions)

            return predictions

        except Exception as e:
            logger.error(f"Prediction Failed: {str(e)}")
            raise AutoDMLError(message="Prediction Failed", details=str(e))


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("temp/spam.csv")
    target = "Category"

    autodml = AutoDMLPipeline(df=df, target=target)
    model_name, param, results, analysis = autodml.run()
    print(results)
