# automdl/pipeline.py

from automdl.preprocessing import Preprocessor
from automdl.modeling import ModelTrainer
from automdl.evaluation import Evaluator
from automdl.optimization import Optimizer


class AutoDMLPipeline:
    def __init__(self, task_type="regression"):
        self.task_type = task_type

        self.preprocessor = Preprocessor()
        self.trainer = ModelTrainer(task_type=self.task_type)
        self.evaluator = Evaluator(task_type=self.task_type)
        self.optimizer = Optimizer(task_type=self.task_type)

    def run(self, df, target_column):
        print("Starting AutoDML Pipeline...")

        X_train, X_test, y_train, y_test = self.preprocessor.process(df, target_column)
        best_model = self.optimizer.optimize(X_train, y_train)

        results = self.evaluator.evaluate(best_model, X_test, y_test)

        print("Pipeline Finished!")
        return best_model, results
