from sklearn.pipeline import Pipeline


class CreditScorePipeline:

    def __init__(self, preprocessor, model):

        self.preprocessor = preprocessor
        self.model = model

    def build(self):

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    self.preprocessor
                ),
                (
                    "model",
                    self.model
                )
            ]
        )

        return pipeline