import pandas as pd

from sklearn.model_selection import train_test_split

from config.config import (
    DATA_PATH,
    TARGET,
    TEST_SIZE,
    RANDOM_STATE
)

class DataLoader:

    def load_data(self):

        df = pd.read_csv(DATA_PATH)

        return df

    def split_data(self, df):

        X = df.drop(columns=[TARGET])

        y = df[TARGET]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )