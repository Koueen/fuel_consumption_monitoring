import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import pandas as pd

class FuelDistanceLR:  # Inherits from object by default
    def __init__(self):
        self.model = LinearRegression()
    
    def __call__(self, df: pd.DataFrame) -> float:
        X, y = self._preprocess(df)
        self.model.fit(X, y)
        self.r2 = self.model.score(X,y)
        self.mae = mean_absolute_error(y, self.model.predict(X))
        return self._get_residuals(X, y)

    def _preprocess(self, df:pd.DataFrame):
        df = df.rename(
        columns={
            'Total fuel consumed (lifetime) (l)': 'fuel_consumed', # y
            'Total distance travelled (lifetime) (km)': 'distance_travelled', # X
            }
        )
        return df['distance_travelled'].values.reshape(-1,1), df['fuel_consumed'].values.reshape(-1,1)

    def _get_residuals(self,X, y) -> pd.DataFrame:
        res = np.abs(y - self.model.predict(X)).reshape(-1)
        return res
    
    def get_weights(self): 
        # We save the weight and bias in order to predict easily without storing any object.
        intercept = self.model.intercept_
        coefficient = self.model.coef_
        return intercept, coefficient
    
    def get_metrics(self):
        return self.r2, self.mae
    
