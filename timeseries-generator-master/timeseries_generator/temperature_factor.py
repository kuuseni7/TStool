from typing import List, Any, Optional, Union

import numpy as np
from pandas import DataFrame
from pandas._libs.tslibs.timestamps import Timestamp

from timeseries_generator import BaseFactor
from timeseries_generator.utils import get_cartesian_product


class TemperatureFactor(BaseFactor):

    def __init__(
            self,
            feature: str,
            feature_values: List[Any],
            min_temperature_value: float = 20,
            #interval
            # max_factor_value: float = 50,
            col_name: str = "temperature_factor",
    ):
        super().__init__(col_name=col_name, features={feature: feature_values})

        self._feature = feature
        self._feature_values = feature_values
        # if min_temperature > max_factor_value:
        #     raise ValueError(
        #         f'min_factor_value: "{min_temperature}" > max_factor_value: "{max_factor_value}"'
        #     )
        self._min_temperature_value = min_temperature_value
        # self._max_factor_value = max_factor_value

    def generate(
            self,
            start_date: Union[Timestamp, str, int, float],
            end_date: Optional[Union[Timestamp, str, int, float]] = None,
    ) -> DataFrame:
        dr: DataFrame = self.get_datetime_index(
            start_date=start_date, end_date=end_date
        ).to_frame(index=False, name=self._date_col_name)

        # generate consistent temperature
        # current_temperature = min + ((max - min) * value)
        feat_factor = self._min_temperature_value + np.random.random(len(self._feature_values)) * 15

        # generate factor df
        factor_df = DataFrame(
            {self._feature: self._feature_values, self._col_name: feat_factor}
        )

        # cartesian product of factor df and datetime df
        return get_cartesian_product(dr, factor_df)
