#app/utils/data_store.py

import pandas as pd
import uuid


class DataStore:
    def __init__(self):
        self._store = {}

    def save(self, df: pd.DataFrame, dataset_id: str = None) -> str:
        """
        Save a DataFrame to the in-memory store.

        If `dataset_id` is provided (e.g. reusing an existing session),
        the same ID is used and the stored DataFrame is overwritten.
        If `dataset_id` is None, a new UUID is generated.

        Returns the dataset_id (new or reused).
        """
        if dataset_id is None:
            dataset_id = str(uuid.uuid4())
        self._store[dataset_id] = df
        return dataset_id

    def get(self, dataset_id: str) -> pd.DataFrame | None:
        return self._store.get(dataset_id)


# singleton instance
data_store = DataStore()