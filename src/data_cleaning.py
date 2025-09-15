import os
import sys
import pandas as pd

# Add project root (ML_PROJECT_SEASION) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exception.execptions import MLProjectException
from src.data_ingestion import DataIngestion


class DataCleaning:
    """
    Class to clean and preprocess raw data.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean_data(self):
        try:
            # Drop duplicates
            self.data = self.data.drop_duplicates()

            # Handle missing values
            for col in self.data.columns:
                if self.data[col].dtype == "object":
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                else:
                    self.data[col].fillna(self.data[col].mean(), inplace=True)

            # Convert categorical to numerical
            self.data = pd.get_dummies(self.data, drop_first=True)

            return self.data

        except Exception as e:
            raise MLProjectException(
                error_message="Error while cleaning data",
                error_detail=e
            )


if __name__ == "__main__":
    try:
        file_path = os.path.join("data", "raw_data", "project_data.csv")

        # Load data
        data = DataIngestion(file_path).load_data()
        print("🔹 Raw Data:")
        print(data.head())

        # Clean data
        cleaner = DataCleaning(data)
        cleaned_data = cleaner.clean_data()
        print("\n✅ Cleaned Data:")
        print(cleaned_data.head())

    except MLProjectException as e:
        print(e)
