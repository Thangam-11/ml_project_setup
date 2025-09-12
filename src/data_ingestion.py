import os
import sys
import pandas as pd

# Add project root (ML_PROJECT_SEASION) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from exception.execptions import MLProjectException


class DataIngestion:
    """
    Class to handle data ingestion for ML project.
    Loads raw data (CSV) into a pandas DataFrame.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self):
        """
        Load dataset from CSV file.
        Raises MLProjectException if any error occurs.
        """
        try:
            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"File not found: {self.file_path}")

            data = pd.read_csv(self.file_path)
            return data

        except Exception as e:
            raise MLProjectException(
                error_message="Error while loading data in DataIngestion",
                error_detail=e
            )


if __name__ == "__main__":
    try:
        # ✅ Correct file path
        file_path = os.path.join("data", "raw_data", "project_data.csv")
        ingestion = DataIngestion(file_path)
        df = ingestion.load_data()
        print("✅ Data Loaded Successfully!")
        print(df.head())  # Show first 5 rows

    except MLProjectException as e:
        print(e)
