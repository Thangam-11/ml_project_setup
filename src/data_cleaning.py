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

    def convert_to_numeric(self):
        """
        Convert specific columns to numeric types.
        """
        try:
            if "ammonia" in self.data.columns:
                self.data["ammonia"] = pd.to_numeric(self.data["ammonia"], errors="coerce")
            if "is_safe" in self.data.columns:
                self.data["is_safe"] = pd.to_numeric(self.data["is_safe"], errors="coerce")
                # Drop rows where is_safe is still NaN
                self.data = self.data.dropna(subset=["is_safe"])
            return self.data
        except Exception as e:
            raise MLProjectException(
                error_message="Error converting columns to numeric",
                error_detail=e
            )

    def handle_missing_values(self):
        """
        Handle missing values: numeric -> mean, categorical -> mode.
        Reports percentage of missing values.
        """
        try:
            # Report missing value percentages
            features_with_na = [f for f in self.data.columns if self.data[f].isnull().sum() > 0]
            for feature in features_with_na:
                missing_pct = round(self.data[feature].isnull().mean() * 100, 5)
                print(f"⚠️ {feature}: {missing_pct}% missing values")

            # Impute missing values
            for col in self.data.columns:
                if self.data[col].dtype == "object":
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                else:
                    self.data[col].fillna(self.data[col].mean(), inplace=True)

            return self.data
        except Exception as e:
            raise MLProjectException(
                error_message="Error while handling missing values",
                error_detail=e
            )

    def encode_categorical(self):
        """
        Convert categorical variables to dummy/indicator variables.
        """
        try:
            self.data = pd.get_dummies(self.data, drop_first=True)
            return self.data
        except Exception as e:
            raise MLProjectException(
                error_message="Error encoding categorical variables",
                error_detail=e
            )

    def clean_target(self):
        """
        Keep only valid binary values (0,1) for target column is_safe.
        """
        try:
            if "is_safe" in self.data.columns:
                self.data["is_safe"] = pd.to_numeric(self.data["is_safe"], errors="coerce")
                # Keep only rows where is_safe is 0 or 1
                self.data = self.data[self.data["is_safe"].isin([0, 1])]
            return self.data
        except Exception as e:
            raise MLProjectException(
                error_message="Error cleaning target column",
                error_detail=e
            )

    def clean_data(self):
        """
        Perform full cleaning: convert numeric, clean target, handle missing values, encode categorical.
        """
        try:
            self.data = self.data.drop_duplicates()
            self.data = self.convert_to_numeric()
            self.data = self.clean_target()        # <-- FIXED step
            self.data = self.handle_missing_values()
            self.data = self.encode_categorical()
            return self.data
        except Exception as e:
            raise MLProjectException(
                error_message="Error in clean_data pipeline",
                error_detail=e
            )


if __name__ == "__main__":
    try:
        file_path = os.path.join("data", "raw_data", "project_data.csv")

        # Load raw data
        data = DataIngestion(file_path).load_data()
        print("🔹 Raw Data Sample:")
        print(data.head())

        # Clean data
        cleaner = DataCleaning(data)
        cleaned_data = cleaner.clean_data()
        print("\n✅ Cleaned Data Sample:")
        print(cleaned_data.head())

    except MLProjectException as e:
        print(e)
