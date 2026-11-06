"""
====================================================
Module : sequence_generator.py
Project: Airline Passenger Forecasting
Purpose: Generate sequences for RNN/LSTM
====================================================
"""

# Import required library
import numpy as np


class SequenceGenerator:
    """
    Convert a time series into input-output sequences
    using a sliding window approach.
    """

    def __init__(self, sequence_length=12):
        """
        Parameters
        ----------
        sequence_length : int
            Number of previous time steps used
            to predict the next value.
        """
        self.sequence_length = sequence_length

    def create_sequences(self, scaled_df):
        """
        Generate input-output sequences.

        Parameters
        ----------
        scaled_df : pandas.DataFrame
            Scaled passenger dataset.

        Returns
        -------
        X : numpy.ndarray
            Input sequences.

        y : numpy.ndarray
            Target values.
        """

        X = []
        y = []

        # Convert DataFrame to NumPy array
        data = scaled_df.values

        # Create sliding window sequences
        for i in range(len(data) - self.sequence_length):

            X.append(data[i:i + self.sequence_length])

            y.append(data[i + self.sequence_length])

        # Convert lists to NumPy arrays
        X = np.array(X)
        y = np.array(y)

        print("\nSequence Generation Completed.")
        print(f"Input Shape  : {X.shape}")
        print(f"Output Shape : {y.shape}")

        return X, y


# ==================================================
# Test Module
# ==================================================
if __name__ == "__main__":

    # Correct imports for:
    # python -m src.sequence_generator
    from src.data_loader import DataLoader
    from src.preprocessing import Preprocessor

    DATA_PATH = "data/airline_passengers.csv"

    # -----------------------------
    # Load Dataset
    # -----------------------------
    loader = DataLoader(DATA_PATH)
    df = loader.load_data()

    # -----------------------------
    # Scale Dataset
    # -----------------------------
    preprocessor = Preprocessor()
    scaled_df = preprocessor.scale_data(df)

    # -----------------------------
    # Generate Sequences
    # -----------------------------
    generator = SequenceGenerator(sequence_length=12)

    X, y = generator.create_sequences(scaled_df)

    print("\nFirst Input Sequence\n")
    print(X[0])

    print("\nFirst Target\n")
    print(y[0])

    print("\nSummary")
    print(f"Total Samples   : {len(X)}")
    print(f"Sequence Length : {generator.sequence_length}")
    print(f"Input Shape     : {X.shape}")
    print(f"Output Shape    : {y.shape}")

"""
Expected Output

Sequence Generation Completed.

Input Shape  : (132, 12, 1)
Output Shape : (132, 1)

First Input Sequence
[[0.01544402]
 [0.02702703]
 ...
 [0.1042471 ]]

First Target
[0.08494208]
"""