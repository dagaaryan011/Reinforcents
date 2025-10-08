import numpy as np
import pandas as pd
import os

def convert_npy_to_csv(npy_file_path: str, output_csv_path: str):
    print(f"Loading data from: {npy_file_path}")
    if not os.path.exists(npy_file_path):
        print("Error: File not found.")
        return

    try:
        dataset = np.load(npy_file_path, allow_pickle=True)
        if dataset.size == 0:
            print("The .npy file is empty.")
            return

        # Corrected Logic:
        # For each row, take the LAST feature vector from the sequence (row[0][-1])
        # and add the output label (row[1]) to it.
        flattened_data = [row[0][-1] + [row[1]] for row in dataset]

        # Corrected Columns:
        # Add the 'adx_signal' column to match the 5 features from datasetmaker.py
        # plus the final 'output' column.
        column_names = [
            'macd_signal', 
            'rsi_signal', 
            'stoch_signal', 
            'status_signal', 
            'adx_signal', 
            'output'
        ]
        
        df = pd.DataFrame(flattened_data, columns=column_names)

        print(f"Saving {len(df)} rows to: {output_csv_path}")
        df.to_csv(output_csv_path, index=False)
        
        print("\n--- Conversion Complete ---")
        print("First 5 rows of your new CSV:")
        print(df.head())

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # --- Manually set your file paths here ---
    input_npy_file = r'D:\NetworkPrediction\training_data\training_data4.npy'
    output_csv_file = 'inspected_data.csv'
    
    convert_npy_to_csv(npy_file_path=input_npy_file, output_csv_path=output_csv_file)