import pandas as pd
import os

# --- File Names ---
alert_file = 'alert.csv'
drowsy_file = 'drowsiness_data.csv'
output_file = 'training_data.csv'
# --------------------

print(f"Looking for {alert_file} and {drowsy_file}...")

# Check if the files exist
if not os.path.exists(alert_file) or not os.path.exists(drowsy_file):
    print(f"Error: Could not find '{alert_file}' or '{drowsy_file}'.")
    print("Please make sure all files are in the same directory as this script.")
else:
    try:
        # Load the datasets
        alert_df = pd.read_csv(alert_file)
        drowsy_df = pd.read_csv(drowsy_file)
        
        print(f"Loaded {len(alert_df)} alert rows and {len(drowsy_df)} drowsy rows.")

        # Add the 'IsDrowsy' label column
        alert_df['IsDrowsy'] = 0
        drowsy_df['IsDrowsy'] = 1

        # Combine the two DataFrames
        combined_df = pd.concat([alert_df, drowsy_df], ignore_index=True)

        # Save the new combined DataFrame
        combined_df.to_csv(output_file, index=False)

        print(f"\nSUCCESS! 🚀")
        print(f"Created '{output_file}' with {len(combined_df)} total rows.")
        
    except Exception as e:
        print(f"An error occurred: {e}")