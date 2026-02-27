from load_data import build_dataset

# Folder that contains the .mat files
data_path = "P4"

# Load all subjects into one dataset
X, y = build_dataset(data_path)

# Print basic information to confirm everything loaded correctly
print("Dataset shape:", X.shape)   # Expected: (1400, 201, 6)
print("Labels shape:", y.shape)    # Expected: (1400,)
