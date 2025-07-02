import h5py

# Load the .mat file
filename = 'RNN_ML_Ready_Data.mat'

with h5py.File(filename, 'r') as f:
    # Print top-level keys
    print("Top-level keys in the file:")
    for key in f.keys():
        print(key)

    # Show what's inside 'data_train'
    print("\nContents of 'data_train':", list(f['data_train'].keys()))

    X_ref = f['data_train']['X'][0][0]  # grab the reference
    Y_ref = f['data_train']['Y'][0][0]
    print(['data_train']['X'])
    # X = f[X_ref][:]
    # Y = f[Y_ref][:]