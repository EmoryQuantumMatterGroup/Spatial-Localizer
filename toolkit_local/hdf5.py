import h5py
import numpy as np


def save_numpy_to_hdf5(file_path, data, dataset_name="Dataset", overwrite=True, compression=None, compression_opts=None):
    """
    Save a NumPy array to an HDF5 file, with an option to overwrite only the specified dataset.
    
    Parameters:
        file_path (str): Path to the HDF5 file.
        data (np.ndarray): The NumPy array to save.
        dataset_name (str): Name of the dataset in the HDF5 file. Default is "Dataset".
        overwrite (bool): Whether to overwrite an existing dataset with the same name. Default is False.
        compression (str or None): Compression algorithm to use ('gzip', 'lzf', or 'szip'). Default is None (no compression).
        compression_opts (int or tuple, optional): Compression level or additional options for the chosen algorithm. 
                                                   For 'gzip', it's an integer (0-9, with 9 being the highest compression).
                                                   For 'szip', it's a tuple of form (options_mask, pixels_per_block).
                                                   Default is None (use library defaults).
    """
    with h5py.File(file_path, "a") as hdf_file:  # Open in 'a' mode to preserve other datasets
        if dataset_name in hdf_file:
            if not overwrite:
                raise ValueError(f"Dataset '{dataset_name}' already exists in {file_path}. Use overwrite=True to replace it.")
            else:
                # Only delete the specified dataset, not the entire file
                del hdf_file[dataset_name]
        
        # Create the dataset with optional compression
        hdf_file.create_dataset(dataset_name, data=data, compression=compression, compression_opts=compression_opts)
        print(f"Saved dataset '{dataset_name}' to {file_path} with compression='{compression}' and options='{compression_opts}'.")


def load_hdf5_to_numpy(file_path, dataset_name=None):
    """
    Load a NumPy array from an HDF5 file.
    
    Parameters:
        file_path (str): Path to the HDF5 file.
        dataset_name (str, optional): Name of the dataset to load. 
                                      If None, the first dataset found is used.
    
    Returns:
        np.ndarray: The loaded NumPy array.
    """
    with h5py.File(file_path, "r") as hdf_file:
        # If no dataset_name is provided, use the first available key
        if dataset_name is None:
            keys = list(hdf_file.keys())
            if not keys:
                raise ValueError("No datasets found in the HDF5 file.")
            dataset_name = keys[0]
        
        # Check if the dataset exists
        if dataset_name not in hdf_file:
            raise ValueError(f"Dataset '{dataset_name}' not found in the HDF5 file.")
        
        # Load the dataset into a NumPy array
        data = np.array(hdf_file[dataset_name])
    
    return data

        
def save_dict_to_hdf5(dictionary, hdf5_file, group_location="/"):
    """
    Save a dictionary to an HDF5 file at a specified group location,
    checking if groups already exist before creating them.
    """
    def save_group(group, dictionary):
        for key, value in dictionary.items():
            print(key, type(value))
            if isinstance(value, dict):  # Nested dictionary
                if key not in group:
                    subgroup = group.create_group(key)
                else:
                    subgroup = group[key]
                save_group(subgroup, value)
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                # Handle lists of strings
                dt = h5py.special_dtype(vlen=str)
                if key in group:
                    del group[key]  # Delete existing dataset before overwriting
                group.create_dataset(key, data=value, dtype=dt)
            elif isinstance(value, np.ndarray):
                if value.dtype.kind == 'U':  # Handle string arrays
                    dt = h5py.special_dtype(vlen=str)
                    if key in group:
                        del group[key]  # Delete existing dataset before overwriting
                    group.create_dataset(key, data=value, dtype=dt)
                else:
                    if key in group:
                        del group[key]
                    group.create_dataset(key, data=value)
            else:  # Scalar values
                group.attrs[key] = value

    with h5py.File(hdf5_file, "a") as h5file:  # Use "a" mode to append/update
        if group_location not in h5file:
            group = h5file.create_group(group_location)
        else:
            group = h5file[group_location]
        save_group(group, dictionary)

    print(f"Dictionary saved to '{group_location}' in '{hdf5_file}'.")


def load_dict_from_hdf5(hdf5_file, group_location="/"):
    """
    Load a dictionary from an HDF5 file at a specified group location,
    including support for lists of strings and other types.

    Parameters:
        hdf5_file (str): Path to the HDF5 file.
        group_location (str): Group location in the HDF5 file to load.

    Returns:
        dict: The loaded dictionary.
    """
    def load_group(group):
        result = {}
        for key, value in group.attrs.items():
            result[key] = value
        for key, value in group.items():
            if isinstance(value, h5py.Group):  # Nested group
                result[key] = load_group(value)
            else:  # Dataset
                data = value[()]
                if isinstance(data, np.ndarray) and data.dtype.kind in {'U', 'S'}:
                    result[key] = data.astype(str).tolist()  # Decode strings and convert to list if needed
                else:
                    result[key] = data.tolist() if isinstance(data, np.ndarray) else data
        return result

    with h5py.File(hdf5_file, "r") as h5file:
        if group_location not in h5file:
            raise ValueError(f"Group location '{group_location}' does not exist in the file.")
        group = h5file[group_location]
        return load_group(group)
    
    
    