import numpy as np
import scipy


def unwrap(mat_array):
    """Safely unwrap MATLAB cell arrays or nested objects."""
    while isinstance(mat_array, np.ndarray) and mat_array.dtype == object:
        mat_array = mat_array[0]
    return mat_array


def read_custom_epochs(mat_file):
    """Load a custom MATLAB epoched EEG file into an MNE EpochsArray."""
    import mne

    mat = scipy.io.loadmat(mat_file)
    fD = mat["fD"][0][0]

    # Load epoch data and convert from μV to V.
    data_list = [unwrap(fD[1][0][i]) for i in range(fD[1].shape[1])]
    data = np.stack(data_list, axis=0)  # shape: (n_epochs, n_channels, n_times)
    data = data * 1e-6
    n_epochs, n_channels, n_times = data.shape

    # Infer tmin and sampling frequency from the first epoch's time vector.
    times_list = [unwrap(fD[2][0][i]).squeeze() for i in range(fD[2].shape[1])]
    first_times = times_list[0]
    tmin = float(first_times[0])
    dt = np.diff(first_times)
    sfreq = 1.0 / np.mean(dt)

    # Read event labels and assign integer event IDs.
    event_labels = [str(unwrap(e)[0]) for e in fD[3][:, 0]]
    unique_labels = list(sorted(set(event_labels)))
    label_to_id = {label: i + 1 for i, label in enumerate(unique_labels)}
    events = np.array([[i, 0, label_to_id[event_labels[i]]] for i in range(len(event_labels))])

    # Read channel montage information.
    montage_data = unwrap(fD[4][0][0])
    pos = unwrap(montage_data[0])
    ch_type = [str(unwrap(ch)[0]) for ch in unwrap(montage_data[1])]
    unit_list = [str(unwrap(u)[0]) for u in unwrap(montage_data[2])]
    ch_names = [elem[0][0] for elem in montage_data[4]]
    transform = unwrap(montage_data[5])
    standard = str(unwrap(montage_data[6])[0])
    unit_str = str(unwrap(montage_data[7])[0])

    # Convert positions to meters if needed.
    if unit_str.lower() == "mm":
        pos_m = pos / 1000.0
    else:
        pos_m = pos  # Assume meters.

    # Create MNE montage and info.
    montage = mne.channels.make_dig_montage(ch_pos=dict(zip(ch_names, pos_m)), coord_frame="head")
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    info.set_montage(montage)

    # Create the EpochsArray.
    epochs = mne.EpochsArray(data, info, events=events, tmin=tmin, event_id=label_to_id)

    return epochs