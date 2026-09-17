### Save all RDMs

import os
from itertools import product

import mne
import numpy as np
import pandas as pd
import scipy

from src.load_eeg import read_custom_epochs


# Define metric.
metric = "correlation"

# Define the load path.
records_dir = "/scratch/guetlid95/datasets/mcdermott_2024/recordings/"
sub_list = [i for i in os.listdir(records_dir) if i.startswith("sub")]


def fix_leading_trailing(epochs, leading_range=range(1, 6), trailing_range=range(6, 10)):
    """Ensure that epochs strictly alternate between leading and trailing."""
    import numpy as np

    events = epochs.events
    to_drop = []

    for i in range(len(events) - 1):
        curr_code = events[i, -1]
        next_code = events[i + 1, -1]

        if curr_code in leading_range and next_code in leading_range:
            to_drop.append(i)  # Delete first leading.
        elif curr_code in trailing_range and next_code in trailing_range:
            to_drop.append(i + 1)  # Delete last trailing.

    # Remove duplicates and sort.
    to_drop = sorted(set(to_drop))

    if to_drop:
        epochs.drop(to_drop, reason="fix leading/trailing alternation")
        print(f"Dropped epochs at indices: {to_drop}")
    else:
        print("No violations found. All epochs alternate correctly.")

    return epochs


for sub in sub_list:
    for split in ["ac_fi_full", "ac_fi_block1", "ac_fi_block2", "ac_fi_quarter1", "ac_fi_quarter2"]:

        # A. Load and preprocess.
        load_path = os.path.join(records_dir, sub, "EEG", f"eTadff_{sub}.mat")
        save_path = os.path.join(records_dir, sub, "RDM", f"RDM_{sub}_{split}.pkl")

        # Load the epochs and resample.
        print(f"Processing {load_path}")
        epochs = read_custom_epochs(load_path)
        epochs = epochs.resample(100)

        # Select the requested split.
        if split == "ac_fi_block1":
            epochs = epochs[:len(epochs) // 2]  # Select the first half of trials.
        elif split == "ac_fi_block2":
            epochs = epochs[len(epochs) // 2:]  # Select the second half of trials.
        elif split == "ac_fi_quarter1":
            epochs = epochs[:len(epochs) // 4]  # Select the first quarter.
        elif split == "ac_fi_quarter2":
            epochs = epochs[len(epochs) // 4:2 * (len(epochs) // 4)]  # Select the second quarter.
        else:  # Returns the full split.
            pass

        # Ensure epochs alternate correctly and start with a leading trial.
        epochs = fix_leading_trailing(epochs)
        if {v:k for k, v in epochs.event_id.items()}[epochs.events[0][-1]].startswith("trailing"):
            epochs = epochs[1:]
        if len(epochs) % 2 == 1:  # Make sure epochs are of even length.
            epochs = epochs[:-1]

        # B. Split and concatenate leading and trailing epochs.
        leading_epochs = epochs[::2]
        trailing_epochs = epochs[1::2]

        # Concatenate data along the time axis.
        data_leading = leading_epochs.get_data()
        data_trailing = trailing_epochs.get_data()

        n_epochs, n_channels, n_times_leading = data_leading.shape
        _, _, n_times_trailing = data_trailing.shape

        # Create concatenated data and time vectors.
        new_data = np.zeros((n_epochs, n_channels, n_times_leading + n_times_trailing))
        new_times = np.concatenate([leading_epochs.times, trailing_epochs.times + leading_epochs.times[-1]])

        for i in range(n_epochs):
            new_data[i] = np.concatenate([data_leading[i], data_trailing[i]], axis=-1)

        # Create the combined event ID mapping.
        leading_names = [k.replace("leading_", "") for k in epochs.event_id if k.startswith("leading")]
        trailing_names = [k.replace("trailing_", "") for k in epochs.event_id if k.startswith("trailing")]

        used_combinations = []
        new_events = np.zeros((n_epochs, 3), dtype=int)
        counter = 1
        combined_event_id = {}

        for i in range(n_epochs):
            lead_event = epochs.events[2 * i, 2]  # Leading.
            trail_event = epochs.events[2 * i + 1, 2]  # Trailing.

            # Get event names.
            lead_name = [k.replace("leading_", "") for k, v in epochs.event_id.items() if v == lead_event][0]
            trail_name = [k.replace("trailing_", "") for k, v in epochs.event_id.items() if v == trail_event][0]

            combined_name = f"{lead_name}_{trail_name}"

            # Assign a unique event ID only if it has not been used yet.
            if combined_name not in combined_event_id:
                combined_event_id[combined_name] = counter
                counter += 1

            new_events[i] = [i, 0, combined_event_id[combined_name]]

        # Create a new EpochsArray.
        info = epochs.info.copy()
        new_epochs = mne.EpochsArray(
            new_data,
            info=info,
            events=new_events,
            event_id=combined_event_id,
            tmin=new_times[0]
        )

        new_epochs.event_id

        # C. Map joint conditions onto valid/invalid/control conditions.
        translation_dict = {
            ('Barn_church'):"valid_church",
            ('Barn_conference_room'):"invalid_confroom",
            ('beach_church'):"valid_church",
            ('beach_conference_room'):"invalid_confroom",
            ('library_church'):"invalid_church",
            ('library_conference_room'):"valid_confroom",
            ('cave_castle'):"control_castle",
            ('cave_forest'):"control_forest",
            ('restaurant_church'):"invalid_church",
            ('restaurant_conference_room'):"valid_confroom"
        }

        # Number the translated event conditions.
        new_event_id = {k:(v+1) for v, k in enumerate(['valid_church', 'valid_confroom', 'invalid_confroom', 'invalid_church', 'control_castle', 'control_forest'])}

        # Update event IDs using the old-to-new mapping.
        inverted_old_event_id = {v: k for k, v in new_epochs.event_id.items()}
        new_epochs.events[:, 2] = np.array([new_event_id[translation_dict[inverted_old_event_id[e]]] for e in new_epochs.events[:, 2]])
        new_epochs.event_id = new_event_id
        epochs = new_epochs

        # Group epochs by unique event ID.
        event_ids = epochs.events[:, 2]
        unique_events = np.unique(event_ids)

        # Create the event lookup dictionary and group data by event.
        event_id_lookup = {v: k for k, v in epochs.event_id.items()}
        event_dict = {event_id_lookup[event]: epochs.get_data()[event_ids == event] for event in unique_events}

        # Compute the mean across trials for each trial group.
        event_mean_dict = {event: val.mean(axis=0) for event, val in event_dict.items()}  # (channels, time)

        # Convert event means to DataFrames and concatenate them with a MultiIndex.
        event_mean_df = pd.concat(
            {event: pd.DataFrame(data.T, columns=epochs.ch_names, index=np.round(epochs.times, 5))
             for event, data in event_mean_dict.items()},
            names=['event', 'time']
        )

        # Compute the time-resolved RDM.
        time_rdm_df = pd.DataFrame([
            scipy.spatial.distance.pdist(event_mean_df.xs(t, level='time').values, metric='correlation')
            for t in event_mean_df.index.get_level_values('time').unique()
        ], index=event_mean_df.index.get_level_values('time').unique())
        time_rdm_df.index.name = 'time'

        # Save the RDM.
        print(f"Saving {load_path} to {save_path}")
        time_rdm_df.to_pickle(save_path)