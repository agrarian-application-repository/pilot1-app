import numpy as np


class HistoryTracker:
    """
     the object should act as follows. when  initialized, the input argument should be a lenght value.
     Each key should comprise of two arrays, a list of (x,y) pairs, an a boolean array indicating whether the (x,y) coords at that timestamp is valid.
     The HistoryTracker class should also have an update function that, given a list of IDs and the corresponding coordinates, for each ID inserts the (x,y) position in the array and sets it as valid.
     All other keys that are stored by the object but do not appear in the provided new set of id+coords pairs, should replicate the last value and set the boolean mask value as False to indicate the data is artificial.
     If a key that never appeared before appears in the update, that key is added and a list/mask of lenght equal to the initially specified lenght argument is created for that key, with all values in the mask excet the last one being false, and the list of couples is comprised of the same tuple repeated LEN times.
     Finally, at each update step the oldest value in the list of tuples/masks should be dropped similarly to a fifo queue where LEN acts as the queu lenght)
    """

    def __init__(self, window_size: int, update_period_frames: int, update_period_sec: float):
        # window_size indicates the size of the value and mask arrays
        self.window_size = window_size
        self.update_period_frames = update_period_frames
        self.update_period_sec = update_period_sec

        self.data = {}

        self.last_ids_list = []
        self.last_positions_list = []

    def update(self, ids, coordinates):

        """
        Update the dictionary with new tuples and boolean masks.
        - ids: List of keys to update
        - coordinates: List of (x, y) tuples corresponding to the ids
        """
        print("updating history")

        # Iterate over the ids and coordinates provided in the update
        for key, coords in zip(ids, coordinates):

            if key not in self.data:
                # If the key is new, initialize its tuple list and mask array
                self.data[key] = {
                    "coords": [coords] * self.window_size,
                    "valid": [False] * (self.window_size - 1) + [True],
                }

            else:
                # If the key already exists:
                # Add the new value and set the validity indicator to True
                self.data[key]["coords"].append(coords)
                self.data[key]["valid"].append(True)
                # And remove the oldest value to maintain the window size
                del self.data[key]["coords"][0]
                del self.data[key]["valid"][0]

        non_updated_keys = set(self.data.keys()) - set(ids)

        for key in non_updated_keys:
            # Repeat the last value, setting validity indicator to False
            last_value = self.data[key]["coords"][-1]
            self.data[key]["coords"].append(last_value)
            self.data[key]["valid"].append(False)
            # And remove the oldest value to maintain the window size
            del self.data[key]["coords"][0]
            del self.data[key]["valid"][0]

            # delete IDS that have been false for the entire time window
            if not any(self.data[key]["valid"]):
                del self.data[key]

        # save the set of ids and positions contributing to the last update
        self.last_ids_list = ids
        self.last_positions_list = coordinates

    def get_ids_history(self, ids):
        # ids must already exist
        if (not set(ids).issubset(set(self.data.keys()))) or (len(ids) == 0):
            raise ValueError(f"Cannot get history of an id that does not exists. Requested: {set(ids)}, Existing: {set(self.data.keys())}")

        tracked = {}
        for id in ids:
            tracked[id] = {
                "coords": np.array(self.data[id]["coords"]).T,  # (2, TSlenght)
                "valid": np.array(self.data[id]["valid"]).reshape(1, -1)  # (1, TSlenght)
            }

        return tracked

    def get_and_aggregate_ids_history(self, ids):
        """
        Retrieves and aggregates time series data for given IDs.

        Parameters:
            ids (list): List of IDs to fetch history for.

        Returns:
            coords_array (numpy.ndarray): (N, 2, TS_length) array of coordinates.
            valid_array (numpy.ndarray): (N, 1, TS_length) array of validity flags.
        """
        tracked = self.get_ids_history(ids)  # Get individual histories

        # Extract coordinate and validity arrays
        coords_list = [tracked[id_]["coords"] for id_ in ids]  # List of (2, TS_length) arrays
        valid_list = [tracked[id_]["valid"] for id_ in ids]  # List of (1, TS_length) arrays

        # Convert lists to numpy arrays of shape (N, 2, TS_length) and (N, 1, TS_length)
        aggregated_coords_array = np.stack(coords_list, axis=0)  # Shape: (N, 2, TS_length)
        aggregated_valid_array = np.stack(valid_list, axis=0)  # Shape: (N, 1, TS_length)

        return aggregated_coords_array, aggregated_valid_array

    def get_last_update_arrays(self):
        last_ids_array = np.array(self.last_ids_list)   # (N, )
        last_positions_array = np.array(self.last_positions_list)   # (N,2)
        return last_ids_array, last_positions_array
