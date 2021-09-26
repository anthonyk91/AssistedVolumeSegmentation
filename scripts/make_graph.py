# script to make graph of connected components in a volume
import argparse
import logging
import pickle
from collections import defaultdict
from typing import Dict, Set, Tuple, Union

import cupy as cp
import cupyx.scipy.ndimage as cpnd
import h5py
import numpy as np

VolumeFile = Tuple[str, str]
ArrayTypes = Union[np.ndarray, cp.ndarray, h5py.Dataset, VolumeFile]

TILE_SIZE = np.array([500, 500, 500])
EROSION_STEPS = 10
FOREGROUND_THRESHOLD = 0.5


def read_h5(volume_file: str, data_label: str):
    h5file = h5py.File(volume_file)
    return h5file[data_label]


def get_unique(in_array: cp.ndarray):
    """
    Find unique values in array.  assume array is shape (x,y), and want to find all unique values over y (ie reduce
    the y dimension)

    :param cp.ndarray in_array: Input array of shape (x,y), which will be reduced over the y dimension
    :returns: Array of unique values of in_array, shape (x, unique_y)
    """

    sorted = cp.sort(in_array, axis=1)
    new_values = (sorted[:, 1:] != sorted[:, :-1]).any(axis=0)  # shape (y,)
    # add first value as a new value
    new_values_full = cp.concatenate([cp.array([1], dtype="bool"), new_values])
    chosen_newvalues = sorted[:, new_values_full]
    return chosen_newvalues


def process_tile(
    tile_idx: Tuple[int, int, int],
    h5array: h5py.Dataset,
    assoc_map: defaultdict,
    tile_edges: Dict,
    tile_components: Dict,
):
    """
    Find content for given tile.  extend boundary past reference region in order to
    allow accurate erosion result within the reference region.  assume that erosion
    uses a filter of size 3x3, which requires 1 pixel of surrounding region for each
    erosion step
    """
    tile_idx_arr = np.array(tile_idx)
    reference_start = tile_idx_arr * TILE_SIZE
    reference_max = reference_start + TILE_SIZE

    extended_start = reference_start - EROSION_STEPS
    extended_max = reference_max + EROSION_STEPS

    # get extended region from data.  to handle cases at the edges where the extended
    # region is not populated by data, create a zero-filled array and then copy the
    # defined region
    extended_size = TILE_SIZE + 2 * EROSION_STEPS
    extended_region = cp.zeros(extended_size, dtype=bool)
    valid_start = np.maximum(extended_start, 0)
    source_size = np.array(h5array.shape[1:])
    valid_end = np.minimum(extended_max, source_size)
    valid_data_raw = cp.array(
        h5array[
            0,
            valid_start[0] : valid_end[0],
            valid_start[1] : valid_end[1],
            valid_start[2] : valid_end[2],
        ]
    )
    valid_data_bool = cp.greater_equal(valid_data_raw, FOREGROUND_THRESHOLD)

    insert_start = np.maximum(-extended_start, 0)
    insert_end = extended_size - np.maximum(extended_max - source_size, 0)
    extended_region[
        insert_start[0] : insert_end[0],
        insert_start[1] : insert_end[1],
        insert_start[2] : insert_end[2],
    ] = valid_data_bool

    # produce eroded results
    current_region = extended_region
    erosion_results = [current_region]
    for _ in range(EROSION_STEPS):
        eroded_region = cpnd.binary_erosion(current_region)
        erosion_results.append(eroded_region)
        current_region = eroded_region

    # find connected components for each erosion level
    label_results = [
        cpnd.label(erosion_result) for erosion_result in erosion_results
    ]

    # find size and bounds of each component, and relationships between connected components in each level
    tile_component_details = []
    prev_label_array = None
    for label_array, label_count in label_results:
        level_component_details = {}
        for label_num in range(1, label_count + 1):
            value_mask = label_array == label_num
            # find bounds
            xvals, yvals, zvals = cp.where(value_mask)
            bounds = cp.stack(
                [
                    cp.min(xvals),
                    cp.max(xvals) + 1,
                    cp.min(yvals),
                    cp.max(yvals) + 1,
                    cp.min(zvals),
                    cp.max(zvals) + 1,
                ]
            ).get()
            center = cp.array(
                [
                    cp.mean(xvals),
                    cp.mean(yvals),
                    cp.mean(zvals),
                ]
            ).get()
            size = int(cp.sum(value_mask))
            # find parent as the component label in the previous erosion level.  there should
            # always be a unique parent component that covers all defined pixels for this component
            # choose an arbitrary position within this region
            if prev_label_array is None:
                parent_component_num = None
            else:
                parent_component_num = prev_label_array[
                    xvals[0], yvals[0], zvals[0]
                ]
            level_component_details[label_num] = (
                bounds,
                center,
                size,
                parent_component_num,
            )
            prev_label_array = label_array
        tile_component_details.append(level_component_details)
    tile_components[tile_idx] = tile_component_details

    # find connections between tiles by comparing with preceding neighbours
    for assoc in ["x", "y", "z"]:
        if assoc == "x":
            if tile_idx[0] == 0:
                continue
            prev_tile = tile_idx_arr - [1, 0, 0]
        elif assoc == "y":
            if tile_idx[:, 0] == 0:
                continue
            prev_tile = tile_idx_arr - [0, 1, 0]
        elif assoc == "z":
            if tile_idx[:, :, 0] == 0:
                continue
            prev_tile = tile_idx_arr - [0, 0, 1]

        # get surfaces for matching previous tile, and remove from dict as it will no longer
        # be needed
        tile_pair = (prev_tile, tile_idx)
        prev_surfaces = tile_edges.pop(tile_pair)
        # level_associations = []
        for level_num, ((label_array, label_num), prev_surface) in enumerate(
            zip(label_results, prev_surfaces)
        ):
            if assoc == "x":
                this_surface = label_array[0, :, :]
            elif assoc == "y":
                this_surface = label_array[:, 0, :]
            elif assoc == "z":
                this_surface = label_array[:, :, 0]

            joined_surfaces = cp.stack(
                [prev_surface, this_surface]
            )  # shape (2, y, z)
            joined_surfaces_flat = cp.reshape(joined_surfaces, (2, -1))
            unique_pairs = get_unique(joined_surfaces_flat)
            zero_mask = (unique_pairs == 0).any(axis=0)
            nonzero_pairs = unique_pairs[
                :, ~zero_mask
            ].T.get()  # shape (unique_nonzero_vals, 2)

            # find association pairs and record in bi-directional association map
            for assoc_pair in nonzero_pairs:
                # if (assoc_pair == 0).any():
                #     continue
                prev_key = (prev_tile, level_num, int(assoc_pair[0]))
                this_key = (tile_idx, level_num, int(assoc_pair[1]))
                assoc_map[this_key].add(prev_key)
                assoc_map[prev_key].add(this_key)
            # level_associations.append(unique_pairs)
        # # record associations
        # component_associations[tile_pair] = level_associations

    # record surfaces for following neighbours
    neighbour_surfaces_x, neighbour_surfaces_y, neighbour_surfaces_z = (
        [],
        [],
        [],
    )
    for label_array, label_num in label_results:
        neighbour_surfaces_x = label_array[-1, :, :]
        neighbour_surfaces_y = label_array[:, -1, :]
        neighbour_surfaces_z = label_array[:, :, -1]

    tile_edges[
        (tile_idx, tuple(tile_idx_arr + [1, 0, 0]))
    ] = neighbour_surfaces_x
    tile_edges[
        (tile_idx, tuple(tile_idx_arr + [0, 1, 0]))
    ] = neighbour_surfaces_y
    tile_edges[
        (tile_idx, tuple(tile_idx_arr + [0, 0, 1]))
    ] = neighbour_surfaces_z


def find_volume_components(
    volume_file: str,
    outfile: str,
    data_label: str,
):
    """
    Find connected components at various erosion levels in the given volume
    """

    # open file as HDF5
    logging.info(
        "Opening volume file %s with data label %s" % (volume_file, data_label)
    )
    h5array = read_h5(volume_file, data_label)

    # initialise tile association maps
    # component_associations maps from a tuple (prev_tile_idx, next_tile_idx) to a list over
    # erosion levels, each an array of shape (2, connection_pairs) representing components that
    # are connected between tiles.
    # assoc_map maps from a tuple (tile_idx, level, id) to a set of connected tiles
    # (other_tile_idx, level, other_id), as a bi-directional map of connections
    # tile_edges is a map from a tuple (prev_tile_idx, next_tile_idx) to a list over
    # erosion levels, each an array of shape (tile_size, tile_size) representing the surface of
    # tile prev_tile_idx that adjoins tile next_tile_idx
    # tile_components is a map from tile_idx to a list over erosion levels, each a dict mapping
    # from each label number to a tuple of (bounds, center, size, parent_num).  bounds and center
    # are defined within the tile, size is the number of covered voxels within the tile, and
    # parent_num is the component number in the previous erosion level within the tile (or None if
    # erosion level is zero).
    # component_associations = {}
    assoc_map = defaultdict(set)
    tile_edges = {}
    tile_components = {}

    # step over individual tiles and collect properties
    dims = np.array(h5array.shape[1:])
    tile_steps = np.ceil(dims / TILE_SIZE).astype("int")
    for tile_x in range(tile_steps[0]):
        for tile_y in range(tile_steps[1]):
            for tile_z in range(tile_steps[1]):
                tile_idx = (tile_x, tile_y, tile_z)

                # process tile
                process_tile(
                    tile_idx,
                    h5array,
                    assoc_map,
                    tile_edges,
                    tile_components,
                )

    # combine results
    find_combined_components(tile_components, assoc_map, tile_steps, outfile)


def find_combined_components(
    tile_components: Dict,
    assoc_map: defaultdict,
    tile_steps: np.ndarray,
    outfile: str,
):
    """
    Given a dictionary representing components within individual tiles, and associations between
    components in different tiles, find global components by combining associated components from
    different tiles and defining based on merged properties (eg size, center) in global coordinates.

    Save results in output directory

    :param Dict tile_components: Map from tile_idx to a list over erosion levels, each a dict mapping
        from each label number to a tuple of (bounds, center, size, parent_num).  bounds and center
        are defined within the tile, size is the number of covered voxels within the tile, and
        parent_num is the component number in the previous erosion level within the tile (or None if
        erosion level is zero).
    :param Dict component_associations: Map from a tuple (prev_tile_idx, next_tile_idx) to a list over
        erosion levels, each an array of shape (2, connection_pairs) representing components that
        are connected between tiles.
    :param np.ndarray tile_steps: Number of tiles, shape (x, y, z)
    :param str outfile: Output file to write global component results (as pickle)
    """
    # global_components is a list over erosion levels, each a dict mapping from global component id
    # to a tuple of (bounds, center, size, global_parent_num)
    global_components = [{}] * (EROSION_STEPS + 1)

    # global_id_map is a map from a tuple of (tile_idx, erosion_level, local_id) to global_id
    global_id_map = {}

    # first make bi-directional graph of connected components between tiles, so that when one
    # component is examined all connected components in neighbouring tiles can also be identified
    # component_connections is a map from a tuple (tile_idx, erosion_level, id) to a set of tuples
    # (tile_idx, erosion_level, id) of all connected neighbours.  this is defined bi-directionally

    # define next available global component ids for each level.  start ids at 1 (0 is background)
    next_global_ids = [1] * (EROSION_STEPS + 1)

    # step over tiles and local components, and map each to global components, including merging
    for tile_x in tile_steps[0]:
        for tile_y in tile_steps[1]:
            for tile_z in tile_steps[2]:
                tile_id = (tile_x, tile_y, tile_z)
                tile_level_components = tile_components[tile_id]
                for level_num, level_components in enumerate(
                    tile_level_components
                ):
                    level_global_components = global_components[level_num]
                    for (
                        label_number,
                        local_component_details,
                    ) in level_components.items():
                        # check if this component is already associated with a global component
                        component_key = (tile_id, level_num, label_number)
                        if component_key in global_id_map:
                            # tile component is associated with an existing global component, merge into
                            # the global component
                            global_id = global_id_map[component_key]
                            merged_global_component = merge_components(
                                local_component_details,
                                level_global_components[global_id],
                            )
                            # replace with merged global component
                            level_global_components[
                                global_id
                            ] = merged_global_component
                        else:
                            # this component is not associated with an existing global component.
                            # create a new global component and mark all connected components as
                            # associated
                            # find global component of parent of this component in the previous
                            # erosion level.  assume this is defined.
                            global_parent_id = None
                            (
                                bounds,
                                center,
                                size,
                                local_parent,
                            ) = local_component_details
                            if local_parent is not None:
                                parent_local_key = (
                                    tile_id,
                                    level_num - 1,
                                    local_parent,
                                )
                                global_parent_id = global_id_map[
                                    parent_local_key
                                ]
                            global_component_details = (
                                bounds,
                                center,
                                size,
                                global_parent_id,
                            )
                            global_id = next_global_ids[level_num]
                            next_global_ids[level_num] += 1
                            level_global_components[
                                global_id
                            ] = global_component_details

                            # find associated components and mark as associated with this global
                            # component
                            found_keys = set()
                            find_all_associations(
                                assoc_map, component_key, found_keys
                            )
                            for assoc_key in found_keys:
                                global_id_map[assoc_key] = global_id
                        # level components
                    # erosion level
                # z
            # y
        # x
    # write out results to file
    components_per_level = [
        len(level_components) for level_components in global_components
    ]
    logging.info(
        "Found components per erosion level: %s" % components_per_level
    )

    logging.info("Writing output to %s")
    write_data = global_components
    with open(outfile, "wb") as out_handle:
        pickle.dump(write_data, out_handle)


def merge_components(local_component, target_component):
    """
    Find resulting component from merging the first (local) component into the second.  The resulting
    component will maintain the parent identifier of the target component.
    """
    local_bounds, local_center, local_size, local_parent = local_component
    target_bounds, target_center, target_size, target_parent = target_component

    merged_bounds = [
        min(local_bounds[0], target_bounds[0]),
        max(local_bounds[1], target_bounds[1]),
        min(local_bounds[2], target_bounds[2]),
        max(local_bounds[3], target_bounds[3]),
        min(local_bounds[4], target_bounds[4]),
        max(local_bounds[5], target_bounds[5]),
    ]

    merged_size = local_size + target_size

    # use weighted averaging to find center.  the center point is not guaranteed to occur at a
    # position containing the component (eg if it is "C" shape)
    merged_center = (
        local_center * local_size + target_center * target_size
    ) / merged_size

    return merged_bounds, merged_center, merged_size, target_parent


def find_all_associations(
    assoc_map: defaultdict, component_key: Tuple, found_keys: Set
):
    """
    Find all associated components from the given component key and populate found_keys.  Recursive
    function.
    """
    out_assocs = assoc_map[component_key]
    for assoc in out_assocs:
        if assoc not in found_keys:
            found_keys.add(assoc)
            find_all_associations(assoc_map, assoc, found_keys)


def init_logging():
    """ Initialise logging """
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v", "--volume_file", help="Volume file to read (HDF5)", required=True
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output file",
        default="graph_output.pkl",
        required=False,
    )
    parser.add_argument(
        "-l",
        "--data_label",
        help="Data array label in volume file",
        default="data",
        required=False,
    )

    args = parser.parse_args()

    logging.info("Find graph of components in volume")

    find_volume_components(
        args.volume_file,
        args.output,
        args.data_label,
    )


if __name__ == "__main__":
    main()
