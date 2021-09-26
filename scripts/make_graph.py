# script to make graph of connected components in a volume
import logging
from typing import Union

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


def find_components(data_array_input: ArrayTypes) -> Tuple[ArrayTypes, int]:
    """
    Find connected components of the given array.  If it is a large array (likely a h5 Dataset), find
    connected components using individual tiles and write the output to a data file.  If it is a small
    array, find the connected components directly and return.
    """
    if isinstance(data_array_input, VolumeFile):
        data_array = read_h5(*data_array_input)
    else:
        data_array = data_array_input

    if (data_array.shape < limit_array_size).all():
        # find components directly
        cparray = cp.asarray(data_array)
        return cpnd.label(cparray)

    # todo: change return type for h5 based result, so able to be handled
    # find connected components using separate tiles
    # or consider trying running label with output=h5_data directly? not sure how that will work


def process_tile(
    tile_idx: np.ndarray,
    h5array: h5py.Dataset,
    global_components: Dict,
    tile_component_edges: Dict,
):
    # find content for given tile.  extend boundary past reference region in order to
    # allow accurate erosion result within the reference region.  assume that erosion
    # uses a filter of size 3x3, which requires 1 pixel of surrounding region for each
    # erosion step
    reference_start = tile_idx * TILE_SIZE
    reference_max = reference_start + TILE_SIZE

    extended_start = reference_start - EROSION_STEPS
    extended_max = reference_max + EROSION_STEPS

    # get extended region from data.  to handle cases at the edges where the extended
    # region is not populated by data, create a zero-filled array and then copy the
    # defined region
    extended_region = cp.zeros(TILE_SIZE + 2 * EROSION_STEPS, dtype=bool)
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
    insert_end = np.minimum(-(extended_end - source_size), 0)
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
    tile_component_details = {}
    for label_array, label_count in label_results:
        for label_num in range(label_count):
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
            size = int(cp.sum(value_mask))
            tile_component_details[label_num] = (bounds, size)

    # connect tile components with global components by comparing with preceding neighbours
    if tile_idx[0] > 0:
        prev_tile = tile_idx - [1, 0, 0]
        prev_surface = tile_component_edges[(prev_tile, tile_idx)]

    # record surfaces for following neighbours
    # todo: update to use global values
    tile_component_edges[(tile_idx, tile_idx + [1, 0, 0])] = label_array
    # find bounding surfaces from label array to connect components in neighbouring tiles

    # global_components: Dict,
    # tile_component_edges: Dict,

    # todo: add collection of surfaces of previous tiles as input, and match current tile edge to continue existing
    #      components


def find_volume_components(
    volume_file: str,
    outdir: str,
    data_label: str,
):
    # open file as HDF5
    logging.info(
        "Opening volume file %s with data label %s" % (volume_file, data_label)
    )
    h5array = read_h5(volume_file, data_label)

    # step over individual tiles and collect properties
    dims = np.array(h5array.shape[1:])
    tile_steps = np.ceil(dims / TILE_SIZE).astype("int")
    for tile_x in range(tile_steps[0]):
        for tile_y in range(tile_steps[1]):
            for tile_z in range(tile_steps[1]):
                tile_idx = np.array([tile_x, tile_y, tile_z])

                # process tile
                results = process_tile(
                    tile_idx,
                    h5array,
                )
                # record results

    # combine results

    # find connected components in initial volume

    # perform morphological erosion


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
        help="Output directory",
        default="graph_output",
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
