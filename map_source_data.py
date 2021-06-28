# read all source data within tiles and store each as an individual .nii.gz file

import argparse
import logging
import os

import nibabel as nib
import numpy as np

from common import (
    completed_value,
    data_suffix,
    get_annot_map,
    get_full_path,
    get_source_tile_data,
    get_source_tile_stored,
    init_logging,
    load_config,
)


def map_source_data(config, subdir_num):
    """
    Produce source data files for each tile in the specified source folder under the project
    path

    :param config:  Config dictionary
    :param subdir_num: Subdir number
    """
    # get annotation map of covered tiles
    annot_map, annot_header, _ = get_annot_map(config, subdir_num)
    annot_tiles = np.stack(
        np.where(annot_map == completed_value), axis=1
    )  # (num_pieces, 3)

    # check each annotation tile if it exists in stored source data,
    # and copy if not
    for this_tile in annot_tiles:
        if get_source_tile_stored(
            config, subdir_num, this_tile, check_only=True
        ):
            logging.info("Tile %s already stored, skipping" % this_tile)
            continue

        # read source data
        tile_data = get_source_tile_data(config, subdir_num, this_tile)

        # write to stored tile
        indices_str = [str(x) for x in this_tile.tolist()]
        data_path = get_full_path(config, subdir_num, "source_piece_path")
        filename = "_".join(indices_str) + data_suffix
        os.makedirs(data_path, exist_ok=True)
        source_file = os.path.join(data_path, filename)

        if os.path.exists(source_file):
            raise RuntimeError(
                "Unexpected, file %s already exists" % source_file
            )

        aff_matrix = np.identity(4, dtype="float")
        np.fill_diagonal(aff_matrix, config["annotation_size"])

        nifti_object = nib.Nifti1Image(tile_data, aff_matrix)
        logging.info(
            "Writing tile %s as Nifti v1 output to %s"
            % (this_tile, source_file)
        )
        nib.save(nifti_object, source_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", help="Project config file", required=True
    )
    parser.add_argument(
        "--subdir", help="Data subdirectory number", required=True
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    logging.info("Mapping source data")

    if not isinstance(args.subdir, str) or not args.subdir.isnumeric():
        raise RuntimeError("Subdir should be a number")
    subdir_num = int(args.subdir)

    map_source_data(
        config,
        subdir_num,
    )


if __name__ == "__main__":
    init_logging()
    main()
