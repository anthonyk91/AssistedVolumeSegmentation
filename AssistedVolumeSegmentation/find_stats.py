# find mean/std stats from source data
# also include methods to read from generated data and create an annotation array file

import argparse
import logging

import numpy as np
import tqdm

from AssistedVolumeSegmentation.common import (
    completed_value,
    get_annot_map,
    get_source_data,
    init_logging,
    load_config,
)


def find_data_stats(
    config,
    subdir_str,
):
    subdir_num = int(subdir_str)

    annotation_scale = np.array(config["annotation_size"])
    annot_map, _, _ = get_annot_map(config, subdir_num)

    annot_map_mask = annot_map.astype("bool")
    annot_map_indices = np.stack(
        np.where(annot_map_mask == completed_value), axis=1
    )  # (n,3)

    # choose a number of pieces at random
    test_pieces = 10
    crop_size = annotation_scale
    means = []
    vars = []
    mins = []
    maxs = []
    for _ in tqdm.tqdm(range(test_pieces)):
        chosen_indices = annot_map_indices[
            np.random.choice(len(annot_map_indices))
        ]
        data_crop_origin = chosen_indices * annotation_scale
        source_data = get_source_data(
            config, subdir_num, data_crop_origin, crop_size
        )
        means.append(source_data.mean())
        vars.append(source_data.var())
        mins.append(source_data.min())
        maxs.append(source_data.max())

    all_mean = np.mean(means)
    all_var = np.mean(vars)
    all_std = np.sqrt(all_var)
    all_min = np.min(mins)
    all_max = np.max(maxs)

    logging.info(
        "Sampled %d pieces, mean %f std %f min %d max %d"
        % (test_pieces, all_mean, all_std, all_min, all_max)
    )


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", help="Project config file", required=True
    )
    parser.add_argument(
        "--subdir", help="Data subdirectory number", required=False
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    logging.info("Find sample data stats")

    find_data_stats(
        config,
        args.subdir,
    )


if __name__ == "__main__":
    main()
