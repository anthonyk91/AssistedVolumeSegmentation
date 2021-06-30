# get overview of data
# read data from given files, and produce a reduced image of each data file

import argparse
import logging
import math
import os
import shutil
import subprocess

import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

from common import (
    default_seg_file,
    get_file_list,
    get_full_path,
    get_source_data_path,
    init_logging,
    load_config,
    overview_bound_size,
)


def reduce_tiff_stack(config, bound_size, subdir_num, launch_editor):
    """
    Read a stack of tiff files from the given dir, and resize the stack to the given size.

    :param config: Configuration map
    :param bound_size: Maximum size of the stack, as (height, width, layers)
    :param subdir_num: Number of subdirectory data to read
    :param launch_editor: If true, launch slicer editor using generated data
    :return: Array of source images reduced to given size
    """
    # select source data path according to index in config file. assume source data path is an array
    data_path = get_source_data_path(config, subdir_num)
    overviews_path = get_full_path(config, subdir_num, "overviews_path")
    write_data_path = get_full_path(
        config, subdir_num, "overview_reduced_data"
    )
    write_coverage_path = get_full_path(
        config, subdir_num, "overview_coverage"
    )
    logging.info(
        "reading from stack dir %s, writing to %s, %s, bound size %s"
        % (data_path, write_data_path, write_coverage_path, bound_size)
    )

    # get the list of files
    file_list = get_file_list(data_path)
    input_stacks = len(file_list)

    # read the first image to get dimensions.  assume images are the same size
    first_img = Image.open(file_list[0])
    input_w, input_h = first_img.size  # (width, height)
    input_size = (input_h, input_w, input_stacks)

    # find the downsample factor to fit within the given bounds.  only allow reducing size
    ratios = [min(1.0, x / y) for x, y in zip(bound_size, input_size)]
    reduce_ratio = np.min(ratios)

    # find target image size
    target_size = tuple([math.ceil(x * reduce_ratio) for x in input_size])

    def find_frame_score(input_idx, div_steps, out_slice):
        lower_bound = max(input_idx, div_steps[out_slice])
        upper_bound = min(input_idx + 1, div_steps[out_slice + 1])
        return upper_bound - lower_bound

    # read slices in turn
    logging.info(
        "found %d stacks, input dim (w %d, h %d), target shape %s"
        % (len(file_list), input_w, input_h, target_size)
    )
    input_slices = {}
    division_steps = np.arange(target_size[2] + 1) * (1.0 / reduce_ratio)
    output_slice = 0
    range_start = 0
    range_end = math.ceil(division_steps[1])
    output_slices = []
    for count, this_file in tqdm(enumerate(file_list), total=len(file_list)):
        this_img = Image.open(this_file)

        resized_img = this_img.resize(target_size[:2], Image.NEAREST)
        input_slices[count] = resized_img

        # check if we have enough slices to output the next slice
        if count >= range_end - 1:
            slices = np.stack(
                [
                    np.array(input_slices[x])
                    for x in range(range_start, range_end)
                ]
            )  # (slices, width, height)
            slice_weights = np.array(
                [
                    find_frame_score(x, division_steps, output_slice)
                    for x in range(range_start, range_end)
                ]
            )  # (slices,)

            reduced_slice = (slices * slice_weights[:, None, None]).sum(
                axis=0
            ) / slice_weights.sum()
            output_slices.append(reduced_slice)

            # update output slice number and input slice range
            output_slice += 1
            if output_slice < target_size[2]:
                range_start = math.floor(division_steps[output_slice])
                range_end = math.ceil(division_steps[output_slice + 1])

                # remove stored input slices outside of range
                remove_input_frames = [
                    x
                    for x in input_slices.keys()
                    if x < range_start or x > range_end
                ]
                for x in remove_input_frames:
                    input_slices.pop(x)

    # write output slices
    # create Nifti1 format object from array
    # define affine transform matrix representing scaling transform
    aff_matrix = np.identity(4, dtype="float")
    np.fill_diagonal(aff_matrix, [1.0 / reduce_ratio] * 3)
    output_array = np.stack(output_slices, axis=2)
    nifti_object = nib.Nifti1Image(output_array, aff_matrix)
    os.makedirs(overviews_path, exist_ok=True)
    nib.save(nifti_object, write_data_path)

    # create sample coverage segmentation file
    if not os.path.exists(write_coverage_path):
        script_path = os.path.dirname(os.path.abspath(__file__))
        default_file = os.path.join(script_path, default_seg_file)
        shutil.copyfile(default_file, write_coverage_path)
        logging.info(
            "Copying initial coverage annotation from %s to %s"
            % (default_file, write_coverage_path)
        )

    logging.info(
        "Created overview data in: %s , and empty coverage annotation in: %s"
        % (write_data_path, write_coverage_path)
    )
    if launch_editor:
        args = [config["slicer_path"], write_data_path, write_coverage_path]
        logging.info("Launching Slicer editor with arguments: %s" % args)
        subprocess.run(" ".join(args), shell=True)
    else:
        logging.info(
            "You can open this in Slicer manually, or launch it with: %s %s %s"
            % (config["slicer_path"], write_data_path, write_coverage_path)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", help="Project config file", required=True
    )
    parser.add_argument(
        "--subdir", help="Data subdirectory number", required=True
    )
    parser.add_argument(
        "--launch", help="Launch Slicer to edit piece", action="store_true"
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    if not isinstance(args.subdir, str) or not args.subdir.isnumeric():
        raise RuntimeError("Subdir should be a number")
    subdir_num = int(args.subdir)

    # read all tiff files from given directory and downsample to given size
    reduce_tiff_stack(
        config,
        overview_bound_size,
        subdir_num,
        args.launch,
    )


if __name__ == "__main__":
    init_logging()
    main()
