# get overview of data
# read data from given files, and produce a reduced image of each data file

import argparse
import h5py
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


def reduce_source_data(config, bound_size, subdir_num, launch_editor):
    """
    Read source data, from either a stack of tiff files from the given dir or an HDF5 data file, and resize the 
    source volume data to the given size.

    :param config: Configuration map
    :param bound_size: Maximum size of the volume, as (height, width, layers)
    :param subdir_num: Number of subdirectory data to read
    :param launch_editor: If true, launch slicer editor using generated data
    :return: Array of source images reduced to given size
    """
    source_format = config["source_data_format"][subdir_num]
    read_tiff_stack = (source_format == "tiff-stack")

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
        "reading from source %s, format %s, writing to %s, %s, bound size %s"
        % (data_path, source_format, write_data_path, write_coverage_path, bound_size)
    )

    if read_tiff_stack:
        # get the list of files
        file_list = get_file_list(data_path)
        input_stacks = len(file_list)

        # read the first image to get dimensions.  assume images are the same size
        first_img = Image.open(file_list[0])
        input_w, input_h = first_img.size  # (width, height)
        input_dims = (input_h, input_w)
        input_count = input_stacks
        iterate_dim = 2
    else:
        # open as hdf5 file
        h5_file = h5py.File(data_path, "r")
        h5_data_key = config["source_hdf5_dataset_name"]
        h5_data = h5_file[h5_data_key]
        input_size = h5_data.shape
        # iterate over the first dimension (x), which is different to when reading from image stacks.
        # it is much faster to slice on the initial dimension(s) than on later ones (eg z).
        input_count = input_size[0]
        input_dims = input_size[1:3]
        iterate_dim = 0

    # find the downsample factor to fit within the given bounds.  only allow reducing size
    ratios = [min(1.0, x / y) for x, y in zip(bound_size, input_size)]
    reduce_ratio = np.min(ratios)

    # find target image size
    target_imgs = math.ceil(input_count * reduce_ratio)
    target_dims = tuple([math.ceil(x * reduce_ratio) for x in input_dims])

    def find_frame_score(input_idx, div_steps, out_slice):
        lower_bound = max(input_idx, div_steps[out_slice])
        upper_bound = min(input_idx + 1, div_steps[out_slice + 1])
        return upper_bound - lower_bound

    # read slices in turn
    logging.info(
        "found %d images(slices), input dim (w %d, h %d), target dim %s, %d slices"
        % (input_count, input_dims[1], input_dims[0], target_dims, target_imgs)
    )
    input_slices = {}
    division_steps = np.arange(target_imgs + 1) * (1.0 / reduce_ratio)
    output_slice = 0
    range_start = 0
    range_end = math.ceil(division_steps[1])
    output_slices = []
    for count in tqdm(range(input_count)):
        if read_tiff_stack:
            this_file = file_list[count]
            this_img = Image.open(this_file)
        else:
            this_img = Image.fromarray(h5_data[count])

        resized_img = this_img.resize(target_dims, Image.NEAREST)
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
            if output_slice < target_imgs:
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
    output_array = np.stack(output_slices, axis=iterate_dim)
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

    # read source data from tiff files in given directory or an HDF5 file, and downsample to given size
    reduce_source_data(
        config,
        overview_bound_size,
        subdir_num,
        args.launch,
    )

if __name__ == "__main__":
    init_logging()
    main()
