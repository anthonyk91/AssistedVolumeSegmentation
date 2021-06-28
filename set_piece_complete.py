# script to record that a piece that is in progress should be stored in the completed annotations
import argparse
import logging
import os
import shutil

import nibabel as nib
import numpy as np

from common import (
    annot_suffix_segments,
    assign_cropped_region,
    data_suffix,
    get_annotation,
    get_full_path,
    get_project_data,
    init_logging,
    load_config,
    write_annot_file,
)


def move_progress_piece(
    config,
    subdir_num,
    piece_index_list,
):
    piece_index = np.array([int(x) for x in piece_index_list])

    # check if piece already completed
    index_name = "_".join([str(x) for x in piece_index])
    piece_file = index_name + annot_suffix_segments
    completed_piece_path = get_full_path(
        config, subdir_num, "completed_piece_path"
    )
    completed_file_full = os.path.join(completed_piece_path, piece_file)
    if os.path.exists(completed_file_full):
        raise RuntimeError(
            "Piece with index %s already exists in completed piece path"
            % piece_index
        )

    # check if in progress piece is present
    inprogress_piece_path = get_full_path(
        config, subdir_num, "inprogress_piece_path"
    )
    inprogress_annotfile_full = os.path.join(inprogress_piece_path, piece_file)
    if not os.path.exists(inprogress_annotfile_full):
        raise RuntimeError(
            "Piece with index %s not present in in-progress piece path"
            % piece_index
        )

    # open project data file to get data offset
    project_data_file = get_full_path(config, subdir_num, "project_data_file")
    stored_data = get_project_data(project_data_file)
    this_tile_name = str(piece_index)

    # get tile offset, and crop to maintain only the region specific for this tile (exclude
    # repeated boundary)
    if (
        "offsets" not in stored_data
        or this_tile_name not in stored_data["offsets"]
    ):
        raise RuntimeError(
            "Could not get offset for tile %s from project data file %s"
            % (this_tile_name, project_data_file)
        )
    annot_offset = np.array(stored_data["offsets"][this_tile_name])

    # read annotations and create cropped tile (without overlap)
    # todo: change to use binary labelmap format instead of segmentation as default.
    #       save resulting output in labelmap format.  this requires changing the training
    #       data loader so that it reads segments in labelmap format rather than as
    #       segmentation files.  currently the output is segmentation data but the header
    #       copied from the input doesn't match as the default save format is now labelmap.
    #       One option is to simply write out here in labelmap format, as later the dataloader
    #       will automatically convert labelmap to segmentation.
    annot_data, annot_header, data_suboffset = get_annotation(
        piece_index, inprogress_piece_path
    )
    if len(annot_data.shape) == 4:
        num_segments = annot_data.shape[0]
    else:
        raise RuntimeError(
            "Unexpected, no annotated segments in piece with index %s"
            % piece_index
        )

    annot_full_offset = (annot_offset + data_suboffset).astype("int")
    annot_size = np.array(annot_data.shape[-3:])
    print(
        "annot_full_offset",
        annot_full_offset,
        "annot_size",
        annot_size,
        "annot_offset",
        annot_offset,
    )
    # annot_max = annot_full_offset + annot_size
    tile_size = np.array(config["annotation_size"])
    piece_offset = piece_index * tile_size
    piece_max = piece_offset + tile_size

    cropped_annot = np.zeros(
        (num_segments,) + tuple(config["annotation_size"]),
        dtype=annot_data.dtype,
    )
    tile_crop_min = np.clip(annot_full_offset - piece_offset, 0, tile_size - 1)
    # tile_crop_max = np.clip(annot_max - piece_offset, 0, tile_size)
    data_crop_min = np.clip(
        piece_offset - annot_full_offset, 0, annot_size - 1
    )
    # data_crop_max = np.clip(piece_max - annot_full_offset, 0, annot_size)

    print("annot shape", annot_data.shape, "data crop min", data_crop_min)
    assign_cropped_region(
        annot_data, data_crop_min, cropped_annot, tile_crop_min, tile_size
    )

    # read data tile and remove any overlap
    piece_data_file = index_name + data_suffix
    inprogress_datafile_full = os.path.join(
        inprogress_piece_path, piece_data_file
    )
    inprogress_data = nib.load(inprogress_datafile_full)
    data_size = np.array(inprogress_data.shape)
    data_crop_min = np.clip(piece_offset - annot_offset, 0, data_size - 1)
    data_crop_max = np.clip(piece_max - annot_offset, 0, data_size)
    # just take data array from previous data, and crop. using slicer will
    # maintain the crop offset in the affine transform
    cropped_data = inprogress_data.dataobj[
        data_crop_min[0] : data_crop_max[0],
        data_crop_min[1] : data_crop_max[1],
        data_crop_min[2] : data_crop_max[2],
    ]
    affine = np.identity(4, dtype="float")
    cropped_data_obj = nib.Nifti1Image(cropped_data, affine)

    # save annotation to completed path
    full_extent = tuple(tile_size - 1)
    # update all extent fields of segment to full tile
    extent_tags = [x for x in annot_header.keys() if x[-7:] == "_Extent"]
    for t in extent_tags:
        annot_header[t] = "0 %d 0 %d 0 %d" % full_extent
    # remove data sub-offset field as annotation now is the full extent of the tile
    annot_header["space origin"] = np.zeros((3,), dtype="float")

    # todo: fix output.  currently it seems to fail for the annotated file to be
    #   loaded into slicer.  check what differences there are with the annotation
    #   files generated with get_annotation_piece
    # scales = np.identity(3, dtype="float")
    scales = annot_header["space directions"][-3:]
    write_annot_file(
        index_name,
        completed_piece_path,
        cropped_annot,
        annot_header,
        scales,
    )

    # save data to completed path
    completed_data_file = os.path.join(completed_piece_path, piece_data_file)
    logging.info(
        "Writing completed piece data output to %s" % completed_data_file
    )
    nib.save(cropped_data_obj, completed_data_file)

    # move piece from inprogress to be removed
    removed_piece_path = get_full_path(
        config, subdir_num, "removed_piece_path"
    )
    os.makedirs(removed_piece_path, exist_ok=True)
    logging.info("Moving removed inprogress files to %s" % removed_piece_path)
    shutil.move(inprogress_annotfile_full, removed_piece_path)
    shutil.move(inprogress_datafile_full, removed_piece_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        help="Specified index values (x,y,z list) for annotation piece",
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--config_file", help="Project config file", required=True
    )
    parser.add_argument(
        "--subdir", help="Data subdirectory number", required=True
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    if not isinstance(args.subdir, str) or not args.subdir.isnumeric():
        raise RuntimeError("Subdir should be a number")
    subdir_num = int(args.subdir)

    logging.info("Moving a completed annotation piece")

    move_progress_piece(
        config,
        subdir_num,
        args.index,
    )


if __name__ == "__main__":
    init_logging()
    main()
