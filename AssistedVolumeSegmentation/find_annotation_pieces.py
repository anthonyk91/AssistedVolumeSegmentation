# create map of annotation tiles from coverage map

import argparse
import logging

import nibabel as nib
import nrrd
import numpy as np

from AssistedVolumeSegmentation.common import (
    annotation_field_map,
    get_full_path,
    init_logging,
    load_config,
    read_segment_file,
)


def find_annotation_pieces(config, subdir_num, write_format="nrrd"):
    coverage_file = get_full_path(config, subdir_num, "overview_coverage")
    output_file = get_full_path(config, subdir_num, "pieces_overview")

    seg_map, seg_scales = read_segment_file(coverage_file)

    # resample reduced map so that each voxel represents one annotation block.  pad to prevent
    # clipping edge values (scipy.ndarray.zoom rounds dimensions which can lead to clipping)
    annotation_size = np.array(config["annotation_size"])
    seg_dims = np.array(seg_map.shape)
    seg_reduce = seg_scales / annotation_size

    # reduce coverage block to a map of annotation pieces.  step over each annotation piece
    # and take the max of the corresponding region
    annotation_map_size = np.ceil(seg_reduce * seg_dims).astype("int")

    logging.info(
        "Reducing segmentations from %s to %s"
        % (seg_map.shape, annotation_map_size)
    )
    # todo: check choice of boundaries is correct, prev used rounding but this lead to creating
    #       pieces that were past the edge of the segmentation map (?)
    annot_seg_scale = annotation_size / seg_scales
    x_boundaries = np.floor(
        np.arange(annotation_map_size[0] + 1) * annot_seg_scale[0]
    ).astype("int")
    y_boundaries = np.floor(
        np.arange(annotation_map_size[1] + 1) * annot_seg_scale[1]
    ).astype("int")
    z_boundaries = np.floor(
        np.arange(annotation_map_size[2] + 1) * annot_seg_scale[2]
    ).astype("int")

    x_reduced = np.stack(
        [
            seg_map[xmin:xmax].max(axis=0)
            for xmin, xmax in zip(x_boundaries[:-1], x_boundaries[1:])
        ]
    )  # (annot_x, seg_y, seg_z)
    y_reduced = np.stack(
        [
            x_reduced[:, ymin:ymax].max(axis=1)
            for ymin, ymax in zip(y_boundaries[:-1], y_boundaries[1:])
        ],
        axis=1,
    )  # (annot_x, annot_y, seg_z)
    annot_map = np.stack(
        [
            y_reduced[:, :, zmin:zmax].max(axis=2)
            for zmin, zmax in zip(z_boundaries[:-1], z_boundaries[1:])
        ],
        axis=2,
    )  # (annot_x, annot_y, seg_z)

    logging.info(
        "Number of occupied annotation pieces %d from %d, annotation pieces shape %s, scales %s"
        % (
            np.sum(annot_map),
            np.prod(annot_map.shape),
            annot_map.shape,
            annotation_size,
        )
    )

    # write annotation piece occupancy into map
    if write_format == "nrrd":
        # create NRRD output file
        scales = np.zeros((3, 3), dtype="float")
        np.fill_diagonal(scales, annotation_size * [-1, -1, 1])
        index_offset = np.zeros((3,), dtype="int")
        space_origin = (index_offset * annotation_size).astype("float")
        header = {
            "space": "left-posterior-superior",  # copied from Slicer example
            "kinds": [
                "domain",
                "domain",
                "domain",
            ],  # copied from Slicer example
            "space directions": scales,
            "space origin": space_origin,
        }

        logging.info("Writing NRRD output to %s" % output_file)
        nrrd.write(
            output_file,
            annot_map,
            header,
            custom_field_map=annotation_field_map,
        )
    elif write_format == "nifti":
        # create Nifti1 format object from array
        # define affine transform matrix representing scaling transform
        aff_matrix = np.identity(4, dtype="float")
        np.fill_diagonal(aff_matrix, annotation_size)

        nifti_object = nib.Nifti1Image(annot_map, aff_matrix)
        logging.info("Writing Nifti v1 output to %s" % output_file)
        nib.save(nifti_object, output_file)


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config_file", help="Project config file", required=True
    )
    parser.add_argument(
        "-s", "--subdir", help="Data subdirectory number", required=True
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    if not isinstance(args.subdir, str) or not args.subdir.isnumeric():
        raise RuntimeError("Subdir should be a number")
    subdir_num = int(args.subdir)

    # read coverage map and produce output of map of annotation pieces
    find_annotation_pieces(config, subdir_num)


if __name__ == "__main__":
    main()
