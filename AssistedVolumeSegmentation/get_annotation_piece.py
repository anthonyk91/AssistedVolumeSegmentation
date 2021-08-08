# create a new annotation array file from the data
# also include methods to read from generated data and create an annotation array file

import argparse
import glob
import json
import logging
import os
import pickle
import random
import subprocess
from typing import Any, Dict, List, Optional

import nibabel as nib
import nrrd
import numpy as np
from filelock import FileLock

from AssistedVolumeSegmentation.common import (
    annot_suffix_labelmap,
    annot_suffix_segments,
    annotation_field_map,
    check_index,
    check_index_str,
    completed_value,
    data_suffix,
    flat_to_indexed,
    get_annot_map,
    get_annotation,
    get_completed_map,
    get_full_path,
    get_project_data,
    get_source_data,
    get_tiles_of_interest,
    indexed_to_flat,
    init_logging,
    load_config,
    piece_overlap,
    segment_properties,
    write_annot_file,
)

sample_tags = "TerminologyEntry:Segmentation category and type - 3D Slicer General Anatomy list~SRT^T-D0050^Tissue~SRT^T-D0050^Tissue~^^~Anatomic codes - DICOM master list~^^~^^|"
annot_dtype = "uint8"


def choose_annotation_piece(annot_map, completed_map, in_progress_map):
    """
    From the map of available and completed pieces, select a piece to be annotated next

    :param annot_map: Overview map of covered pieces
    :param completed_map: Map of completed pieces
    :param in_progress_map: Map of in progress pieces
    :return: Array of size (3,) of index of chosen piece
    """

    # use random selection from incomplete pieces
    if np.max(annot_map) > completed_value:
        # just squash to 1/0 values
        logging.warning("Unexpected, annotation map contains values >1")
        annot_map = np.clip(annot_map, 0, completed_value)

    incomplete_pieces = annot_map - completed_map - in_progress_map
    logging.info("%d incomplete pieces" % incomplete_pieces.sum())

    # todo: add more sophisticated selection, such as encouraging groups (eg 2x2?) which allow more
    #       offset training piece selection. (alternatively just use larger annotation pieces? but
    #       that may not be so suitable for the annotater)

    incomplete_mask = incomplete_pieces.astype("bool")
    incomplete_indices = np.stack(
        np.where(incomplete_mask == completed_value), axis=1
    )  # (n,3)

    return incomplete_indices[np.random.choice(len(incomplete_indices))]


def strip_header(header, piece_index):
    # get all segment properties from header, and map to a new segment id
    num_segments = header["sizes"][0]
    seg_fields_list = []
    for file_seg_id in range(num_segments):
        existing_fields = [
            "Segment%d_%s" % (file_seg_id, x) for x in segment_properties
        ]
        existing_values = [header[x] for x in existing_fields]
        existing_segment_fields = dict(
            zip(segment_properties, existing_values)
        )

        # record the segment that this is based on, including id and source file/index
        existing_segment_fields["referring_segment"] = file_seg_id
        existing_segment_fields["referring_index"] = piece_index
        seg_fields_list.append(existing_segment_fields)

    return seg_fields_list


def get_neighbour_annotation(
    annot_index,
    base_position,
    data_crop_min,
    data_crop_max,
    annotation_scale,
    d_array,
    completed_path,
):
    # get neighbouring completed annotation from file and return cropped section
    neighbour_index = annot_index + d_array
    (
        neighbour_data,
        neighbour_header,
        data_suboffset_float,
    ) = get_annotation(neighbour_index, completed_path, segment_format=True)

    seg_fields_list = strip_header(neighbour_header, neighbour_index)

    print(
        "neighbour_index",
        neighbour_index,
        "annotation_scale",
        annotation_scale,
        "data_suboffset_float",
        data_suboffset_float,
        "base_position",
        base_position,
    )
    data_offset = (
        neighbour_index * annotation_scale + data_suboffset_float
    ).astype("int")

    neighbour_sections = neighbour_data.shape[0]
    if neighbour_sections == 0:
        return None

    # get bounding section in source coordinates according to relationship from base index
    crop_min_source = np.zeros((3,), dtype="int")
    crop_max_source = np.zeros((3,), dtype="int")
    for idx in range(3):
        if d_array[idx] == -1:
            crop_min_source[idx] = base_position[idx] - piece_overlap[idx]
            crop_max_source[idx] = crop_min_source[idx] + piece_overlap[idx]
        elif d_array[idx] == 0:
            crop_min_source[idx] = base_position[idx]
            crop_max_source[idx] = crop_min_source[idx] + annotation_scale[idx]
        elif d_array[idx] == 1:
            crop_min_source[idx] = base_position[idx] + annotation_scale[idx]
            crop_max_source[idx] = crop_min_source[idx] + piece_overlap[idx]

    # fill overlap region with data
    data_crop_size = data_crop_max - data_crop_min
    neighbour_annot_size = (neighbour_sections,) + tuple(data_crop_size)
    neighbour_annot = np.zeros(
        neighbour_annot_size, dtype=neighbour_data.dtype
    )
    print("creating annotation with size", neighbour_annot_size)

    # assume data offset is in absolute coordinates over the source space
    neighbour_data_dim = np.array(neighbour_data.shape)[1:]
    print(
        "neighbour",
        d_array,
        "crop min",
        crop_min_source,
        "data_offset",
        data_offset,
        "neigh dim",
        neighbour_data_dim,
    )
    neighbour_min = np.clip(
        crop_min_source - data_offset, 0, neighbour_data_dim
    )
    neighbour_max = np.clip(
        crop_max_source - data_offset, 0, neighbour_data_dim
    )
    overlap_min = np.clip(
        data_offset + neighbour_min - crop_min_source, 0, data_crop_size
    )
    overlap_max = np.clip(
        data_offset + neighbour_max - crop_min_source, 0, data_crop_size
    )
    logging.info(
        "Cropping neighbour annotation, piece offset %s, data crop range %s %s size %s, neighbour origin %s range %s %s data range %s %s"
        % (
            d_array,
            data_crop_min,
            data_crop_max,
            data_crop_size,
            data_offset,
            neighbour_min,
            neighbour_max,
            overlap_min,
            overlap_max,
        )
    )
    print(
        "neigh annot",
        neighbour_annot.shape,
        "neigh data",
        neighbour_data.shape,
    )
    neighbour_annot[
        :,
        overlap_min[0] : overlap_max[0],
        overlap_min[1] : overlap_max[1],
        overlap_min[2] : overlap_max[2],
    ] = neighbour_data[
        :,
        neighbour_min[0] : neighbour_max[0],
        neighbour_min[1] : neighbour_max[1],
        neighbour_min[2] : neighbour_max[2],
    ]

    # find annotations that exist within overlap region
    populated = np.max(neighbour_annot, axis=(1, 2, 3)) > 0  # (num_segs,)
    if np.sum(populated) == 0:
        return None

    valid_annot = neighbour_annot[populated]
    valid_fields = [x for x, y in zip(seg_fields_list, populated) if y]

    return valid_annot, valid_fields


def find_generated_pieces(
    config: Dict[str, Any],
    fold_number: int,
):
    """
    Get a list of piece tile indices that have been generated
    """
    generator_path = get_full_path(config, None, "generator_output_path")
    generator_files_path = os.path.join(
        generator_path, "fold_%d" % fold_number
    )
    generated_glob = "instseg*.pickle"
    generated_list = glob.glob(
        os.path.join(generator_files_path, generated_glob)
    )

    found_indices = []
    for f in generated_list:
        file = os.path.split(f)[-1]
        filebase, _ = os.path.splitext(file)
        flat_index = int(filebase.split("_")[-1])
        subdir_and_index = flat_to_indexed(flat_index, config)
        found_indices.append(subdir_and_index)

    if len(found_indices) == 0:
        raise RuntimeError(
            "Could not find any generated tiles from path %s"
            % generator_files_path
        )

    return found_indices


def get_generated_piece(
    config: Dict[str, Any],
    chosen_subdir_num: Optional[int],
    chosen_index: Optional[np.ndarray],
    fold_number: int,
):
    """
    Get generated segmentation and data with given index

    :param Dict[str, Any] config: Config dictionary
    :param Optional[int] chosen_subdir_num: Subdir num of chosen generated piece (or None if not defined)
    :param Optional[np.ndarray] chosen_index: Index of chosen generated piece, array of shape (3,) (or None if not defined)
    :param int fold_number: Fold number to get data from
    :return:
    """
    # get list of available generated pieces.
    found_indices = find_generated_pieces(config, fold_number)

    # filter found tiles based on specified subdir/index
    selected_indices = []
    for check_subdir, check_index in found_indices:
        if chosen_subdir_num is not None and check_subdir != chosen_subdir_num:
            continue
        if (
            chosen_index is not None
            and not (chosen_index == check_index).all()
        ):
            continue
        selected_indices.append((check_subdir, check_index))

    if len(selected_indices) == 0:
        raise RuntimeError(
            "No valid generated tiles with specified subdir %s and index %s, not found in generated pieces: %s"
            % (
                chosen_subdir_num or "Any",
                chosen_index or "Any",
                found_indices,
            )
        )

    chosen_subdir_num, chosen_index = random.choice(found_indices)

    # read generated segmentation
    generated_filename = "instseg_pid_%d.pickle" % indexed_to_flat(
        chosen_subdir_num, chosen_index, config
    )
    generator_path = get_full_path(config, None, "generator_output_path")
    generator_files_path = os.path.join(
        generator_path, "fold_%d" % fold_number
    )
    generated_path = os.path.join(generator_files_path, generated_filename)
    generated_annot = pickle.load(open(generated_path, "rb"))
    if generated_annot.ndim == 4:
        generated_annot = generated_annot[None, :, :, :, :]
    if generated_annot.shape[1] > 1:
        raise RuntimeError(
            "Unexpected, dim 2 of generated segments > 1 (%s)"
            % (generated_annot.shape,)
        )
    # generated_annot = np.round(generated_annot).astype("int")[:,0,:,:,:] # (segs, x, y, z)
    generated_annot = (generated_annot > config["generated_threshold"]).astype(
        "int"
    )[
        :, 0, :, :, :
    ]  # (segs, x, y, z)

    # filter out empty pieces
    filled_pieces = np.sum(generated_annot, axis=(1, 2, 3)) > 0
    logging.info(
        "Read generated data with %d pieces, after thresholding keeping %d"
        % (generated_annot.shape[0], filled_pieces.sum())
    )
    generated_annot = generated_annot[filled_pieces]

    # todo: check how segments are generated, appears to generate several with identical
    #       segmentations.  could be during generation or during processing here

    return generated_annot, chosen_subdir_num, chosen_index


def make_header(num_segments):
    # make sample header definitions for given segments
    seg_fields_list = []
    for file_seg_id in range(num_segments):
        # just use dummy values
        dummy_values = [""] * len(segment_properties)
        segment_fields = dict(zip(segment_properties, dummy_values))

        # specify a few values
        segment_fields["ID"] = "Generated_%d" % file_seg_id
        segment_fields["Name"] = segment_fields["ID"]
        segment_fields["NameAutoGenerated"] = "1"
        segment_fields["Color"] = "%f %f %f" % tuple(np.random.random(3))
        segment_fields["ColorAutoGenerated"] = "1"
        segment_fields["Tags"] = sample_tags

        seg_fields_list.append(segment_fields)

    return seg_fields_list


def get_source_piece(
    config,
    subdir_num,
    annot_index,
    annot_map,
    completed_map,
    annotation_scale,
    completed_path,
    tile_annotation=None,
):
    # find crop window from data for the given piece
    data_origin_base = annot_index * annotation_scale
    data_max_base = data_origin_base + annotation_scale
    data_crop_origin = data_origin_base
    data_crop_max = data_max_base
    piece_overlap_arr = np.array(piece_overlap)
    logging.info(
        "crop base origin %s max %s" % (data_origin_base, data_max_base)
    )

    # find any bounding pieces that have been completed and update data bounds
    included_neighbours = []
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                if dx == 0 and dy == 0:
                    continue
                d_array = [dx, dy, dz]
                adj_index = annot_index + d_array
                # check if out of index bounds
                if (adj_index < 0).any() or (
                    adj_index >= annot_map.shape
                ).any():
                    continue
                # check if neighbour is completed
                if not completed_map[adj_index[0], adj_index[1], adj_index[2]]:
                    continue
                # have a completed neighbour, extend range to include overlap data
                crop_origin_adj = np.minimum(0, d_array * piece_overlap_arr)
                crop_max_adj = np.maximum(0, d_array * piece_overlap_arr)

                data_crop_origin = np.minimum(
                    data_crop_origin, data_origin_base + crop_origin_adj
                )
                data_crop_max = np.maximum(
                    data_crop_max, data_max_base + crop_max_adj
                )

                # record to include existing annotation from neighbour
                included_neighbours.append(d_array)

    logging.info(
        "annotated neighbours: %s, crop adj origin %s max %s"
        % (included_neighbours, data_crop_origin, data_crop_max)
    )

    crop_size = data_crop_max - data_crop_origin
    data_arr = get_source_data(config, subdir_num, data_crop_origin, crop_size)

    # define initial annotation segmentation, using segmentation from neighbours
    initial_annots = []
    initial_fields = []
    if tile_annotation is not None:
        # add existing annotation to expanded region
        crop_size = data_crop_max - data_crop_origin
        num_generated_segments = tile_annotation.shape[0]
        generated_annot = np.zeros(
            (num_generated_segments,) + tuple(crop_size), dtype="int"
        )

        # fill tile region with generated annotation
        crop_min = data_origin_base - data_crop_origin
        crop_max = crop_min + annotation_scale
        slices = tuple(
            [slice(None)] + [slice(x, y) for x, y in zip(crop_min, crop_max)]
        )
        generated_annot[slices] = tile_annotation

        initial_annots.append(generated_annot)
        initial_fields.extend(make_header(num_generated_segments))
    for d_array in included_neighbours:
        neighbour_result = get_neighbour_annotation(
            annot_index,
            data_origin_base,
            data_crop_origin,
            data_crop_max,
            annotation_scale,
            d_array,
            completed_path,
        )
        if neighbour_result is None:
            # no segments in neighbour
            continue
        neighbour_annot, neighbour_fields = neighbour_result
        initial_annots.append(neighbour_annot)
        initial_fields.extend(neighbour_fields)

    # combine annotation segments
    piece_associations = []
    initial_field_map = {}
    if len(initial_annots) > 0:
        # stack annotations based on segments
        initial_annot_array = np.concatenate(initial_annots, axis=0).astype(
            annot_dtype
        )  # (num_segs, x, y, z)
        full_extent = tuple(np.array(initial_annot_array.shape[1:]) - 1)

        # update fields to assign new ids
        for new_seg_id, fields in enumerate(initial_fields):
            new_names = [
                "Segment%d_%s" % (new_seg_id, x) for x in segment_properties
            ]
            existing_values = [fields[x] for x in segment_properties]
            this_segment_fields = dict(zip(new_names, existing_values))

            # update extent values with full piece extent
            this_segment_fields["Segment%d_Extent" % new_seg_id] = (
                "0 %d 0 %d 0 %d" % full_extent
            )

            # todo: change definition of referring_index etc to pass as values rather than
            #       adding to header fields
            if "referring_index" in fields and "referring_segment" in fields:
                this_association = (
                    annot_index,
                    new_seg_id,
                    fields["referring_index"],
                    fields["referring_segment"],
                )
                piece_associations.append(this_association)

            initial_field_map.update(this_segment_fields)

        logging.info(
            "resulting data shape %s, initial annot %s origin %s"
            % (data_arr.shape, initial_annot_array.shape, data_crop_origin)
        )
    else:
        # create segment file with one empty segment.
        initial_annot_array = None

    return (
        data_arr,
        initial_annot_array,
        initial_field_map,
        data_crop_origin,
        piece_associations,
    )


def get_subdir_and_index(
    config: Dict[str, Any],
    subdir_str: Optional[str],
    index_str: Optional[str],
):
    """
    Find the subdir and index from the specified strings, and check if valid
    """
    chosen_index = None
    chosen_subdir_num = None
    annot_map = None
    completed_map = None
    in_progress_map = None

    # check chosen subdir and index if specified
    if subdir_str is not None:
        if not isinstance(subdir_str, str) or not subdir_str.isnumeric():
            raise RuntimeError("Subdir should be a number")
        chosen_subdir_num = int(subdir_str)

    if chosen_subdir_num is not None:
        annot_map, annot_header, _ = get_annot_map(config, chosen_subdir_num)
        completed_map = get_completed_map(
            config, chosen_subdir_num, annot_map.shape
        )
        in_progress_map = get_completed_map(
            config, chosen_subdir_num, annot_map.shape, find_in_progress=True
        )

    if index_str is not None:
        if chosen_subdir_num is None:
            raise RuntimeError(
                "Subdir number must be defined when index is defined"
            )

        # check if specified index is part of annotation map and not completed
        chosen_index = check_index_str(index_str, completed_map, annot_map)
        # also check in progress map
        check_index(chosen_index, in_progress_map, annot_map)

    return (
        chosen_subdir_num,
        chosen_index,
        annot_map,
        completed_map,
        in_progress_map,
    )


def get_subdir_and_index_vals(
    config: Dict[str, Any],
    chosen_subdir_num: int,
    index_vals: np.ndarray,
):
    """
    Find the subdir and index from the specified strings, and check if valid
    """
    # check chosen subdir and index
    annot_map, annot_header, _ = get_annot_map(config, chosen_subdir_num)
    completed_map = get_completed_map(
        config, chosen_subdir_num, annot_map.shape
    )
    in_progress_map = get_completed_map(
        config, chosen_subdir_num, annot_map.shape, find_in_progress=True
    )

    # check if specified index is part of annotation map and not completed
    check_index(index_vals, completed_map, annot_map)
    # also check in progress map
    check_index(index_vals, in_progress_map, annot_map)

    return (
        annot_map,
        completed_map,
        in_progress_map,
    )


def make_annotation_piece(
    config: Dict[str, Any],
    index_str: Optional[List[str]],
    launch_editor: bool,
    read_generated: bool,
    subdir_str: Optional[str],
    fold_number_str: str,
    read_preferred: bool,
    write_method="segments",
):
    """
    Create a new annotation piece from the source data, including overlap with neighbouring annotations
    """
    chosen_index = None
    chosen_subdir_num = None
    annot_map = None
    completed_map = None
    in_progress_map = None

    # choose from the preferred segment list if defined. this will override subdir_str or
    # index_str values
    if read_preferred:
        # select the first index value available from the preferred list that is not already annotated
        # or in progress
        preferred_tiles = get_tiles_of_interest(config)
        if subdir_str is not None:
            # filter preferred tiles according to specified subdir
            specified_subdir_num = int(subdir_str)
            preferred_tiles = [
                tile_index
                for tile_index in preferred_tiles
                if tile_index[0] == specified_subdir_num
            ]
        if read_generated:
            # reduce preferred list to only include generated tiles
            generated_tiles = find_generated_pieces(
                config, int(fold_number_str)
            )

            def is_generated_tile(subdir, index_array):
                for generated_subdir, generated_index in generated_tiles:
                    if (
                        generated_subdir == subdir
                        and (index_array == generated_index).all()
                    ):
                        return True
                return False

            preferred_tiles = [
                tile_index
                for tile_index in preferred_tiles
                if is_generated_tile(tile_index[0], np.array(tile_index[1:]))
            ]

        for index_vals in preferred_tiles:
            chosen_subdir_num = index_vals[0]
            chosen_index = np.array(index_vals[1:])

            try:
                (
                    annot_map,
                    completed_map,
                    in_progress_map,
                ) = get_subdir_and_index_vals(
                    config, chosen_subdir_num, chosen_index
                )
            except RuntimeError as e:
                logging.info(
                    "Skipping tile of interest %s, reason: %s"
                    % (index_vals, e)
                )
                continue
            break

        if chosen_subdir_num is None or chosen_index is None:
            logging.warning(
                "Read from preferred tiles of interest specified but none found"
            )
        else:
            logging.info(
                "Using preferred tile, subdir %d index %s"
                % (chosen_subdir_num, chosen_index)
            )
    else:
        # try and read subdir and index from parameters
        (
            chosen_subdir_num,
            chosen_index,
            annot_map,
            completed_map,
            in_progress_map,
        ) = get_subdir_and_index(config, subdir_str, index_str)

    annotation_scale = np.array(config["annotation_size"])

    if read_generated:
        # read annotation from generated data
        if chosen_index is not None and chosen_subdir_num is not None:
            logging.info(
                "Reading generated data with subdir %d and index: %s"
                % (chosen_subdir_num, chosen_index)
            )
        else:
            logging.info("Reading random generated data piece")

        if fold_number_str is None:
            fold_number = 0
        else:
            fold_number = int(fold_number_str)

        # get generated segmentation, and return chosen subdir and index of not specified
        (
            generated_annot,
            new_chosen_subdir_num,
            new_chosen_index,
        ) = get_generated_piece(
            config, chosen_subdir_num, chosen_index, fold_number
        )
        if chosen_subdir_num is None:
            # update with newly chosen values
            chosen_subdir_num = new_chosen_subdir_num
            annot_map, annot_header, _ = get_annot_map(
                config, chosen_subdir_num
            )
            completed_map = get_completed_map(
                config, chosen_subdir_num, annot_map.shape
            )
        if chosen_index is None:
            chosen_index = new_chosen_index
    else:
        # choose a new piece to annotate from all incomplete pieces
        # require subdir_num to be defined
        if chosen_subdir_num is None:
            # choose a subdir at random
            chosen_subdir_num = np.random.randint(len(config["subdir_paths"]))

        # read maps if not specified
        if (
            annot_map is None
            or completed_map is None
            or in_progress_map is None
        ):
            annot_map, annot_header, _ = get_annot_map(
                config, chosen_subdir_num
            )
            completed_map = get_completed_map(
                config, chosen_subdir_num, annot_map.shape
            )
            in_progress_map = get_completed_map(
                config,
                chosen_subdir_num,
                annot_map.shape,
                find_in_progress=True,
            )

        if chosen_index is None:
            chosen_index = choose_annotation_piece(
                annot_map, completed_map, in_progress_map
            )
        generated_annot = None

    logging.info("Chosen index: %s" % (chosen_index))

    # check if output file already exists
    index_name = "_".join([str(x) for x in chosen_index])
    data_file = index_name + data_suffix
    inprogress_piece_path = get_full_path(
        config, chosen_subdir_num, "inprogress_piece_path"
    )
    os.makedirs(inprogress_piece_path, exist_ok=True)
    data_file_full = os.path.join(inprogress_piece_path, data_file)
    if os.path.exists(data_file_full):
        raise RuntimeError(
            "Output data file %s already exists" % data_file_full
        )

    annot_file = index_name + annot_suffix_segments
    annot_file_full = os.path.join(inprogress_piece_path, annot_file)
    if os.path.exists(annot_file_full):
        raise RuntimeError(
            "Output annotation file %s already exists" % annot_file_full
        )

    completed_piece_path = get_full_path(
        config, chosen_subdir_num, "completed_piece_path"
    )
    # get source data
    (
        source_data,
        initial_annot,
        initial_segment_fields,
        offset,
        piece_associations,
    ) = get_source_piece(
        config,
        chosen_subdir_num,
        chosen_index,
        annot_map,
        completed_map,
        annotation_scale,
        completed_piece_path,
        generated_annot,
    )

    # write annotation piece data
    aff_matrix = np.identity(4, dtype="float")
    nifti_object = nib.Nifti1Image(source_data, aff_matrix)
    logging.info("Writing annotation piece data output to %s" % data_file)
    nib.save(nifti_object, data_file_full)

    # write initial annotation
    # create header using same format as label map from Slicer
    # todo: check if sign change is needed.  slicer example had -ve for x,y scale values
    scales = np.identity(3, dtype="float")
    inprogress_piece_path = get_full_path(
        config, chosen_subdir_num, "inprogress_piece_path"
    )
    if write_method == "labelmap":
        annot_file = index_name + annot_suffix_labelmap
        header = {
            "space": "left-posterior-superior",  # copied from Slicer example
            "kinds": [
                "domain",
                "domain",
                "domain",
            ],  # copied from Slicer example
            "space directions": scales,
            "space origin": offset.astype("float"),
        }
        logging.info("Writing initial annotation output to %s" % annot_file)
        annot_file_full = os.path.join(inprogress_piece_path, annot_file)
        nrrd.write(
            annot_file_full,
            initial_annot,
            header,
            custom_field_map=annotation_field_map,
        )
    elif write_method == "segments":
        annot_file_full = write_annot_file(
            index_name,
            inprogress_piece_path,
            initial_annot,
            initial_segment_fields,
            scales,
        )
    else:
        raise RuntimeError("Unexpected write method: %s" % write_method)

    # record piece associations between annotated sections
    logging.info(
        "Recording %d associations between segments" % len(piece_associations)
    )
    project_data_file = get_full_path(
        config, chosen_subdir_num, "project_data_file"
    )
    lock = FileLock(project_data_file + ".lock")
    with lock:
        # read existing records from file
        stored_data = get_project_data(project_data_file, lock)

        if "associations" not in stored_data:
            stored_data["associations"] = {}
        assoc_map = stored_data["associations"]

        this_tile_name = str(chosen_index)

        # update records in assoc_map
        for this_assoc in piece_associations:
            _, ref_segment, source_index, source_segment = this_assoc
            ref_segment_name = str(ref_segment)
            if this_tile_name not in assoc_map:
                assoc_map[this_tile_name] = {}

            # record one-way association to refer to the segment that was defined first.
            # two-way assocs would not be unique if more than one reference exists
            assoc_map[this_tile_name][ref_segment_name] = (
                source_index.tolist(),
                source_segment,
            )

        # record tile offset for this piece in the records file, as storing it in the
        # meta-data of the grid object is unreliable if it is read/written to in slicer,
        # which does not seem to store additional header tags when saving
        if "offsets" not in stored_data:
            stored_data["offsets"] = {}
        offset_map = stored_data["offsets"]
        offset_map[this_tile_name] = offset.astype("int").tolist()
        print("new offset map:", offset_map)

        # write out json data
        with open(project_data_file, "w") as assoc_file:
            json.dump(stored_data, assoc_file, indent=4)

    logging.info(
        "Created data piece in: %s , and empty annotation in: %s"
        % (data_file_full, annot_file_full)
    )
    if launch_editor:
        args = [config["slicer_path"], data_file_full, annot_file_full]
        logging.info("Launching Slicer editor with arguments: %s" % args)
        subprocess.run(" ".join(args), shell=True)
    else:
        logging.info(
            "You can open this in Slicer manually, or launch it with: %s %s %s"
            % (config["slicer_path"], data_file_full, annot_file_full)
        )


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--index",
        help="Specified index values (x,y,z list) for annotation piece",
        required=False,
        nargs="+",
    )
    parser.add_argument(
        "-g",
        "--read_generated",
        help="Read piece from generated segmentations",
        required=False,
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--launch",
        help="Launch Slicer to edit piece",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--preferred",
        help="Choose from preferred tile list",
        action="store_true",
    )
    parser.add_argument(
        "-c", "--config_file", help="Project config file", required=True
    )
    parser.add_argument(
        "-s", "--subdir", help="Data subdirectory number", required=False
    )
    parser.add_argument(
        "-f",
        "--fold_number",
        help="Fold number to get generated data from",
        required=False,
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    logging.info("Preparing an annotation piece")

    make_annotation_piece(
        config,
        args.index,
        args.launch,
        args.read_generated,
        args.subdir,
        args.fold_number,
        args.preferred,
    )


if __name__ == "__main__":
    main()
