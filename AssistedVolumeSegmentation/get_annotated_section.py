# from the set of annotated pieces, extract a section with relevant annotated sections.
# this requires combining data from neighbouring slices and matching associated annotation segments
import argparse
import logging
from typing import Any, Dict, Tuple

import numpy as np

from AssistedVolumeSegmentation.common import (
    assign_cropped_region,
    completed_value,
    get_all_subdirs,
    get_annotation,
    get_completed_map,
    get_data_piece,
    get_full_path,
    get_layer_segments,
    get_project_data,
    get_random_subdir,
    init_logging,
    load_config,
)


def check_segment(
    start_tile, this_section_offset, completed_map, section_size, tile_size
):
    """
    Check if a given segment is valid, ie all tiles within the range of the segment
    are annotated

    :param start_tile:
    :param this_section_offset:
    :param completed_map:
    :param section_size:
    :param tile_size:
    :return:
    """
    section_max_inclusive = this_section_offset + section_size - 1
    max_tile_inclusive = section_max_inclusive // tile_size
    # check tiles covered by section
    slices = tuple(
        [slice(p, q) for p, q in zip(start_tile, max_tile_inclusive + 1)]
    )
    all_completed = completed_map[slices].all()

    return all_completed


def find_all_subdir_sections(config):
    """
    Find all sections within each subdir, and return as a list of tuples

    :param config: Config dictionary
    :return: List of all sections, each a tuple of (subdir_num, section_offset)
    """
    all_subdir_numbers = get_all_subdirs(config)
    # get all sections within each subdir, and record as a list of (subdir_num, offset) tuples
    subdir_sections = [
        (subdir_num, offset)
        for subdir_num in all_subdir_numbers
        for offset in find_all_sections(config, subdir_num)
    ]

    return subdir_sections


def find_all_sections(config, subdir_num):
    """
    Find all data sections with the given dimensions that fit within the given annotated pieces

    :param config: Config dictionary
    :param subdir_num: Number of subdir to read from
    :return: Array of offsets representing valid section positions
    """

    # get all completed annotations
    completed_map = get_completed_map(config, subdir_num)
    completed_tiles = np.stack(
        np.where(completed_map == completed_value), axis=1
    )  # (num_pieces, 3)

    # for each piece, find starting offsets for sections that are valid
    tile_size = np.array(config["annotation_size"])
    section_size = np.array(config["section_dimensions"])
    tile_offsets = completed_tiles * tile_size[None, :]  # (num_pieces, 3)

    valid_sections = []
    map_x, map_y, map_z = completed_map.shape
    for tile_index, tile_offset in zip(completed_tiles, tile_offsets):
        valid_offsets = np.ones(tile_size, dtype="bool")

        # find range of tiles covered by extent of sections
        max_section_inclusive = (
            (tile_offset + tile_size - 1) + section_size - 1
        )
        max_covered_tile_inclusive = max_section_inclusive // tile_size

        # step over neighbouring tiles and remove offsets that reach incomplete tiles
        for tile_x in range(tile_index[0], max_covered_tile_inclusive[0] + 1):
            for tile_y in range(
                tile_index[1], max_covered_tile_inclusive[1] + 1
            ):
                for tile_z in range(
                    tile_index[2], max_covered_tile_inclusive[2] + 1
                ):
                    valid_position = (
                        tile_x < map_x and tile_y < map_y and tile_z < map_z
                    )
                    if (
                        valid_position
                        and completed_map[tile_x, tile_y, tile_z]
                    ):
                        continue

                    # invalid tile, remove all base tile offsets that cover this tile
                    # find range of base tile offsets that reach this tile
                    this_tile_offset = (
                        np.array([tile_x, tile_y, tile_z]) * tile_size
                    )
                    base_offset = this_tile_offset - section_size + 1
                    min_offset_tile = np.clip(
                        base_offset - tile_offset, 0, tile_size - 1
                    )
                    max_offset_tile = np.clip(
                        min_offset_tile + (tile_size - 1) + section_size,
                        0,
                        tile_size,
                    )

                    # fill range of tile from this base offset
                    slices = tuple(
                        [
                            slice(m, n)
                            for m, n in zip(min_offset_tile, max_offset_tile)
                        ]
                    )
                    valid_offsets[slices] = 0

        valid_offsets_indices = (
            np.stack(np.where(valid_offsets), axis=1) + tile_offset[None, :]
        )  # (offsets,3)
        valid_sections.append(valid_offsets_indices)
        logging.info(
            "tile %s offset %s valid sections %d"
            % (tile_index, tile_offset, len(valid_offsets_indices))
        )
        # print("tile %s offset %s valid sections %d" % (tile_index, tile_offset, len(valid_offsets_indices)))

    return np.concatenate(
        valid_sections, axis=0
    )  # Array of shape (positions, 3). ##List of length (positions), each an array of size (3,)


# not used
def find_all_sections_slow(config, subdir_num):
    """
    Find all data sections with the given dimensions that fit within the given annotated pieces

    :param config: Config map
    :param section_dim: Dimensions of sections to create
    :return: List of offsets representing valid section positions
    """

    # get all completed annotations
    completed_map = get_completed_map(config, subdir_num)
    completed_pieces = np.stack(
        np.where(completed_map == completed_value), axis=1
    )  # (num_pieces, 3)

    # for each piece, find starting offsets for sections that are valid
    tile_size = np.array(config["annotation_size"])
    section_size = np.array(config["section_dimensions"])
    piece_offsets = completed_pieces * tile_size[None, :]  # (num_pieces, 3)

    valid_sections = []
    for t, p in zip(completed_pieces, piece_offsets):
        this_valid_sections = [
            section_offset
            for dx in range(tile_size[0])
            for dy in range(tile_size[1])
            for dz in range(tile_size[2])
            for section_offset in [p + [dx, dy, dz]]
            if check_segment(
                t, section_offset, completed_map, section_size, tile_size
            )
        ]
        logging.info(
            "tile %s offset %s valid sections %d"
            % (t, p, len(this_valid_sections))
        )
        valid_sections.extend(this_valid_sections)

    return valid_sections  # [(3,)]


def find_random_section(config, subdir_num):
    """
    Select a random section from the annotated data in the given subdir

    :param config: Config dictionary
    :param subdir_num: Subdir number
    :return: Array of shape (3,) representing section offset in source units
    """
    valid_sections = find_all_sections(config, subdir_num)
    logging.info("Found %d valid sections" % (len(valid_sections),))

    return valid_sections[np.random.choice(len(valid_sections))]


def verify_section(config, subdir_num, section_offset):
    """
    Check if the section with the given offset is valid, ie all of the enclosed region
    of the section is annotated

    :param config: Config dictionary
    :param subdir_num: Subdir number
    :param section_offset: Offset of section in source units
    :return: True if valid section
    """

    completed_map = get_completed_map(config, subdir_num)
    tile_size = np.array(config["annotation_size"])
    section_size = np.array(config["section_dimensions"])
    start_tile = section_offset // tile_size

    return check_segment(
        start_tile, section_offset, completed_map, section_size, tile_size
    )


# not used
def find_random_section_stochastic(config, subdir_num, max_tries=1e4):
    completed_map = get_completed_map(config, subdir_num).astype("bool")

    # choose random tile positions that have an origin within the annotated tiles, loop until
    # a valid tile is found.  this is not the most efficient way but seems reasonable enough
    # and allows a flat distribution to be produced of all sections enclosed within the annotation
    # region.
    section_origin = None
    tile_size = np.array(config["annotation_size"])
    section_dim = np.array(config["section_dimensions"])
    for i in range(max_tries):
        # choose a random tile for origin
        annot_tiles = np.stack(np.where(completed_map), axis=1)  # (tiles,3)
        chosen_tile = annot_tiles[np.random.choice(len(annot_tiles))]

        # choose a random position within tile
        candidate_origin_tile = np.random.randint(
            [0, 0, 0], config["annotation_size"]
        )
        candidate_origin = tile_size * chosen_tile + candidate_origin_tile

        # check if this is a valid section
        # check all tiles covered by section are defined
        candidate_max_inclusive = candidate_origin + section_dim - 1
        max_tile_inclusive = candidate_max_inclusive // tile_size
        slices = [
            slice(p, q) for p, q in zip(chosen_tile, max_tile_inclusive + 1)
        ]
        check_tiles = completed_map[slices]
        valid_tile = check_tiles.all()

        if valid_tile:
            section_origin = candidate_origin
            break

    if section_origin is None:
        raise RuntimeError(
            "Unable to find valid section after %d tries" % (max_tries)
        )
    return section_origin


def get_section(
    config: Dict[str, Any], subdir_num: int, section_offset: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Find annotation pieces corresponding with the given section

    :param Dict[str, Any] config: Config dictionary
    :param int subdir_num: Subdir number
    :param np.ndarray section_offset: Offset in source units of given section, shape (3,)
    :return: Tuple of array of annotated segment data of shape (x,y,z) with dimensions of section and
            values as integers of segment numbers, array of source data with shape (x,y,z), and count
            of number of annotated segments in the section
    """

    completed_piece_path = get_full_path(
        config, subdir_num, "completed_piece_path"
    )
    project_data_file = get_full_path(config, subdir_num, "project_data_file")
    section_dim = config["section_dimensions"]

    stored_data = get_project_data(project_data_file)
    assoc_map = stored_data["associations"]

    piece_size = np.array(config["annotation_size"])
    min_piece = section_offset // piece_size
    max_inclusive_piece = (section_offset + section_dim - 1) // piece_size

    def get_segment_number(piece_index, segment_index, segment_map):
        piece_name = str(np.array(piece_index))
        piece_seg_map = segment_map.get(piece_name, {})
        return piece_seg_map.get(segment_index, None)

    def set_segment_number(
        piece_index, segment_index, segment_map, new_seg_id
    ):
        # record the given segment in the segment map
        piece_name = str(np.array(piece_index))
        if piece_name not in segment_number_map:
            segment_map[this_piece_name] = {}
        segment_map[this_piece_name][segment_index] = new_seg_id

    def get_segment_association(piece_index, segment_index, association_map):
        piece_name = str(np.array(piece_index))
        piece_seg_map = association_map.get(piece_name, {})
        return piece_seg_map.get(segment_index, None)

    section_annot = None
    section_data = None

    # get completed annotation pieces
    copy_dest_offset = np.zeros((3,), dtype="int")
    segment_number_map = {}
    next_segment_id = 0
    for x in range(min_piece[0], max_inclusive_piece[0] + 1):
        for y in range(min_piece[1], max_inclusive_piece[1] + 1):
            for z in range(min_piece[2], max_inclusive_piece[2] + 1):
                this_piece_index = np.array([x, y, z])
                this_piece_name = str(this_piece_index)
                this_annot = get_annotation(
                    this_piece_index, completed_piece_path
                )
                this_data = get_data_piece(
                    this_piece_index, completed_piece_path
                )

                # determine segment number for each segment
                (
                    this_annot_data,
                    this_annot_header,
                    this_annot_suboffset,
                ) = this_annot
                segment_layers = get_layer_segments(this_annot_data)
                for piece_segment_num, segment_layer in segment_layers:
                    # first check if this segment is already defined, eg from being initialised
                    # from a referring segment
                    if get_segment_number(
                        this_piece_index, piece_segment_num, segment_number_map
                    ):
                        continue

                    # next check if this segment refers to another segment
                    ref_segment = get_segment_association(
                        this_piece_index, piece_segment_num, assoc_map
                    )
                    if ref_segment is not None:
                        source_index, source_segment_sum = ref_segment

                        # check if reference segment is defined
                        ref_seg_id = get_segment_number(
                            source_index,
                            source_segment_sum,
                            segment_number_map,
                        )
                        if ref_seg_id is not None:
                            # set number for this segment as well
                            set_segment_number(
                                this_piece_index,
                                piece_segment_num,
                                segment_number_map,
                                ref_seg_id,
                            )
                            continue

                        # create new segment ID and record for this segment and referenced (source) segment
                        set_segment_number(
                            this_piece_index,
                            piece_segment_num,
                            segment_number_map,
                            next_segment_id,
                        )
                        set_segment_number(
                            source_index,
                            source_segment_sum,
                            segment_number_map,
                            next_segment_id,
                        )
                        next_segment_id += 1
                        continue

                    # this segment does not refer to another segment, create a new segment number for it
                    set_segment_number(
                        this_piece_index,
                        piece_segment_num,
                        segment_number_map,
                        next_segment_id,
                    )
                    next_segment_id += 1

                if section_annot is None or section_data is None:
                    # create empty annotation and data tensors.  store annotations in labelmap representation,
                    # with a separate int value for each section (and 0 background)
                    section_annot = np.zeros(
                        section_dim, dtype=this_annot_data.dtype
                    )
                    section_data = np.zeros(
                        section_dim, dtype=this_data.get_data_dtype()
                    )

                # crop corresponding section from annotation.  the annotation array is expected to
                # cover the full tile, however check the tile offset anyway in case it was opened and
                # saved again in Slicer

                # handle different cases for setting segment ids in the resulting cropped section
                if config["segmentation_method"] == "semantic":
                    # the annot data is in layer format, maintain the original ids and flatten
                    # layers just using max
                    id_map = this_annot_data.max(axis=0)
                elif config["segmentation_method"] == "instance":
                    # first make flat array containing segment ids.  segments should be non-overlapping,
                    # however when flattening the largest seg id is taken.
                    # to map segment ids read from file to output segment numbers for the segment, use
                    # vectorize to map values
                    vfunc = np.vectorize(
                        lambda x: get_segment_number(
                            this_piece_index, x, segment_number_map
                        )
                        if x > 0
                        else 0
                    )
                    mapped_annot_data = vfunc(this_annot_data)
                    # flatten layers using max function
                    id_map = mapped_annot_data.max(axis=0)  # (x,y,z)
                else:
                    raise RuntimeError(
                        "Unknown segmentation_method: %s"
                        % config["segmetation_method"]
                    )

                piece_offset = this_piece_index * piece_size
                annot_data_offset = piece_offset + this_annot_suboffset.astype(
                    "int"
                )
                section_min_annot_piece = section_offset - annot_data_offset
                assign_cropped_region(
                    id_map,
                    section_min_annot_piece,
                    section_annot,
                    copy_dest_offset,
                    section_dim,
                )

                # crop corresponding section from data and assign
                section_min_piece = section_offset - piece_offset
                assign_cropped_region(
                    np.array(this_data.dataobj),
                    section_min_piece,
                    section_data,
                    copy_dest_offset,
                    section_dim,
                )

    num_segments = next_segment_id
    return section_annot, section_data, num_segments


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section_offset", help="Origin of section", required=False, nargs="+"
    )
    parser.add_argument(
        "--config_file", help="Project config file", required=True
    )
    parser.add_argument(
        "--subdir", help="Data subdirectory number", required=True
    )
    args = parser.parse_args()

    config = load_config(args.config_file)

    def get_subdir_num(subdir_str):
        if not isinstance(subdir_str, str) or not subdir_str.isnumeric():
            raise RuntimeError("Subdir should be a number")
        return int(args.subdir)

    section_offset = args.section_offset
    if section_offset is None or args.subdir is None:
        if args.subdir is None:
            subdir_num = get_random_subdir(config)
        else:
            subdir_num = get_subdir_num(args.subdir)
        section_offset = find_random_section(config, subdir_num)
    else:
        subdir_num = get_subdir_num(args.subdir)
        section_offset = np.array([int(x) for x in args.section_offset])
        if not verify_section(
            config,
            subdir_num,
            section_offset,
        ):
            raise RuntimeError("Section offset %s not valid" % section_offset)

    logging.info(
        "Retrieving an annotation section with dim %s, offset %s"
        % (config["section_dimensions"], section_offset)
    )

    section_annot, section_data, _ = get_section(
        config,
        subdir_num,
        section_offset,
    )

    # display some properties of resulting arrays
    logging.info(
        "section annotation shape %s min %d max %d dtype %s"
        % (
            section_annot.shape,
            section_annot.min(),
            section_annot.max(),
            section_annot.dtype,
        )
    )
    logging.info(
        "section data shape %s min %d max %d dtype %s"
        % (
            section_data.shape,
            section_data.min(),
            section_data.max(),
            section_data.dtype,
        )
    )


if __name__ == "__main__":
    main()
