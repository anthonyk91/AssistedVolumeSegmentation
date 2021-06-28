import glob
import json
import logging
import os
import random
import shutil
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import nibabel as nib
import nrrd
import numpy as np
import yaml
from filelock import FileLock
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from PIL import Image

NiftiImage = Union[Nifti1Image, Nifti2Image]

# define suffixes for created files
annot_suffix_labelmap = "_annot_labelmap.nrrd"
annot_suffix_segments = "_annot.seg.nrrd"
data_suffix = "_data.nii.gz"
default_seg_file = "Segmentation_empty.seg.nrrd"

piece_overlap = (5, 5, 5)
completed_value = 1
overview_bound_size = (1024, 1024, 512)

annotation_field_map = {
    "space": "string",
    "kinds": "string list",
    "space directions": "double matrix",
    "space origin": "double vector",
}

segmentation_field_map = {
    "space": "string",
    "kinds": "string list",
    "space directions": "double matrix",
    "space origin": "double vector",
    "measurement frame": "double matrix",
}

segment_properties = [
    "Color",
    "ColorAutoGenerated",
    "Extent",
    "ID",
    "Name",
    "NameAutoGenerated",
    "Tags",
]

PieceIndex = Tuple[int, int, int]


def find_path_pieces(path: str) -> List[PieceIndex]:
    """
    Find pieces in the given path, for example which pieces have been completed or excluded

    :param path: Path to find pieces in
    :return: List of piece indices found in the path
    """
    piece_glob = os.path.join(path, "*_*_*" + annot_suffix_segments)
    found_pieces = glob.glob(piece_glob)

    found_indices = []
    for f in found_pieces:
        file = os.path.split(f)[-1]
        filebase, _ = os.path.splitext(file)
        index_strings = filebase.split("_")[:3]
        index: PieceIndex = tuple([int(x) for x in index_strings])
        found_indices.append(index)

    logging.info("Finding %d pieces in %s" % (len(found_indices), path))
    return found_indices


def get_completed_map(
    config: Dict[str, Any],
    subdir_num: int,
    annot_pieces_dim: Optional[Tuple[int, int, int]] = None,
) -> np.ndarray:
    """
    Get map of pieces that have been annotated and recorded as completed (ie the pieces
    in the completed directory)

    :param Dict[str, Any] config: Configuration dictionary
    :param int subdir_num: Subdirectory number
    :param Optional[Tuple[int, int, int]] annot_pieces_dim: Dimensions of full annotation pieces map
    :return: Array of annotation pieces, of same size as annotation map, representing which pieces
           have been completed
    """
    if annot_pieces_dim is None:
        # get annotation piece map from file
        annot_map, annot_header, annotation_scale = get_annot_map(
            config, subdir_num
        )
        annot_pieces_dim = annot_map.shape

    completed_piece_path = get_full_path(
        config, subdir_num, "completed_piece_path"
    )
    found_indices = find_path_pieces(completed_piece_path)

    # create occupancy grid of found completed annotations
    completed_map = np.zeros(annot_pieces_dim, dtype="int")
    if len(found_indices) > 0:
        i = np.array(found_indices)
        completed_map[i[:, 0], i[:, 1], i[:, 2]] = completed_value

    return completed_map


def get_annot_map(
    config: Dict[str, Any], subdir_num: int
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Get map of annotation pieces from file, representing which pieces are within the
    covered region.

    :param Dict[str, Any] config: Configuration dictionary
    :param int subdir_num: Subdirectory number
    :return: Tuple of array of annotation pieces, shape (x, y, z), with values indicating if each
           piece is within the covered region, dictionary of header recorded in annotation file,
           and array with values=3, representing scale of annotation map compared to source data.
    """
    annotation_piece_file = get_full_path(
        config, subdir_num, "pieces_overview"
    )
    excluded_path = get_full_path(config, subdir_num, "excluded_piece_path")

    annot_map, annot_header = nrrd.read(
        annotation_piece_file, custom_field_map=annotation_field_map
    )
    annotation_scale_matrix = annot_header["space directions"]
    annotation_scale = np.abs(
        np.round(np.diagonal(annotation_scale_matrix)).astype("int")
    )  # should be same as annotation_size
    all_annotated = annot_map.sum()

    # find pieces in the excluded folder and remove from the annotation map
    excluded_indices = find_path_pieces(excluded_path)

    # create occupancy grid of found completed annotations
    if len(excluded_indices) > 0:
        i = np.array(excluded_indices)
        annot_map[i[:, 0], i[:, 1], i[:, 2]] = 0

    logging.info(
        "%d annotation pieces (%d total, %d excluded) from %s"
        % (
            annot_map.sum(),
            all_annotated,
            len(excluded_indices),
            annot_map.shape,
        )
    )
    return annot_map, annot_header, annotation_scale


def get_project_data(project_data_file: str, lock: Optional[FileLock] = None):
    """
    Read project data fields from given project data file

    :param str project_data_file: Path to project data file
    :param Optional[FileLock] lock: Lock to use if already created.  This allows a write to be performed under
                 the same lock
    :return: Dict of fields read from data file
    """
    if lock is None:
        lock = FileLock(project_data_file + ".lock")
    stored_data = {}
    with lock:
        # read existing records from file
        if os.path.exists(project_data_file):
            try:
                with open(project_data_file, "r") as assoc_file:
                    stored_data = json.load(assoc_file)
            except FileNotFoundError:
                logging.info(
                    "Piece association file %s not found" % project_data_file
                )
            except IOError:
                logging.info(
                    "Piece association file %s not found" % project_data_file
                )

            if stored_data == {}:
                raise RuntimeError(
                    "Could not read project data from file %s"
                    % project_data_file
                )

    return stored_data


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

    # todo: add output to log file in project dir as well


def get_annotation(
    piece_index: np.ndarray, data_path: str
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray]:
    """
    Get annotation data for the given piece index

    :param np.ndarray piece_index: Array with values=3, of index of piece to read
    :param str data_path: Full path of directory containing given piece
    :return:  Tuple of annotation piece data in segment format with shape (nsegs, x, y, z),
         dictionary of header information read from annotation file, and offset of data array
         within the space of the tile
    """

    indices_str = [str(x) for x in piece_index.tolist()]
    filename = "_".join(indices_str) + annot_suffix_segments
    neighbour_file = os.path.join(data_path, filename)

    input_data, header = nrrd.read(neighbour_file)

    data_suboffset = header["space origin"]
    seg_data = make_seg_format(input_data)

    return seg_data, header, data_suboffset


def write_annot_file(
    index_name: str,
    write_path: str,
    annot_data: np.ndarray,
    annot_fields: Dict[str, Any],
    scales: np.ndarray,
) -> str:
    """
    Write data to annotation file

    :param str index_name: Name representing index of annotation piece
    :param str write_path: Path to write to
    :param np.ndarray annot_data: Annotation data to write, in segment format with shape (nsegs, x, y, z)
    :param Dict[str, Any] annot_fields: Header information to write to file
    :param np.ndarray scales: Array of scale/direction information written in header
    :return: Full path of file written to
    """
    annot_file = index_name + annot_suffix_segments
    annot_write_path = os.path.join(write_path, annot_file)
    os.makedirs(write_path, exist_ok=True)
    if annot_data is None or annot_fields is None:
        # copy default segment file
        script_path = os.path.dirname(os.path.abspath(__file__))
        default_file = os.path.join(script_path, default_seg_file)
        shutil.copyfile(default_file, annot_write_path)
        logging.info(
            "Copying initial annotation from %s to %s"
            % (default_file, annot_write_path)
        )
    else:
        space_array = np.concatenate(
            [np.full((1, 3), np.nan, dtype="float"), scales], axis=0
        )  # (4,3)
        header = {
            "space": "right-anterior-superior",  # copied from Slicer example
        }
        header.update(annot_fields)

        # set space values last to ensure that segmentation properties are defined,
        # as the source data may be in labelmap format
        header.update(
            {
                "kinds": [
                    "list",
                    "domain",
                    "domain",
                    "domain",
                ],  # copied from Slicer example
                "space directions": space_array,
                "space origin": np.zeros((3,), dtype="float"),
            }
        )
        logging.info("Writing annotation output to %s" % annot_write_path)

        # produce layered format representation
        layer_data, segment_layer_header = make_layer_format(annot_data)

        # update header information to show layers for each segment
        header.update(segment_layer_header)

        nrrd.write(
            annot_write_path,
            layer_data,
            header,
            custom_field_map=segmentation_field_map,
        )

    return annot_write_path


def get_data_piece(
    piece_index: Union[np.ndarray, List[int]], piece_path: str
) -> NiftiImage:
    """
    Given a piece index, read the data from file in the given path and return as a
    Nifti data object

    :param Union[np.ndarray, List[int]] piece_index: List/array of piece index
    :param str piece_path: Path containing data file
    :return: Nifti data object for piece
    """
    index_name = "_".join([str(x) for x in piece_index])
    piece_filename = index_name + data_suffix
    piece_filepath = os.path.join(piece_path, piece_filename)
    piece_data = nib.load(piece_filepath)
    return piece_data


def assign_cropped_region(
    source_array: np.ndarray,
    source_offset: np.ndarray,
    dest_array: np.ndarray,
    dest_offset: np.ndarray,
    copy_dim: np.ndarray,
) -> None:
    """
    Given a source and destination arrays, copy data from source to destination for a given
    region, specified by an offset within each array, and a dimension representing the size
    of the region to be copied.  Only data within the region on both source and dest will be
    copied.  This assumes each array is 3-dimensional

    :param source_array: Source array, shape (x, y, z)
    :param source_offset: Offset from source array origin to copy from (3,)
    :param dest_array: Destination array, shape (x, y, z)
    :param dest_offset: Offset from destination array origin to copy to (3,)
    :param copy_dim: Size of region to copy, shape (x, y, z)
    """
    source_size = np.array(source_array.shape[-3:])
    dest_size = np.array(dest_array.shape[-3:])
    copy_size = np.array(copy_dim)

    source_crop_min = np.clip(source_offset, 0, source_size - 1)
    source_crop_max = np.clip(source_offset + copy_size, 0, source_size)
    dest_crop_min = np.clip(dest_offset, 0, dest_size - 1)
    dest_crop_max = np.clip(dest_offset + copy_size, 0, dest_size)

    # find common copy region and update copy range
    common_dim = np.minimum(
        source_crop_max - source_crop_min, dest_crop_max - dest_crop_min
    )
    source_crop_max = source_crop_min + common_dim
    dest_crop_max = dest_crop_min + common_dim

    # create slice definition for defined dimensions in offset/dim, and cover all preceding
    # dimensions
    source_full_dims = [slice(None)] * (
        source_array.ndim - len(source_crop_min)
    )
    source_slices = source_full_dims + [
        slice(x, y) for x, y in zip(source_crop_min, source_crop_max)
    ]
    dest_full_dims = [slice(None)] * (dest_array.ndim - len(dest_crop_min))
    dest_slices = dest_full_dims + [
        slice(x, y) for x, y in zip(dest_crop_min, dest_crop_max)
    ]
    dest_array[tuple(dest_slices)] = source_array[tuple(source_slices)]


def get_file_list(data_dir: str, extension: str = "tiff") -> List[str]:
    """ Get list of files in given directory with given extension """
    glob_pattern = os.path.join(data_dir, "*.%s" % extension)
    glob_list = glob.glob(glob_pattern)
    sorted_files = sorted(glob_list)
    return sorted_files


def load_config(config_file: str) -> Dict[str, Any]:
    """
    Given a config file, read and return values as a dict

    :param str config_file: Config file path
    :return: Configuration dict
    """
    # todo: add checks that each required field is defined?

    config = yaml.safe_load(open(config_file, "r"))

    return config


def read_segment_file(
    in_file: str, format: str = "segmentation"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read the given segmentation from file

    :param str in_file: Source file path to read from
    :param str format: Format that
    :return: Tuple of array of segmentation data, and array of scales of segmentation data
    """
    logging.info("Reading overview segmentation from %s" % in_file)
    seg_map, seg_header = nrrd.read(in_file)
    # todo: change overview segmentation file from labelmap to .seg.nrrd

    # todo: check orientation of segment map, and use of values from the following array.
    #       they appear to be negated in examples, is that important?
    seg_scales = np.abs(np.diagonal(seg_header["space directions"]))

    if format == "labelmap":
        # labelmap format, which is a binary map of the full space. this is simpler to read in
        # however is a bit more awkward for the user to create in Slicer so not preferred
        pass
    elif format == "segmentation":
        # read coverage data from .seg.nrrd file
        if seg_map.ndim > 3:
            # reduce if multiple segments present (take sum)
            seg_map = np.clip(seg_map.sum(axis=0), 0, completed_value)

        # pad array so it covers the full region (from origin) without an offset. this may
        # crop larger values, ie the tensor may start at 0,0,0 and have size smaller than the
        # original space, but this shouldn't cause problems
        space_origin = seg_header["space origin"]
        grid_origin = np.round(space_origin / seg_scales).astype("int")

        padding_before = list(zip(grid_origin, [0, 0, 0]))
        seg_map = np.pad(seg_map, padding_before, mode="constant")
    else:
        raise RuntimeError("Unknown format %s" % format)

    return seg_map, seg_scales


def get_cropped_source_data(
    stack_list: List[str], crop_origin: np.ndarray, crop_max: np.ndarray
) -> np.ndarray:
    """
    Read data from the given image files in an image stack

    :param List[str] stack_list: List of filenames representing images in a stack
    :param np.ndarray crop_origin: Origin of region to crop, array of shape (3,)
    :param np.ndarray crop_max: Max position of region to crop, array of shape (3,)
    :return: Cropped source data as an array of shape (x,y,z)
    """
    stack_files = stack_list[crop_origin[2] : crop_max[2]]
    img_slices = []
    for f in stack_files:
        img = Image.open(f)
        img_arr = np.array(img)

        # crop from image
        img_crop = img_arr[
            crop_origin[0] : crop_max[0],
            crop_origin[1] : crop_max[1],
        ]
        img_slices.append(img_crop)

    return np.stack(img_slices, axis=2)


def get_source_tile_stored(
    config: Dict[str, Any],
    subdir_num: int,
    tile_index: np.ndarray,
    check_only=False,
) -> Union[bool, np.ndarray]:
    """
    Read source tile from stored data folder and return data, or check if present

    :param Dict[str, Any] config: Config dictionary
    :param int subdir_num: Subdir data number
    :param np.ndarray tile_index: Tile index number, array of shape (3,)
    :param bool check_only: Only check if data is present or not
    :return: Either array containing data from source data,
           or bool value representing if data is present
    """
    # try and read from file
    indices_str = [str(x) for x in tile_index.tolist()]
    data_path = get_full_path(config, subdir_num, "source_piece_path")
    filename = "_".join(indices_str) + data_suffix
    source_file = os.path.join(data_path, filename)

    if check_only:
        return os.path.exists(source_file)

    nii_data = nib.load(source_file)
    return np.array(nii_data.dataobj)


def get_source_data_stored(
    config: Dict[str, Any],
    subdir_num: int,
    data_crop_origin: np.ndarray,
    crop_size: np.ndarray,
    check_only=False,
) -> Union[bool, np.ndarray]:
    """
    Get source data from the given crop position from locally stored tile data, or check
    if it is present

    :param Dict[str, Any] config: Config dictionary
    :param int subdir_num: Subdir data number
    :param np.ndarray data_crop_origin: Origin in source units of section to crop, array of shape (3,)
    :param np.ndarray crop_size: Size of section to crop, array of shape (3,)
    :param bool check_only: Only check if data is present or not
    :return: Either array containing specified cropped region from source data,
           or bool value representing if data is present
    """

    # check if the crop region is covered by stored source tiles
    tile_size = np.array(config["annotation_size"])
    min_tile = data_crop_origin // tile_size
    max_tile_inclusive = (data_crop_origin + crop_size) // tile_size
    section_data = None
    for x in range(min_tile[0], max_tile_inclusive[0] + 1):
        for y in range(min_tile[1], max_tile_inclusive[1] + 1):
            for z in range(min_tile[2], max_tile_inclusive[2] + 1):
                this_tile_index = np.array([x, y, z])

                # try and read file
                source_tile = get_source_tile_stored(
                    config, subdir_num, this_tile_index, check_only
                )
                if check_only:
                    if not source_tile:
                        return False
                else:
                    if section_data is None:
                        section_data = np.zeros(
                            crop_size, dtype=source_tile.dtype
                        )

                    tile_offset = this_tile_index * tile_size
                    tile_section_offset = tile_offset - data_crop_origin
                    assign_cropped_region(
                        source_tile,
                        np.array([0, 0, 0]),
                        section_data,
                        tile_section_offset,
                        tile_size,
                    )

    if check_only:
        return True

    return section_data


def get_source_data(
    config: Dict[str, Any],
    subdir_num: int,
    data_crop_origin: np.ndarray,
    crop_size: np.ndarray,
) -> np.ndarray:
    """
    Get source data from the given crop position in the given subdir, using
    a crop size from the specified section dimensions

    :param Dict[str, Any] config: Config dictionary
    :param int subdir_num: Subdir data number
    :param np.ndarray data_crop_origin: Origin in source units of section to crop, array of shape (3,)
    :param np.ndarray crop_size: Size of section to crop, array of shape (3,)
    :return: Array containing specified cropped region from source data
    """

    if get_source_data_stored(
        config, subdir_num, data_crop_origin, crop_size, check_only=True
    ):
        logging.info(
            "Reading source data with origin %s, size %s from stored data"
            % (data_crop_origin, crop_size)
        )
        return get_source_data_stored(
            config, subdir_num, data_crop_origin, crop_size, check_only=False
        )
    else:
        logging.info(
            "Reading source data with origin %s, size %s from stack data"
            % (data_crop_origin, crop_size)
        )
        return get_source_data_stack(
            config, subdir_num, data_crop_origin, crop_size
        )


def get_source_data_stack(
    config: Dict[str, Any],
    subdir_num: int,
    data_crop_origin: np.ndarray,
    crop_size: np.ndarray,
) -> np.ndarray:
    """
    Get source data from the given crop position from image stack data, using
    a crop size from the specified section dimensions

    :param Dict[str, Any] config: Config dictionary
    :param int subdir_num: Subdir data number
    :param np.ndarray data_crop_origin: Origin in source units of section to crop, array of shape (3,)
    :param np.ndarray crop_size: Size of section to crop, array of shape (3,)
    :return: Array containing specified cropped region from source data
    """
    source_data_path = get_source_data_path(config, subdir_num)
    section_dims = np.asarray(crop_size)

    # read data for given piece from source data
    stack_list = get_file_list(source_data_path)

    data_crop_max = data_crop_origin + section_dims
    logging.info(
        "get source data, crop origin %s max %s"
        % (data_crop_origin, data_crop_max)
    )

    return get_cropped_source_data(stack_list, data_crop_origin, data_crop_max)


def get_source_tile_data(
    config: Dict[str, Any],
    subdir_num: int,
    tile_index: np.ndarray,
    pad_to_size=True,
) -> np.ndarray:
    """
    Get source data for the given tile in the given subdir

    :param Dict[str, Any] config: Config dictionary
    :param int subdir_num: Subdir number to read from
    :param np.ndarray tile_index: Index of tile to read, as array of 3 values
    :param bool pad_to_size: If read data is smaller than tile size, pad with zeros to fill tile size
    :return: Array of tile data, with dims=3 and shape (x, y, z)
    """
    tile_size = np.array(config["annotation_size"])
    data_crop_origin = tile_index * tile_size

    read_data = get_source_data(
        config, subdir_num, data_crop_origin, tile_size
    )

    read_shape = np.array(read_data.shape)
    if np.any(read_shape < tile_size) and pad_to_size:
        # add padding to fill to size.  this should be added to larger index edge as tiles
        # should have origin matching data source origin
        pad_size = tile_size - read_shape
        read_data = np.pad(
            read_data, list(zip([0, 0, 0], pad_size)), mode="constant"
        )

    return read_data


def flat_to_indexed(
    flat_pos: int, shapes: np.ndarray
) -> Tuple[int, np.ndarray]:
    """
    Convert a position from flat format (single number) to indexed format (subdir number and array of 3 numbers)

    :param int flat_pos: Position in flat format
    :param np.ndarray  shapes: Shape of space that the flat format is calculated from (array of 3 numbers)
    :return: Tuple of the resulting subdir number and position in indexed format
    """
    positions = []
    remainder = flat_pos
    while len(shapes) >= 1:
        multiplier = np.prod(shapes)
        last_pos = remainder // multiplier
        positions.append(last_pos)
        remainder -= multiplier * last_pos
        shapes = shapes[:-1]
    positions.append(remainder)

    subdir_num = positions[0]

    return subdir_num, np.array(positions[:0:-1])


def indexed_to_flat(
    subdir_num: int, indexed_pos: np.ndarray, shapes: np.ndarray
):
    """
    Convert a position from indexed format (array of 3 numbers and subdir number) to flat format (single number)

    :param int subdir_num: Subdir number
    :param np.ndarray indexed_pos: Position in indexed format, ie array of 3 values
    :param np.ndarray shapes: Shape of space that the flat format is calculated from (array of 3 numbers)
    :return: Value of position in flat representation
    """
    if len(indexed_pos) != len(shapes):
        raise RuntimeError(
            "indexed_to_flat, expect positions and shapes to be same size, got %s, %s"
            % (indexed_pos, shapes)
        )

    multiplier = 1
    sum = 0
    for idx in range(len(shapes)):
        sum += indexed_pos[idx] * multiplier
        multiplier *= shapes[idx]

    # include subdir value
    sum += subdir_num * multiplier

    return sum


# todo: change all ".._full" param calls to use function call
def get_full_path(config, subdir_num, param_name):
    """
    Provide the full path for the given config parameter, using the given subdirectory

    :param config: Config dictionary
    :param subdir_num: Number of subdir
    :param param_name: Name of parameter containing a relative file path
    :return: Full path of given file including base directory and subdir number
    """
    if subdir_num is None:
        # return project path that is not under a subdir.  check that given
        # parameter name is appropriate
        if param_name not in ["generator_output_path"]:
            raise RuntimeError(
                "Unexpected, requesting path relative to project dir for parameter %s"
                % param_name
            )
        return os.path.join(config["project_folder"], config[param_name])
    else:
        # subdir_path = "subdir_%d" % subdir_num
        if subdir_num >= len(config["subdir_paths"]):
            raise RuntimeError(
                "Invalid subdir number %d, have %d subdirs: %s"
                % (
                    subdir_num,
                    len(config["subdir_paths"]),
                    config["subdir_paths"],
                )
            )
        subdir_path = config["subdir_paths"][subdir_num]
        return os.path.join(
            config["project_folder"], subdir_path, config[param_name]
        )


def get_source_data_path(config, subdir_num):
    """
    Return the source data path for the given subdir number

    :param config: Config dictionary
    :param subdir_num: Number of subdir
    :return: Full path to given source data
    """
    # assume data is a list corresponding with subdirs
    data_paths = config["source_data_paths"]
    if not isinstance(data_paths, list):
        raise RuntimeError(
            "Unexpected, source_data_paths entry in config should be a list corresponding with subdirs"
        )
    if subdir_num >= len(data_paths):
        raise RuntimeError(
            "Unexpected, could not get source_data_paths entry %d, found %d in list"
            % (subdir_num, len(data_paths))
        )

    return data_paths[subdir_num]


def get_all_subdirs(config):
    """
    Find all subdir numbers in the project path according to the given configuration

    :param config: Config dictionary
    :return: List of numbers representing subdirs
    """
    # check subdir paths exist for all defined subdirs in config
    dir_list_relative = config["subdir_paths"]
    dir_list = [
        os.path.join(config["project_folder"], x) for x in dir_list_relative
    ]
    existing_dirs = [x for x in dir_list if os.path.exists(x)]
    if len(existing_dirs) != len(dir_list_relative):
        raise RuntimeError(
            "%d subdirs given in config file, but only %d are present: %s"
            % (len(dir_list_relative), len(existing_dirs), dir_list_relative)
        )

    return list(range(len(existing_dirs)))


def get_random_subdir(config):
    """
    Find list of available subdirs and choose one at random

    :param config: Config dictionary
    :return: Number of chosen subdir
    """
    subdir_numbers = get_all_subdirs(config)

    return random.choice(subdir_numbers)


def make_seg_format(input_data):
    """
    Ensure data is in segment format, not labelmap.  Make new array and return if required

    :param np.ndarray input_data: Input data array, in either labelmap, segment or layer format
    :return: Data array, in segment format (nsegs, x, y, z)
    """

    def make_segments_from_layer(layer_data):
        num_segments = layer_data.max()
        seg_maps = [
            (input_data == x).astype(input_data.dtype)
            for x in range(1, num_segments + 1)
        ]
        seg_data = np.stack(seg_maps)
        return seg_data

    if input_data.ndim == 3:
        # remap from labelmap format, with one layer
        seg_data = make_segments_from_layer(input_data)
        print(
            "remapping input data from",
            input_data.shape,
            "to shape",
            seg_data.shape,
        )
        return seg_data
    elif input_data.ndim == 4 and np.max(input_data) > 1:
        # currently in layer format, expand into segment format
        # expand each layer at a time
        num_segments = np.max(input_data)
        seg_data = np.zeros((num_segments,) + input_data.shape[1:])
        for layer in range(input_data.shape[0]):
            this_layer_data = input_data[layer]
            seg_data += make_segments_from_layer(this_layer_data)
        print(
            "remapping input data from",
            input_data.shape,
            "to shape",
            seg_data.shape,
        )
        return seg_data
    return input_data


def make_labelmap_format(input_data):
    """
    Ensure data is in labelmap format, not segment.  Make new array and return if required

    :param np.ndarray input_data: Input data array, in either labelmap, segment or layer format
    :return: Data array, in labelmap format (x, y, z)
    """
    # remap labelmap format if used
    if len(input_data.shape) == 4:
        num_segments = input_data.shape[0]
        if input_data.shape[0] > 1 and np.max(input_data) == 1:
            # data is segment format
            index_vals = np.arange(1, num_segments + 1)
            # reduce using max segment number to resolve overlaps
            labelmap = np.max(
                input_data * index_vals[:, None, None, None], axis=0
            )
            print(
                "remapping input data from",
                input_data.shape,
                "to shape",
                labelmap.shape,
            )
            return labelmap
        else:
            # data is in layer format, merge down
            labelmap = np.max(input_data, axis=0)
            print(
                "remapping input data from",
                input_data.shape,
                "to shape",
                labelmap.shape,
                ", squashing layers using max (which can lose overlap information)",
            )
            return labelmap

    return input_data


def make_layer_format(
    input_data: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Ensure data is in layer format, and convert if necessary.  Make new array and return if
    required

    :param np.ndarray input_data: Input data array, in either labelmap, segment or layer format
    :return: Tuple of data array, in layer format (l, x, y, z) with max_value=num_segments,
             and header indicating which layer each segment is in, and the corresponding
             label value
    """
    if input_data.ndim == 3:
        # currently in labelmap format, return as layer format with single layer
        layer_data = input_data[None, :, :, :]
        num_segments = np.max(input_data) + 1
    elif input_data.ndim == 4:
        # check if in segment format
        if input_data.shape[0] > 1 and np.max(input_data) == 1:
            num_segments = input_data.shape[0]
            index_vals = np.arange(1, num_segments + 1)
            # make representation with one layer for each segment, with corresponding values
            layer_data = input_data * index_vals[:, None, None, None]
            # merge layers down one at a time
            for check_layer in range(num_segments - 1, 0, -1):
                seg_layer_filled = layer_data[check_layer] > 0
                next_layer_filled = layer_data[check_layer - 1] > 0
                has_overlaps = np.logical_and(
                    seg_layer_filled, next_layer_filled
                ).any()
                if not has_overlaps:
                    # merge layers
                    merged_layer = (
                        layer_data[check_layer] + layer_data[check_layer - 1]
                    )
                    # update layers
                    layer_data = np.concatenate(
                        [
                            layer_data[: check_layer - 1],
                            merged_layer[None],
                            layer_data[check_layer + 1 :],
                        ],
                        axis=0,
                    )

            print(
                "remapping input data from",
                input_data.shape,
                "to shape",
                layer_data.shape,
            )
        else:
            print("leaving current layer format")
            num_segments = np.max(input_data) + 1
            layer_data = input_data
    else:
        raise RuntimeError(
            "Unexpected, given segment data with %d layers, expect 3 or 4"
            % input_data.ndim
        )

    # produce header representing corresponding layer for each segment
    segment_vals = np.arange(1, num_segments + 1)
    matching_layer_mask = segment_vals[:, None, None, None, None] == layer_data
    matching_layer = matching_layer_mask.any(axis=(2, 3, 4)).argmax(
        axis=1
    )  # shape (num_segments,)

    segment_layer_header = {
        "Segment%d_Layer" % seg_idx: matching_layer[seg_idx]
        for seg_idx in range(num_segments)
    }
    # update to specify values in map as well
    segment_layer_header.update(
        {
            "Segment%d_LabelValue" % seg_idx: seg_idx + 1
            for seg_idx in range(num_segments)
        }
    )
    return layer_data, segment_layer_header
