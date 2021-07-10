#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Example Data Loader for the LIDC data set. This dataloader expects preprocessed data in .npy or .npz files per patient and
a pandas dataframe in the same directory containing the meta-info e.g. file paths, labels, foregound slice-ids.
"""


import os
import random

import h5py
import numpy as np
import utils.dataloader_utils as dutils

# batch generator tools from https://github.com/MIC-DKFZ/batchgenerators
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase
from batchgenerators.dataloading.multi_threaded_augmenter import (
    MultiThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.transforms.crop_and_pad_transforms import (
    CenterCropTransform,
)
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform as Mirror,
)
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.utility_transforms import (
    ConvertSegToBoundingBoxCoordinates,
)
from common import (
    flat_to_indexed,
    get_annot_map,
    get_completed_map,
    get_source_data,
    get_source_tile_data,
    indexed_to_flat,
    load_config,
)
from get_annotated_section import find_all_subdir_sections, get_section

CONFIG_ENV_VAR = "ANNOTATION_CONFIG"
GENERATE_SUBDIR = "GENERATE_SUBDIR_NUMBER"
GENERATE_FULL = "GENERATE_FULL"


def get_train_generators(cf, logger):
    """
    wrapper function for creating the training batch generator pipeline. returns the train/val generators.
    selects patients according to cv folds (generated by first run/fold of experiment):
    splits the data into n-folds, where 1 split is used for val, 1 split for testing and the rest for training. (inner loop test set)
    If cf.hold_out_test_set is True, adds the test split to the training data.
    """
    config_file = os.environ[CONFIG_ENV_VAR]
    config = load_config(config_file)

    all_sections = find_all_subdir_sections(config)

    # separate into training and validation folds randomly
    fold_ratios = config["train_validation_splits"]
    # rng = np.random.default_rng(seed=config["split_random_seed"])
    # rng.shuffle(all_sections)
    rnd = random.Random(config["split_random_seed"])
    rnd.shuffle(all_sections)
    split_idx = round(fold_ratios[0] * len(all_sections))
    train_sections = all_sections[:split_idx]
    val_sections = all_sections[split_idx:]

    logger.info(
        "Loaded %d annotation sections, using %d train, %d val"
        % (len(all_sections), len(train_sections), len(val_sections))
    )

    train_pipeline = create_data_gen_pipeline(
        train_sections, cf=cf, annotation_config=config, is_training=True
    )
    val_pipeline = create_data_gen_pipeline(
        val_sections, cf=cf, annotation_config=config, is_training=False
    )
    batch_gen = {
        "train": train_pipeline,
        "val_sampling": val_pipeline,
        "n_val": len(val_sections),
    }
    # batch_gen["val_patient"] = create_data_gen_pipeline(
    #     val_sections, cf=cf, annotation_config=config, is_training=False
    # )

    return batch_gen


def get_test_generator(cf, logger):
    """
    wrapper function for creating the test batch generator pipeline.
    selects patients according to cv folds (generated by first run/fold of experiment)
    If cf.hold_out_test_set is True, gets the data from an external folder instead.
    """
    # set up as method to read source data over the whole space (except annotated tiles)
    # and produce generated tiles
    config_file = os.environ[CONFIG_ENV_VAR]
    config = load_config(config_file)

    generate_full_output = GENERATE_FULL in os.environ

    if generate_full_output:
        # specify all covered tiles
        logger.info("Producing output for full data")
        num_subdirs = len(config["subdir_paths"])
        batch_data = []
        for subdir_num in range(num_subdirs):
            annot_map, _, _ = get_annot_map(config, subdir_num)
            logger.info(
                "Subdir %d, %d covered tiles" % (subdir_num, annot_map.sum())
            )
            valid_tiles = np.stack(np.where(annot_map > 0), axis=1)
            batch_data.extend([(subdir_num, t) for t in valid_tiles])

    else:
        # generate tiles for one specified subdir
        subdir_num = int(os.environ[GENERATE_SUBDIR])

        # generate tiles from unannotated regions
        annot_map, annot_header, annotation_scale = get_annot_map(
            config, subdir_num
        )

        completed_map = get_completed_map(config, subdir_num, annot_map.shape)
        incomplete_map = annot_map - completed_map

        logger.info(
            "Found %d incomplete tiles out of %d"
            % (incomplete_map.sum(), annot_map.sum())
        )

        # choose subset of tiles to cover
        incomplete_tiles = np.stack(np.where(incomplete_map > 0), axis=1)
        chosen_indices = np.random.choice(
            len(incomplete_tiles),
            (config["generate_number_tiles"],),
            replace=False,
        )
        chosen_tiles = incomplete_tiles[chosen_indices]  # (num_tile, 3)

        logger.info("Using %d incomplete tiles" % (len(chosen_tiles),))

        # todo: find sections for incomplete tiles?  need to find how this is done with the automated
        #      overlapping tile selection

        # # for now just choose an origin section from each tile
        # test_sections = [x * annotation_scale for x in chosen_tiles]
        # logger.info("Using %d incomplete sections" % (len(test_sections),))

        # create data as a list of tuples, each with the subdir number and tile offset
        batch_data = [(subdir_num, x) for x in chosen_tiles]

    batch_gen = {}
    batch_iterator = PatientBatchIterator(
        batch_data, cf, config, generate_full_output
    )
    batch_gen["test"] = batch_iterator
    # batch_gen["test"] = create_data_gen_pipeline(
    #     test_sections,
    #     cf=cf,
    #     annotation_config=config,
    #     is_training=False,
    #     segments_defined=False,
    # )

    # find how many patches per instance
    # patch_size = batch_iterator.patch_size
    # patch_crop_coords_list = dutils.get_patch_crop_coords(
    #     np.zeros(config["annotation_size"]), patch_size, min_overlap=np.array(patch_size).min()
    # )

    # print("num patches %d" % len(patch_crop_coords_list))
    # batch_gen["n_test"] = len(patch_crop_coords_list)  # test_sections)
    # batch_gen["n_test"] = len(chosen_tiles)
    batch_gen["n_test"] = min(len(batch_data), config["generate_number_tiles"])

    # set up for full export if parameter defined in environ
    if generate_full_output:
        batch_gen["exporter"] = BatchExporter(cf, config)
        batch_gen["repeat_test_output"] = True
    return batch_gen


def create_data_gen_pipeline(
    split_data, cf, annotation_config, is_training=True, segments_defined=True
):
    """
    create mutli-threaded train/val/test batch generation and augmentation pipeline.
    :param patient_data: dictionary containing one dictionary per patient in the train/test subset.
    :param is_training: (optional) whether to perform data augmentation (training) or not (validation/testing)
    :param segments_defined: Don't load segments, set as zero instead
    :return: multithreaded_generator
    """

    # create instance of batch generator as first element in pipeline.
    data_gen = BatchGenerator(
        split_data,
        batch_size=cf.batch_size,
        cf=cf,
        annotation_config=annotation_config,
        segments_defined=segments_defined,
    )

    # add transformations to pipeline.
    my_transforms = []
    if is_training:
        mirror_transform = Mirror(axes=np.arange(cf.dim))
        my_transforms.append(mirror_transform)

        spatial_transform = SpatialTransform(
            patch_size=cf.patch_size[: cf.dim],
            patch_center_dist_from_border=cf.da_kwargs["rand_crop_dist"],
            do_elastic_deform=cf.da_kwargs["do_elastic_deform"],
            alpha=cf.da_kwargs["alpha"],
            sigma=cf.da_kwargs["sigma"],
            do_rotation=cf.da_kwargs["do_rotation"],
            angle_x=cf.da_kwargs["angle_x"],
            angle_y=cf.da_kwargs["angle_y"],
            angle_z=cf.da_kwargs["angle_z"],
            do_scale=cf.da_kwargs["do_scale"],
            scale=cf.da_kwargs["scale"],
            random_crop=cf.da_kwargs["random_crop"],
        )

        my_transforms.append(spatial_transform)
    else:
        my_transforms.append(
            CenterCropTransform(crop_size=cf.patch_size[: cf.dim])
        )

    my_transforms.append(
        ConvertSegToBoundingBoxCoordinates(
            cf.dim,
            get_rois_from_seg_flag=False,
            class_specific_seg_flag=cf.class_specific_seg_flag,
        )
    )
    all_transforms = Compose(my_transforms)
    # multithreaded_generator = SingleThreadedAugmenter(data_gen, all_transforms)
    multithreaded_generator = MultiThreadedAugmenter(
        data_gen,
        all_transforms,
        num_processes=cf.n_workers,
        seeds=range(cf.n_workers),
    )
    return multithreaded_generator


class BatchExporter:
    """
    Class for exporting generated segmentations.  Results of segmentations are passed to this class which stores
    results into a HDF5 file, representing the segmentation over the full data.
    """

    def __init__(self, cf, annotation_config):
        self.cf = cf
        self.annotation_config = annotation_config
        self.tile_size = np.array(
            self.annotation_config["annotation_size"]
        )  # shape (3,)
        self.output_datasets = []
        self.init_output()

    def init_output(self):
        """ Initialise output to write to, as a HDF5 file """
        if "full_generated_data_file" not in self.annotation_config:
            raise RuntimeError(
                'Field "full_generated_data_file" not defined in project config file'
            )
        filename = self.annotation_config["full_generated_data_file"]

        if self.annotation_config["full_generate_format"] == "real_valued":
            assert self.annotation_config["segmentation_method"] == "semantic"

        # produce output for each subdir
        for subdir_num in range(len(self.annotation_config["subdir_paths"])):
            # get dimensions of region
            annot_map, _, _ = get_annot_map(
                self.annotation_config, subdir_num
            )  # shape (tiles_x, tiles_y, tiles_z)
            annotation_extent = np.array(annot_map.shape) * self.tile_size

            output_full_path = os.path.join(
                self.annotation_config["project_folder"],
                self.annotation_config["subdir_paths"][subdir_num],
                filename,
            )

            if self.annotation_config["full_generate_format"] == "real_valued":
                output_dtype = "f"
                # todo: allow multiple classes to be output.  currently semantic segmentations are squashed into one class.
                num_classes = 1  # self.annotation_config["semantic_segmentation_classes"]
                output_shape = [num_classes] + annotation_extent.tolist()
            else:
                output_dtype = "i"
                output_shape = annotation_extent.tolist()

            # initialise HDF5 file for output
            if os.path.exists(output_full_path):
                raise RuntimeError(
                    "Output file %s already exists" % output_full_path
                )
            h5file = h5py.File(output_full_path, "w")
            print(
                "creating array generated_data with shape",
                output_shape,
                "dtype",
                output_dtype,
            )
            h5_dataset = h5file.create_dataset(
                "generated_data", shape=output_shape, dtype=output_dtype
            )
            self.output_datasets.append(h5_dataset)

    def export_segmentation(
        self, identifier: int, segmentation_data: np.ndarray
    ):
        """
        Write given segmentation to stored array

        :param int identifier: Identifier for a given tile, which is used as a patient ID
        :param np.ndarray segmentation_data: Array of resulting segmentation data, of shape (segs, 1, x, y, z) or
               (1, x, y, z), with float values representing segmentation foreground confidence
        """
        # get tile index from identifier
        subdir_num, index = flat_to_indexed(identifier, self.annotation_config)

        # record tile in HDF5 file
        subdir_dataset = self.output_datasets[subdir_num]
        origin = index * self.tile_size
        max = origin + self.tile_size

        if segmentation_data.ndim == 4:
            # add extra dim for consistency with format with ndim=5
            segmentation_data = segmentation_data[None, :, :, :, :]

        if self.annotation_config["full_generate_format"] == "real_valued":
            # write real value output
            # todo: for multi-class case, ensure that the data contains segmentation data in the correct
            #       index, for example if only class 2 is present, make sure it has seg_dims=2 and the values
            #       are written in the corresponding dimension
            if segmentation_data.shape[0] != subdir_dataset.shape[0]:
                raise RuntimeError(
                    "Unexpected, exported data shape %s, export dataset shape %s"
                    % (segmentation_data.shape, subdir_dataset.shape)
                )
            subdir_dataset[
                :, origin[0] : max[0], origin[1] : max[1], origin[2] : max[2]
            ] = segmentation_data[:, 0, :, :, :]
        elif self.annotation_config["full_generate_format"] == "flattened":
            # find integer values for segmentation output using threshold and flatten down
            seg_nums = np.arange(segmentation_data.shape[0])
            threshold_array = (
                segmentation_data
                > self.annotation_config["generated_threshold"]
            ).astype("int")
            flat_array = (
                seg_nums[:, None, None, None] * threshold_array[:, 0, :, :, :]
            ).max(axis=0)
            subdir_dataset[
                origin[0] : max[0], origin[1] : max[1], origin[2] : max[2]
            ] = flat_array


class BatchGenerator(SlimDataLoaderBase):
    """
    creates the training/validation batch generator. Samples n_batch_size patients (draws a slice from each patient if 2D)
    from the data set while maintaining foreground-class balance. Returned patches are cropped/padded to pre_crop_size.
    Actual patch_size is obtained after data augmentation.
    :param data: data dictionary as provided by 'load_dataset'.
    :param batch_size: number of patients to sample for the batch
    :return dictionary containing the batch data (b, c, x, y, (z)) / seg (b, 1, x, y, (z)) / pids / class_target
    """

    def __init__(
        self, data, batch_size, cf, annotation_config, segments_defined=True
    ):
        super(BatchGenerator, self).__init__(data, batch_size)

        self.cf = cf
        self.crop_margin = (
            np.array(self.cf.patch_size) / 8.0
        )  # min distance of ROI center to edge of cropped_patch.
        self.p_fg = 0.5

        self.annotation_config = annotation_config
        self.segments_defined = segments_defined

        self.next_instances = 0

    def generate_train_batch(self):
        # choose sections in turn to generate a batch
        batch_instances = self._data[
            self.next_instances : self.next_instances + self.batch_size
        ]
        if len(batch_instances) < self.batch_size:
            # shuffle list and start from beginning
            random.shuffle(self._data)
            remaining_insts = self.batch_size - len(batch_instances)
            # batch_instances = np.concatenate(
            #     [batch_instances, self._data[:remaining_insts]], axis=0
            # )
            # join as lists
            batch_instances = batch_instances + self._data[:remaining_insts]
            self.next_instances = remaining_insts

        while len(batch_instances) < self.batch_size:
            # print(
            #     "Unexpected, could not find %d instances to make batch, total available %d"
            #     % (self.batch_size, len(self._data))
            # )
            # batch_instances = np.concatenate(
            #     [batch_instances, batch_instances], axis=0
            # )[: self.batch_size]
            # join as lists
            batch_instances = (batch_instances + batch_instances)[
                : self.batch_size
            ]

        if len(batch_instances) < self.batch_size:
            raise RuntimeError(
                "Unexpected, could not find %d instances to make batch, total available %d"
                % (self.batch_size, len(self._data))
            )

        # get data for each instance
        batch_data, batch_segs, batch_num_segments = [], [], []
        for this_section in batch_instances:
            this_subdir_num, this_section_offset = this_section
            if self.segments_defined:
                # get annotation and data from completed sections
                (
                    section_annot,
                    section_data,
                    section_num_segments,
                ) = get_section(
                    self.annotation_config,
                    this_subdir_num,
                    this_section_offset,
                )

                # combine sections if necessary
                if (
                    self.annotation_config["segmentation_method"] == "semantic"
                    and self.annotation_config["semantic_segmentation_classes"]
                    == 1
                    and section_num_segments > 1
                ):
                    # this data contains multiple segments but one semantic segmentation class is needed,
                    # merge into a common class
                    section_annot = (section_annot > 0).astype("int")
                # if self.annotation_config["segmentation_method"] == "semantic" \
                #     and self.annotation_config["semantic_segmentation_classes"] != section_num_segments:
                #     # todo: check this isn't affected if classes are not present with semantic seg
                #     # the number of segment classes does not match the number of segment classes present
                #     # in the data.
                #     raise RuntimeError("Unexpected, %d semantic segmentation classes and %d found" % (self.annotation_config["semantic_segmentation_classes"], section_num_segments))
            else:
                # get data from source and set segmentation as zeros
                section_data = get_source_data(
                    self.annotation_config,
                    this_subdir_num,
                    this_section_offset,
                    self.annotation_config["section_dimensions"],
                )
                section_annot = np.zeros(section_data.shape, dtype="uint8")
                section_num_segments = 1

            # add extra dim so result has dimensions (c,x,y,z) for each instance
            batch_data.append(section_data[None, :, :, :])
            batch_segs.append(section_annot[None, :, :, :])
            batch_num_segments.append(section_num_segments)

        if (
            "normalise_method" in self.annotation_config
            and self.annotation_config["normalise_method"] == "mean_std"
        ):
            # perform mean/std normalisation, and produce data in float format
            data_mean, data_std = self.annotation_config["data_stats"][:2]
            data = (np.array(batch_data, dtype="float") - data_mean) / data_std
        else:
            data = np.array(batch_data, dtype="int32")

        seg = np.array(batch_segs, dtype="uint8")
        batch_pids = np.zeros((seg.shape[0],), dtype="uint8")
        # todo: fix hack adding an extra dim for targets
        # num_segments = np.max(seg, axis=(1, 2, 3, 4))  # (num_insts,)

        # class_target = [np.ones((n + 1,), dtype="uint8") for n in num_segments]
        # set class targets as zeros as seg to bbox adds +1 to each class
        class_target = [
            np.zeros((n + 1,), dtype="uint8") for n in batch_num_segments
        ]
        return {
            "data": data,
            "seg": seg,
            "pid": batch_pids,
            "class_target": class_target,
        }


class PatientBatchIterator(SlimDataLoaderBase):
    """
    creates a test generator that iterates over entire given dataset returning 1 patient per batch.
    Can be used for monitoring if cf.val_mode = 'patient_val' for a monitoring closer to actualy evaluation (done in 3D),
    if willing to accept speed-loss during training.
    :return: out_batch: dictionary containing one patient with batch_size = n_3D_patches in 3D or
    batch_size = n_2D_patches in 2D .
    """

    def __init__(
        self, data, cf, annotation_config, generate_full
    ):  # threads in augmenter
        super(PatientBatchIterator, self).__init__(data, 0)
        self.cf = cf
        # self.patient_ix = 0
        # self.dataset_pids = [v['pid'] for (k, v) in data.items()]
        self.patch_size = cf.patch_size
        if len(self.patch_size) == 2:
            self.patch_size = self.patch_size + [1]

        self.tile_index = 0
        self.annotation_config = annotation_config

        self.num_patches = None
        self.generate_full = generate_full
        self.complete = False

    def generate_train_batch(self):
        if self.generate_full and self.complete:
            # if there is a request for data after the data has been completed, return None
            return None

        # get data for this tile
        chosen_tile = self._data[self.tile_index]
        chosen_tile_subdir_num, chosen_tile_index = chosen_tile
        print(
            "test data reader, loading tile %d of %d (%.1f %%), subdir %d index %s"
            % (
                self.tile_index,
                len(self._data),
                100 * self.tile_index / len(self._data),
                chosen_tile_subdir_num,
                chosen_tile_index,
            )
        )
        # get data from source and set segmentation as zeros

        tile_data = get_source_tile_data(
            self.annotation_config,
            chosen_tile_subdir_num,
            chosen_tile_index,
        ).astype("int32")

        if (
            "normalise_method" in self.annotation_config
            and self.annotation_config["normalise_method"] == "mean_std"
        ):
            # perform mean/std normalisation, and produce data in float format
            data_mean, data_std = self.annotation_config["data_stats"][:2]
            tile_data = (tile_data - data_mean) / data_std

        tile_annot = np.zeros(tile_data.shape, dtype="uint8")
        # set class targets as zeros as seg to bbox adds +1 to each class
        batch_class_targets = np.zeros((1, 2), dtype="uint8")
        # create a unique tile id from index
        tile_id = indexed_to_flat(
            chosen_tile_subdir_num,
            chosen_tile_index,
            self.annotation_config,
        )

        # get 3D targets for evaluation
        if self.cf.dim == 3 or self.cf.merge_2D_to_3D_preds:
            out_data = tile_data[np.newaxis, np.newaxis]  # (b=1,c=1,x,y,z)
            out_seg = tile_annot[np.newaxis, np.newaxis]  # (n=1,c=1,x,y,z)
            out_targets = batch_class_targets

            batch_3D = {
                "data": out_data,
                "seg": out_seg,
                "class_target": out_targets,
                "pid": tile_id,
            }
            converter = ConvertSegToBoundingBoxCoordinates(
                dim=3,
                get_rois_from_seg_flag=False,
                class_specific_seg_flag=self.cf.class_specific_seg_flag,
            )
            batch_3D = converter(**batch_3D)
            batch_3D.update(
                {
                    "patient_bb_target": batch_3D["bb_target"],
                    "patient_roi_labels": batch_3D["class_target"],
                    "original_img_shape": out_data.shape,
                }
            )

        out_batch = batch_3D
        patient_batch = out_batch

        # assume training patch size can be extracted from tile, and tile is larger
        if np.any(np.array(tile_data.shape) < np.array(self.patch_size)):
            raise RuntimeError(
                "Patch size %s should be smaller than tile size %s for all dims"
                % (self.patch_size, tile_data.shape)
            )

        # crop patient-volume to patches of patch_size used during training. stack patches up
        # in batch dimension.
        patch_crop_coords_list = dutils.get_patch_crop_coords(
            tile_data,
            self.patch_size,  # min_overlap=np.array(self.patch_size).min()
        )
        new_img_batch, new_seg_batch, new_class_targets_batch = [], [], []

        for cix, c in enumerate(patch_crop_coords_list):
            seg_patch = tile_annot[c[0] : c[1], c[2] : c[3], c[4] : c[5]]
            new_seg_batch.append(seg_patch)

            new_img_batch.append(
                tile_data[None, c[0] : c[1], c[2] : c[3], c[4] : c[5]]
            )

        data = np.array(new_img_batch)  # (n_patches, c, x, y, z)
        seg = np.array(new_seg_batch)[:, np.newaxis]  # (n_patches, 1, x, y, z)
        batch_class_targets = np.repeat(
            batch_class_targets, len(patch_crop_coords_list), axis=0
        )

        print(
            "data shape %s seg shape %s batch class targets %s"
            % (data.shape, seg.shape, batch_class_targets.shape)
        )
        patch_batch = {
            "data": data,
            "seg": seg,
            "class_target": batch_class_targets,
            "pid": tile_id,
        }
        patch_batch["patch_crop_coords"] = np.array(patch_crop_coords_list)
        patch_batch["patient_bb_target"] = patient_batch["patient_bb_target"]
        patch_batch["patient_roi_labels"] = patient_batch["patient_roi_labels"]
        patch_batch["original_img_shape"] = patient_batch["original_img_shape"]

        converter = ConvertSegToBoundingBoxCoordinates(
            self.cf.dim,
            get_rois_from_seg_flag=False,
            class_specific_seg_flag=self.cf.class_specific_seg_flag,
        )
        patch_batch = converter(**patch_batch)
        out_batch = patch_batch

        self.tile_index += 1
        if self.tile_index == len(self._data):
            self.tile_index = 0
            if self.generate_full:
                self.complete = True

        return out_batch
