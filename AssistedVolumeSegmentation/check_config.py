# script to check if a given config file provides all tags from the template config file

import argparse
import logging
import os
from typing import Any, Dict

import yaml

from AssistedVolumeSegmentation.common import init_logging, load_config

sample_config_file = "sample_project_config.yaml"
source_path = os.path.dirname(os.path.abspath(__file__))
sample_config_path = os.path.join(source_path, os.pardir, sample_config_file)


def check_config(
    config: Dict[str, Any],
):
    # read sample config
    sample_config = yaml.safe_load(open(sample_config_path, "r"))

    # check each sample config is present
    missing_count = 0
    for sample_key in sample_config.keys():
        if sample_key not in config:
            logging.info("Key missing: %s" % sample_key)
            missing_count += 1

    if missing_count == 0:
        logging.info("All keys from sample config found")
    else:
        logging.info(
            "%d keys from sample config not found, please refer to sample config file for examples: %s"
            % (missing_count, sample_config_path)
        )


def main():
    init_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file", help="Project config file", required=True
    )

    args = parser.parse_args()
    config = load_config(args.config_file)

    logging.info("Checking config file: %s" % args.config_file)

    check_config(
        config,
    )


if __name__ == "__main__":
    main()
