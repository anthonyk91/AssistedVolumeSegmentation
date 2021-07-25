import io
import os
import shutil
import site
import subprocess
import sys
import zipfile
from urllib.request import urlopen

from setuptools import find_packages, setup

patch_url = (
    "https://github.com/techtonik/python-patch/archive/refs/heads/master.zip"
)
# temp_zip_name = "testfile.zip"
requirements_file = "requirements.txt"
third_party_path = "3rdparty"
mdt_extracted_name = (
    "medicaldetectiontoolkit-87f279eca3a33920a535f82ae40014fec6e82418"
)
mdt_target_name = "medicaldetectiontoolkit"
patch_file = "mdt_torch1x.patch"
mdt_url = (
    "https://github.com/MIC-DKFZ/medicaldetectiontoolkit/archive/87f279e.zip"
)
patch_install_path = "python-patch-master"

script_path = os.path.dirname(__file__)
path_file = "AssistedVolumeSegmentation.pth"
mdt_path_file = "medicaldetectiontoolkit.pth"


def get_requirements(filename):
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


def install_custom_dependencies():
    # install script dependencies directly
    print("Installing custom dependencies")

    # copy patch script
    resp = urlopen(patch_url).read()
    remote = io.BytesIO(resp)
    zf = zipfile.ZipFile(remote, "r")
    zf.extractall(script_path)

    # import patch
    sys.path.append(os.path.join(script_path, patch_install_path))
    import patch

    # download medicaldetectiontoolkit with specific commit
    resp = urlopen(mdt_url).read()
    remote = io.BytesIO(resp)
    zf = zipfile.ZipFile(remote, "r")
    mdt_extract_path = os.path.join(script_path, third_party_path)
    zf.extractall(mdt_extract_path)

    # rename so path matches patch
    mdt_extracted_path = os.path.join(mdt_extract_path, mdt_extracted_name)
    mdt_target_path = os.path.join(mdt_extract_path, mdt_target_name)
    # remove anything in target path, for example from a previous install/attempt
    if os.path.exists(mdt_target_path):
        shutil.rmtree(mdt_target_path, ignore_errors=True)
    os.rename(mdt_extracted_path, mdt_target_path)

    # apply patch
    patchfile_path = os.path.join(mdt_extract_path, patch_file)
    patchobj = patch.fromfile(patchfile_path)
    result = patchobj.apply(strip=0, root=mdt_extract_path)
    if not result:
        raise RuntimeError("Failed to apply patch on MDT data")

    # install MDT package
    subprocess.check_call(
        [sys.executable, "setup.py", "install"], cwd=mdt_target_path
    )

    # add MDT path to site packages
    site_path = site.getsitepackages()[0]
    mdt_path_file_full = os.path.join(site_path, mdt_path_file)
    with open(mdt_path_file_full, "w") as f:
        f.write(mdt_target_path + "\n")

    print("Done installing custom dependencies")


def main():

    setup(
        name="AssistedVolumeSegmentation",
        version="0.0.1",
        url="https://github.com/anthonyk91/AssistedVolumeSegmentation",
        author="A. Knittel",
        license="MIT",
        description="Assisted Volume Segmentation tools.",
        packages=find_packages(),
        install_requires=get_requirements(requirements_file),
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Developers",
            "Programming Language :: Python :: 3.7",
        ],
        python_requires=">=3.7",
        entry_points={
            "console_scripts": [
                "find_annotation_pieces = AssistedVolumeSegmentation.find_annotation_pieces:main",
                "find_stats = AssistedVolumeSegmentation.find_stats:main",
                "get_annotated_section = AssistedVolumeSegmentation.get_annotated_section:main",
                "get_annotation_piece = AssistedVolumeSegmentation.get_annotation_piece:main",
                "get_data_overview = AssistedVolumeSegmentation.get_data_overview:main",
                "map_source_data = AssistedVolumeSegmentation.map_source_data:main",
                "set_piece_complete = AssistedVolumeSegmentation.set_piece_complete:main",
                "check_config = AssistedVolumeSegmentation.check_config:main",
                # "mdt_exec = exec"
            ],
        },
    )

    # add AssistedVolumeSegmentation path to site packages
    site_path = site.getsitepackages()[0]
    path_file_full = os.path.join(site_path, path_file)
    with open(path_file_full, "w") as f:
        f.write(script_path + "\n")

    try:
        install_custom_dependencies()
    except Exception as e:
        print(
            "Failed to install medicaldetectiontoolkit, training and inference of models will not be available however annotation tools can still be used."
        )


if __name__ == "__main__":
    main()
