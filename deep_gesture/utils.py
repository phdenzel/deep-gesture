"""
deep_gesture.utils

@author: phdenzel
"""
import os
from datetime import datetime
import numpy as np
import deep_gesture as dg


def mkdir_p(pathname):
    """
    Create a directory as if using 'mkdir -p' on the command line

    Args:
        pathname <str> - create all directories in given path

    Return
        pathname <str> - given path returned
    """
    from errno import EEXIST
    try:
        os.makedirs(pathname)
    except OSError as exc:  # Python > 2.5
        if exc.errno == EEXIST and os.path.isdir(pathname):
            pass
        else:
            raise


def generate_filename(prefix=None, name_id=None, part_no=None, extension='npy'):
    """
    Generate a unique filename for deep_gesture data such as landmark arrays or
    feed images

    Kwargs:
        prefix <str> - prefix of file name
        name_id <int> - ID to generate unique identifier
        part_no <int> - optional part number if landmarks part of a sequence
        extension <str> - file extension; default: npy (binary numpy extension)
    """
    date = datetime.today().strftime('%y-%m-%d').replace("-", "")
    if prefix is None:
        prefix = 'deep_gesture'
    if name_id is None:
        name_id = hash(name_id)
    if part_no is None:
        part_no = np.base_repr(abs(hash(os.urandom(42))), 32)
    else:
        part_no = f"{part_no:03d}"
    key = np.base_repr(abs(hash((f'{prefix}_{date}', name_id))), 32)
    fname = f'{prefix}_{date}_{key}_{part_no}.{extension}'
    return fname


def clean_dir(*args, tmp_dir=None):
    """
    Delete all files in a given directory

    Kwargs:
        tmp_dir <str> - path to the tmp directory; default: ~/.deep_gesture/tmp/
    """
    tmp_dir = dg.TMP_DIR if tmp_dir is None else tmp_dir
    dg.utils.mkdir_p(os.path.dirname(tmp_dir))
    filenames = os.listdir(tmp_dir)
    filepaths = [os.path.join(tmp_dir, f) for f in filenames]
    for f in filepaths:
        os.remove(f)


def archive_data(*args, tmp_dir=None, dta_dir=None):
    """
    Archive sequence files by moving them from a source to a target directory

    Kwargs:
        tmp_dir <str> - path to the tmp directory; 
                        default: ~/.deep_gesture/tmp/
    """
    tmp_dir = dg.TMP_DIR if tmp_dir is None else tmp_dir
    dta_dir = dg.DATA_DIR if dta_dir is None else dta_dir
    dg.utils.mkdir_p(tmp_dir)
    dg.utils.mkdir_p(dta_dir)
    filenames = os.listdir(tmp_dir)
    filepaths = [os.path.join(tmp_dir, f) for f in filenames]
    trgtpaths = [os.path.join(dta_dir, f) for f in filenames]
    for src, dest in zip(filepaths, trgtpaths):
        os.rename(src, dest)


def load_data(*args, data_dir=None):
    """
    TODO

    Kwargs:
        data_dir <str> - directory containing the data
    """
    data_dir = dg.DATA_DIR if data_dir is None else data_dir
    files = os.listdir(data_dir)
    print(files)
    
