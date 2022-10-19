"""
deep_gesture.utils

@author: phdenzel
"""
import os
from io import BytesIO
import tarfile
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

def compress_data(files, filename=None, delete=False, verbose=False):
    """
    """
    if filename is None or not filename.endswith('.tar.gz'):
        targz = "{}.tar.gz".format(os.path.commonprefix(trgtpaths))
    else:
        targz = filename
    if verbose:
        print(targz)
    with tarfile.open(targz, "w:gz") as tf:
        for f in files:
            tf.add(f, arcname=os.path.basename(f))
    if delete:
        for f in files:
            os.remove(f)

def extract_tar(tar_archive, verbose=False):
    """
    """
    if not tarfile.is_tarfile(tar_archive):
        return
    with tarfile.open(tar_archive) as tf:
        path = os.path.dirname(tar_archive)
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tf, path)
        

def archive_data(*args, tmp_dir=None, dta_dir=None, file_type=None,
                 compress=True, verbose=False):
    """
    Archive sequence files by moving them from a source to a target directory

    Kwargs:
        tmp_dir <str> - path to the tmp directory; 
                        default: ~/.deep_gesture/tmp/
        dta_dir <str> - path to the data archive directory
                        default: ~/.deep_gesture/data/
        file_type <str> - file type (extension) targeted
    """
    tmp_dir = dg.TMP_DIR if tmp_dir is None else tmp_dir
    dta_dir = dg.DATA_DIR if dta_dir is None else dta_dir
    file_type = "" if file_type is None else file_type
    dg.utils.mkdir_p(tmp_dir)
    dg.utils.mkdir_p(dta_dir)
    filenames = [f for f in os.listdir(tmp_dir) if f.endswith(file_type)]
    filepaths = [os.path.join(tmp_dir, f) for f in filenames]
    trgtpaths = [os.path.join(dta_dir, f) for f in filenames]
    
    for src, dest in zip(filepaths, trgtpaths):
        os.rename(src, dest)

    if compress:
        targz = "{}.tar.gz".format(os.path.commonprefix(trgtpaths))
        compress_data(trgtpaths, filename=targz, delete=True, verbose=verbose)
        # extract_tar(targz)


def load_data(dta_dir=None, extract_from_tar=True, data_file_extension='.npy',
              grouping=('_', -1), verbose=False):
    """
    Extract all data from the data directory (either from tar files or
    directly from .npy files); labels are extracted from the filenames
    (as 'label_date_ID_frame.npy')

    Kwargs:
        dta_dir <str> - directory to the data files
        extract_from_tar <bool> - extract data directly from tarballs
        data_file_extension <str> - data file selection by extension
        grouping <str, int> - grouping instructions: split character and index
                              up to which the filenames are grouped
                              examples: no grouping ('.', -1); default ('_', -1)

    Return:
        features <list> - numpy data arrays (ML features)
        labels <list> - categorical data strings (ML labels)
    """
    features = []
    labels = []
    if extract_from_tar:
        tarfiles = [os.path.join(dta_dir, f) for f in os.listdir(dta_dir)
                    if f.endswith('.tar.gz')]
        for tf in tarfiles:
            with tarfile.open(tf, "r:gz") as tar:
                tar_data = []
                files = [f for f in tar.getnames() if f.endswith(data_file_extension)]
                for f in files:
                    ramfile = BytesIO()
                    ramfile.write(tar.extractfile(f).read())
                    ramfile.seek(0)
                    arr = np.load(ramfile)
                    ramfile.close()
                    tar_data.append(arr)
                gesture = os.path.basename(tf).split("_")[0]
                labels.append(gesture)
                features.append(tar_data)
                if verbose:
                    g = "_".join(os.path.basename(tf).split("_")[:3])
                    print(f"Extracting data: {g} - {len(tar_data)} frames")
    else:
        files = [os.path.join(dta_dir, f) for f in os.listdir(dta_dir)
                 if f.endswith(data_file_extension)]
        file_groups = {}
        for f in files:
            key = grouping[0].join(f.split(grouping[0])[:grouping[1]])
            group = file_groups.get(key, [])
            group.append(f)
            file_groups[key] = group
        for g in file_groups:
            group_data = []
            for f in file_groups[g]:
                arr = np.load(f)
                group_data.append(arr)
            gesture = os.path.basename(g).split("_")[0]
            features.append(group_data)
            labels.append(gesture)
            if verbose:
                g = os.path.basename(g)
                print(f"Extracting data: {g} - {len(group_data)} frames")
    return np.array(features), np.array(labels)
    

if __name__ == "__main__":
    features, labels = load_data(dg.DATA_DIR, extract_from_tar=False, verbose=True)
    print(features.shape, labels.shape)
