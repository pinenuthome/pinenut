import contextlib
import os
import urllib.request


class Config:
    enable_backprop = True
    train = True


@contextlib.contextmanager
def using_config(name, value, config=Config):
    old_value = getattr(config, name)
    setattr(config, name, value)
    try:
        yield
    finally:
        setattr(config, name, old_value)

# ======================================================================
# download function for downloading the dataset
# ======================================================================

cache_dir = os.path.join(os.path.expanduser('~'), '.pinenut')


def show_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    print('\r%d%%' % percent, end='')


def download(url, filename=None):
    if filename is None:
        filename = url[url.rfind('/') + 1:]
    filename = os.path.join(cache_dir, filename)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(filename):
        print('File already exists!')
        return filename

    print('Downloading ' + filename + '...')
    try:
        urllib.request.urlretrieve(url, filename, show_progress)
    except:
        if os.path.exists(filename):
            os.remove(filename)
        print('Download error!')
        raise
    print(' Done!')

    return filename
