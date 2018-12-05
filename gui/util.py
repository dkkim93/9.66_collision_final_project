import os


def read_filename(path):
    filename_n = []

    files = os.listdir(path)
    for filename in files:
        filename_n.append(path + "/" + filename)

    return filename_n
