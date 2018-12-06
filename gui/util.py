import os


def read_filename(path):
    filename_n = []

    files = os.listdir(path)
    for filename in files:
        if ".png" in filename:
            filename_n.append(path + "/" + filename)

    return filename_n
