


def check_path(path):
    if path[-1] is not "/":
        return path + "/"
    else:
        return path