import os


def get_data_split(age: int, direction: bool) -> int:
    """
    Given a particular age, returns the number of images that are older (if direction == 0) or younger (if direction == 1) than that age
    :param age: the age to use as a threshold
    :param direction: a boolean parameter deciding whether to go over or under the threshold
    :return: the number of participants over and under the threshold
    """
    # getting filenames
    fnames = os.listdir("utkcropped")

    if direction == 0:
        return len([fname for fname in fnames if "utkcropped" not in fname and int(fname.split("_")[0]) < age])
    elif direction == 1:
        return len([fname for fname in fnames if "utkcropped" not in fname and int(fname.split("_")[0]) >= age])


if __name__ == "__main__":
    print(get_data_split(30, 1))