import pickle
import time


def timestamp(start: float, messege: str) -> float:
    end = time.time()
    t = end - start
    milisec = int((t % 1) * 1000)
    t = int(t)
    sec = t % 60
    min = (t % 3600) // 60
    hour = t // 3600
    print(
        messege
        + str(hour)
        + "h "
        + str(min)
        + "m "
        + str(sec)
        + "s "
        + str(milisec)
        + "ms"
    )
    return end


def save_model(clf, name, directory) -> None:
    with open(directory + "/" + name + ".pkl", "wb") as fid:
        pickle.dump(clf, fid)


def open_model(name, directory):
    with open(directory + "/" + name + ".pkl", "rb") as fid:
        return pickle.load(fid)


def save_to_file(content, name_of_file: str = "temp.txt"):
    f = open(name_of_file, "w")
    f.write(str(content))
    f.close()
