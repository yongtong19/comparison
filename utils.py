def printProgressBar(
    iteration, total, prefix="", suffix="", decimals=1, length=100, fill=" "
):
    # referred from https://gist.github.com/snakers4/91fa21b9dda9d055a02ecd23f24fbc3d
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "=" * (length - filledLength)
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
    if iteration == total:
        print()
