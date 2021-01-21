import gzip
import os

def read_gzip(path):
    os.listdir(path + "hourly/")
    with gzip.open('big_file.txt.gz', 'rb') as f:
        for line in f:
            print(line)

if __name__ == '__main__':
    #path = "../../experiments/district_7/"
    path = "experiments/district_7/"