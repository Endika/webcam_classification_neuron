import os
import sys
import uuid

photos = int(sys.argv[1])
directory = sys.argv[2]

os.system('mkdir -p dataset/{}'.format(directory))
for index in range(photos):
    os.system(
        'streamer -f jpeg -s 1024 -o dataset/{}/{}_{}.jpeg'.format(
            directory, uuid.uuid4().hex, directory))
