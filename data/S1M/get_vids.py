import os
import pytube
f = open('train_partition.txt', 'r')
lines = f.readlines()
MAX_NUM_VIDS = 70000
i = 0
index = 0
while i < MAX_NUM_VIDS:
  name = 'sports-1m_{0:09d}'.format(i)
  try:
    yt = pytube.YouTube(lines[index].split(' ')[0])
    print lines[index].split(' ')[0], " i=", i," index=", index
    yt.streams.filter(subtype='mp4', res="360p").first().download(filename=name)
    i += 1
    index += 1
  except:
    print "skip:{}".format(lines[i].split(' ')[0])
    index += 1


  # yt.streams
  # os.system('pytube -e mp4 -r 360p -f ' + name + ' ' + lines[i].split(' ')[0])

print 'done.'
