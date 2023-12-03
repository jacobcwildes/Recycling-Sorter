import os
import time

for i in range(100):
	
	cmd= "arecord -D hw:2,0 -d 4 -f cd ~/Desktop/Recycling-Sorter/recordings/glass/glass" + str(i+1) + ".wav -c 1"
	os.system(cmd)
	time.sleep(0.1)
	print("Recording Finished ", (i+1))
