import os
import time

for i in range(5):
	
	cmd= "arecord -D hw:2,0 -d 4 -f cd ~/Desktop/471/project/paper" + str(i+1) + ".wav -c 1"
	os.system(cmd)
	time.sleep(0.1)
