#
#	Fit pulse shapes of FRBs
#
#								

#	--------------------------	Import modules	---------------------------

import os, sys
import numpy as np
from globalpars import *
from profiler import *

def print_instructions():

	#	Print instructions to terminal
	
	print("\n            You probably need some assistance here!\n")
	print("\n Arguments are       --- <FRB name string> <mode> <ncomp>\n")	
	print(" Supported Modes are --- fit        (Fit the burst shape)")
	print("                         other	   (Anything else if required)")
	
	print("\n            Now let's try again!\n")
	
	return(0)

#	--------------------------	Read inputs	-------------------------------
if(len(sys.argv)<4):
	print_instructions()
	sys.exit()

frbname		=	sys.argv[1]										#	FRB name string (YYMMDDx)
exmode		=	sys.argv[2]										#	What to do
ncomp		=	int(sys.argv[3])

#	-------------------------	Do steps	-------------------------------

frbhtr		=	"{}.pkl".format(frbname)
plotdir		=	""							#	Blank = current directory 
xlim		=	[-10.0, 15.0]				#	Time range of plot
fsize		=	[9.0,8.0]					#	Figure size
dtms		=	2.0							#	Tick spacing in time

if(os.path.exists(frbhtr)):
		
	frbfile		=	open(frbhtr,'rb')		
	pdata		=	pkl.load(frbfile)
	frbfile.close()
		
	if(exmode=='fit'):		
		congaussfitter(plotdir,pdata,xlim,fsize,dtms,ncomp,"I")
			
	elif(exmode=='other'):			
		print("Nothing to do as of now...")
					
else:
	print("PANIC - Pickle not found! Have you prepared it?")
