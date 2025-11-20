# catgt save command to separate out regions (by channels)
# place -save line in the catgt parameters

# file name code, appended with a 1 = Thalalmus, 2 = ACC, 3= CA/HC

# format of save command :-save=js,ip1,ip2,channel-list

#INPUTS:
# js = always 2 for our .ap files 
# ip1 = name of input file (i.e. ends in imec0)
# ip2 = name of output file (i.e. ends in imec1 = thalamus, imec2= ACC, imec3= CA/HC)
# channel-list = obvi, the list of channels. All files get channel 384, bc that is SYNC channel

# NPX1, ACC and THAL
'-save=2,0,1,0:36, 38:217, 219:240, 242:374, 376:382, 384' # thalamus cites
'-save=2,0,2, 37, 218, 241, 375,383,384' # ACC cites

# NPX1, ACC and THAL
'-save=2,0,1,0:47, 96:143, 192:203, 384' # thalamus cites
'-save=2,0,2,48:95, 144:191, 204:383, 384' # acc cites

# NPX3, CA and THAL
'-save=2,0,1,192:239, 251:313, 336:383, 384' #thalamus cites
'-save=2,0,3,0:191, 240:250, 314:335, 384' #ca/hc cites
