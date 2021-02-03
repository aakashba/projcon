import pickle
import gc
from itertools import islice
import random


filefid = pickle.load(open("filefid.pkl","rb"))
fidproj = pickle.load(open("fidproj.pkl","rb"))
projfile = pickle.load(open("pidfileid.pkl","rb"))
fidfile = pickle.load(open("filemapping.pkl","rb"))

mainfids = pickle.load(open("filteredfids.pkl","rb"))

print(len(mainfids))
pcontext = {}
count = 0
psize = 10
fsize = 10



for fid in mainfids:
    fileid = fidfile[fid]
    count +=1 
    if count % 10000 == 0:
        print(count)
        gc.collect()

    filefid[fileid].remove(fid)   # remove this fid
    pid = fidproj[fid]
    files = projfile[pid] 

    flist = random.sample(files,min(psize,len(files))) # select psize random files from the project context, all files if less than psize
    newcon = {}

    for f in flist:
        newcon[f] = random.sample(filefid[f],min(fsize,len(filefid[f])))  # trim to fsize random functions in each file


    pcontext[fid]=newcon


pickle.dump(pcontext,open("pcontbase.pkl","wb"))

