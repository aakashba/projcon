import pickle

tdats = pickle.load(open("/nfs/projects/funcom/data/standard/output/newdats.pkl","rb"))

pcontext = pickle.load(open("pcontbase.pkl","rb"))

newcontext = {}
psize = 10 # random data too sparse many small number entries
fsize = 10
count = 0

for fid in pcontext:
    count +=1
    if count % 10000 == 0:
        print(count)
    newcon = []
    p = 0
    for fileid in pcontext[fid]:
        filecon=[]
        p += 1
        f = 0
        for pfid in pcontext[fid][fileid]:
            try:
                filecon.append(tdats[pfid])
                f += 1
            except:
                continue
            if f == fsize:
                break

        newcon.append(filecon)
        if p == psize:
            break

    newcontext[fid] = newcon

pickle.dump(newcontext,open("rprojectdats.pkl","wb"))


