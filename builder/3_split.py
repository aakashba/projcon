import pickle
import sys

splitfile = "/nfs/projects/funcom/data/java/output/trainvaltest_ids.pkl"
mainfile = pickle.load(open("rprojectdats.pkl","rb"))


spliter = pickle.load(open(splitfile,"rb"))

trainfid = spliter['trainfid']

valfid = spliter['valfid']

testfid = spliter['testfid']

train = dict((fid, mainfile[fid]) for fid in trainfid if fid in mainfile.keys())
pickle.dump(train,open("rprojtrain.pkl","wb"))

val = dict((fid, mainfile[fid]) for fid in valfid if fid in mainfile.keys()) 
pickle.dump(val,open("rprojval.pkl","wb"))

test = dict((fid, mainfile[fid]) for fid in testfid if fid in mainfile.keys()) 
pickle.dump(test,open("rprojtest.pkl","wb"))

