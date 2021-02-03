import tokenizer
import pickle
import sys
import uuid
import gc

comlen = 14
sdatlen = 10 # average is 8 functions per file
tdatlen = 25
pdatlen = 10
smllen = 100 # average is 870

def save(obj, filename):
	pickle.dump(obj, open(filename, 'wb'))

box = '/nfs/projects/projcon'

coms_trainf = box+'/output/coms.train'
coms_valf = box+'/output/coms.val'
coms_testf = box+'/output/coms.test'

sdats_trainf = box+'/output/sdats.train'
sdats_valf = box+'/output/sdats.val'
sdats_testf = box+'/output/sdats.test'

tdats_trainf = box+'/output/tdats.train'
tdats_valf = box+'/output/tdats.val'
tdats_testf = box+'/output/tdats.test'

sml_trainf = box+'/output/smls.train'
sml_valf = box+'/output/smls.val'
sml_testf = box+'/output/smls.test'


comstok = tokenizer.Tokenizer().load(box+'/coms.tok')
#sdatstok = tokenizer.Tokenizer().load('sdats.tok')
tdatstok = tokenizer.Tokenizer().load(box + '/tdats.tok')
sdatstok = tdatstok 
pdatstok = tdatstok# note, same tokenizer for tdats and sdats
smlstok = tokenizer.Tokenizer().load(box + '/smls.tok')

com_train = comstok.texts_to_sequences_from_file(coms_trainf, maxlen=comlen)
com_val = comstok.texts_to_sequences_from_file(coms_valf, maxlen=comlen)
com_test = comstok.texts_to_sequences_from_file(coms_testf, maxlen=comlen)
tdats_train = tdatstok.texts_to_sequences_from_file(tdats_trainf, maxlen=tdatlen)
tdats_val = tdatstok.texts_to_sequences_from_file(tdats_valf, maxlen=tdatlen)
tdats_test = tdatstok.texts_to_sequences_from_file(tdats_testf, maxlen=tdatlen)
sml_train = smlstok.texts_to_sequences_from_file(sml_trainf, maxlen=smllen)
sml_val = smlstok.texts_to_sequences_from_file(sml_valf, maxlen=smllen)
sml_test = smlstok.texts_to_sequences_from_file(sml_testf, maxlen=smllen)

sdats_train = dict()
sdats_val = dict()
sdats_test = dict()
pdats_train= dict()
pdats_val= dict()
pdats_test= dict()

for line in open(sdats_trainf, 'r'):
    (fid, sdat) = line.split(',')
    fid = int(fid)
    sdat = sdat.split(';')
    sdats_train[fid] = sdatstok.texts_to_sequences(sdat[:sdatlen], maxlen=tdatlen)

for line in open(sdats_valf, 'r'):
    (fid, sdat) = line.split(',')
    fid = int(fid)
    sdat = sdat.split(';')
    sdats_val[fid] = sdatstok.texts_to_sequences(sdat[:sdatlen], maxlen=tdatlen)
    
for line in open(sdats_testf, 'r'):
    (fid, sdat) = line.split(',')
    fid = int(fid)
    sdat = sdat.split(';')
    sdats_test[fid] = sdatstok.texts_to_sequences(sdat[:sdatlen], maxlen=tdatlen)

count =0 
ptrainf = pickle.load(open("rprojtrain.pkl","rb"))
for fid in ptrainf:
    count +=1
    if count % 10000 == 0:
        print(count)
    pdats_train[fid] = []
    for filecon in ptrainf[fid][:pdatlen]:
        pdats_train[fid].append(pdatstok.texts_to_sequences(filecon[:sdatlen], maxlen=tdatlen))

ptrainf.clear()
gc.collect()
pvalf = pickle.load(open("rprojval.pkl","rb"))

for fid in pvalf:
    pdats_val[fid] = []
    for filecon in pvalf[fid][:pdatlen]:
        pdats_val[fid].append(pdatstok.texts_to_sequences(filecon[:sdatlen], maxlen=tdatlen))

pvalf.clear()

ptestf = pickle.load(open("rprojtest.pkl","rb"))

for fid in ptestf:
    pdats_test[fid] = []
    for filecon in ptestf[fid][:pdatlen]:
        pdats_test[fid].append(pdatstok.texts_to_sequences(filecon[:sdatlen], maxlen=tdatlen))

gc.collect()

assert len(com_train) == len(tdats_train)
assert len(com_val) == len(tdats_val)
assert len(com_test) == len(tdats_test)

out_config = {'tdatvocabsize': tdatstok.vocab_size, 'sdatvocabsize': sdatstok.vocab_size, 'comvocabsize': comstok.vocab_size, 
        'smlvocabsize': smlstok.vocab_size, 'pvocabsize':pdatstok.vocab_size, 'sdatlen': sdatlen, 'tdatlen': tdatlen, 'comlen': comlen,
            'smllen': smllen, 'pdatlen': pdatlen }

dataset = {'ctrain': com_train, 'cval': com_val, 'ctest': com_test, 
			'dstrain': sdats_train, 'dsval': sdats_val, 'dstest': sdats_test,
			'dttrain': tdats_train, 'dtval': tdats_val, 'dttest': tdats_test,
			'strain': sml_train, 'sval': sml_val, 'stest': sml_test, 
                        'ptrain': pdats_train, 'pval': pdats_val, 'ptest': pdats_test,
                        'comstok': comstok, 'tdatstok': tdatstok, 'sdatstok': sdatstok, 'smltok': smlstok, 'pdatstok':pdatstok,
            'config': out_config}

#save(dataset, './archive/dataset_{}.pkl'.format(str(uuid.uuid1())[:8]))
save(dataset, 'dataset_random.pkl')
