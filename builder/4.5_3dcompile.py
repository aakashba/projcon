import tokenizer
import pickle
import sys
import uuid
import gc

comlen = 14
sdatlen = 20 # average is 8 functions per file
tdatlen = 100
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


comstok = tokenizer.Tokenizer().load(box+'/coms.tok')
#sdatstok = tokenizer.Tokenizer().load('sdats.tok')
tdatstok = tokenizer.Tokenizer().load(box + '/tdats.tok')
sdatstok = tdatstok 
pdatstok = tdatstok# note, same tokenizer for tdats and sdats
smlstok = tokenizer.Tokenizer().load(box+'/smls3D.tok')

com_train = comstok.texts_to_sequences_from_file(coms_trainf, maxlen=comlen)
com_val = comstok.texts_to_sequences_from_file(coms_valf, maxlen=comlen)
com_test = comstok.texts_to_sequences_from_file(coms_testf, maxlen=comlen)
tdats_train = tdatstok.texts_to_sequences_from_file(tdats_trainf, maxlen=tdatlen)
tdats_val = tdatstok.texts_to_sequences_from_file(tdats_valf, maxlen=tdatlen)
tdats_test = tdatstok.texts_to_sequences_from_file(tdats_testf, maxlen=tdatlen)

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
pdats = pickle.load(open("rprojectdats.pkl","rb"))
for fid in sdats_train:
    count +=1
    if count % 10000 == 0:
        print(count)
    pdats_train[fid] = []
    try:
        for filecon in pdats[fid][:pdatlen]:
            pdats_train[fid].append(pdatstok.texts_to_sequences(filecon[:sdatlen], maxlen=tdatlen))
    except:
        del tdats_train[fid]
        del com_train[fid]


for fid in sdats_val:
    pdats_val[fid] = []
    try:
     for filecon in pdats[fid][:pdatlen]:
            pdats_val[fid].append(pdatstok.texts_to_sequences(filecon[:sdatlen], maxlen=tdatlen))
    except:
        del tdats_val[fid]
        del com_val[fid]


for fid in sdats_test:
    pdats_test[fid] = []
    try:
        for filecon in pdats[fid][:pdatlen]:
          pdats_test[fid].append(pdatstok.texts_to_sequences(filecon[:sdatlen], maxlen=tdatlen))
    except:
        del tdats_test[fid]
        del com_test[fid]

pdats.clear()
gc.collect()

srcml_nodes = pickle.load(open(box+'/output/dataset.srcml_nodes.pkl', 'rb'))
srcml_edges = pickle.load(open(box+'/output/dataset.srcml_edges.pkl', 'rb'))

strain_nodes = dict()
strain_edges = dict()
sval_nodes = dict()
sval_edges = dict()
stest_nodes = dict()
stest_edges = dict()

for line in open(tdats_trainf, 'r'):
    (fid, tdat) = line.split(',')
    fid = int(fid)
    strain_nodes[fid] = srcml_nodes[fid]
    strain_edges[fid] = srcml_edges[fid]

for line in open(sdats_valf, 'r'):
    (fid, tdat) = line.split(',')
    fid = int(fid)
    sval_nodes[fid] = srcml_nodes[fid]
    sval_edges[fid] = srcml_edges[fid]

for line in open(sdats_testf, 'r'):
    (fid, tdat) = line.split(',')
    fid = int(fid)
    stest_nodes[fid] = srcml_nodes[fid]
    stest_edges[fid] = srcml_edges[fid]

assert len(com_train) == len(tdats_train)
assert len(com_val) == len(tdats_val)
assert len(com_test) == len(tdats_test)

out_config = {'tdatvocabsize': tdatstok.vocab_size, 'sdatvocabsize': sdatstok.vocab_size, 'comvocabsize': comstok.vocab_size, 
        'smlvocabsize': smlstok.vocab_size, 'pvocabsize':pdatstok.vocab_size, 'sdatlen': sdatlen, 'tdatlen': tdatlen, 'comlen': comlen,
            'smllen': smllen, 'pdatlen': pdatlen }

dataset = {'ctrain': com_train, 'cval': com_val, 'ctest': com_test, 
			'dstrain': sdats_train, 'dsval': sdats_val, 'dstest': sdats_test,
			'dttrain': tdats_train, 'dtval': tdats_val, 'dttest': tdats_test,
                        'strain_nodes': strain_nodes, 'sval_nodes': sval_nodes, 'stest_nodes': stest_nodes,
                        'strain_edges': strain_edges, 'sval_edges': sval_edges, 'stest_edges':stest_edges,
                       # 'ptrain': pdats_train, 'pval': pdats_val, 'ptest': pdats_test,
                        'comstok': comstok, 'tdatstok': tdatstok, 'sdatstok': sdatstok, 'smltok': smlstok, 'pdatstok':pdatstok,
            'config': out_config}

#save(dataset, './archive/dataset_{}.pkl'.format(str(uuid.uuid1())[:8]))
save(dataset, 'dataset_3Drandom.pkl')
