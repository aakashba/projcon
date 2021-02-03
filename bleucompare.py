import sys
import pickle
import argparse
import re

from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from myutils import prep, drop

smoothing = SmoothingFunction()

import warnings
warnings.filterwarnings("ignore")

def fil(com):
    ret = list()
    for w in com:
        if not '<' in w:
            ret.append(w)
    return ret

def get_pred_ref(input_file):

    import tokenizer

    prep('preparing predictions list... ')
    preds = dict()
    predicts = open(input_file, 'r')
    for c, line in enumerate(predicts):
        (fid, pred) = line.split('\t')
        fid = int(fid)
        pred = pred.split()
        pred = fil(pred)
        preds[fid] = pred
    predicts.close()
    drop()

    re_0001_ = re.compile(r'([^a-zA-Z0-9 ])|([a-z0-9_][A-Z])') # not sure what this is ? vocabulary?

    refs = dict()
    newpreds = dict()
    d = 0
    targets = open('%s/coms.test' % (dataprep), 'r')
    for line in targets:
        (fid, com) = line.split(',')
        fid = int(fid)
        com = com.split()
        com = fil(com)

        try:
            newpreds[fid] = preds[fid]
        except KeyError as ex:
            continue
        
        refs[fid] = [com]

    return newpreds,refs

def bleu_so_far(refs, preds,indicator):
    refs = list(refs.values())
    preds = list(preds.values())
    if indicator is "short":
        Ba = corpus_bleu(refs, preds, (0.33,0.33,0.34,0), smoothing_function=smoothing.method7)
    else:
        Ba = corpus_bleu(refs, preds)
    B1 = corpus_bleu(refs, preds, weights=(1,0,0,0))
    B2 = corpus_bleu(refs, preds, weights=(0,1,0,0))
    B3 = corpus_bleu(refs, preds, weights=(0,0,1,0))
  
    Ba = round(Ba * 100, 2)
    B1 = round(B1 * 100, 2)
    B2 = round(B2 * 100, 2)
    B3 = round(B3 * 100, 2)
    
    ret = ''
    ret += ('for %s functions\n' % (len(preds)))
    ret += ('Ba %s\n' % (Ba))
    ret += ('B1 %s\n' % (B1))
    ret += ('B2 %s\n' % (B2))
    ret += ('B3 %s\n' % (B3))

    if indicator is "long":
        B4 = corpus_bleu(refs, preds, weights=(0,0,0,1))
        B4 = round(B4 * 100, 2)
        ret += ('B4 %s\n' % (B4))

    return ret


def bad_set(refs,preds):
    bset = list()
    gset = dict()
    count = 0
    for i in preds:
        try:
            Bs = round(sentence_bleu(refs[i],preds[i], weights=(0.25,0.25,0.25,0.25), auto_reweigh = True, smoothing_function=smoothing.method7),2)
            gset[i] = Bs
            if Bs <= 0.17:
                bset.append(i)
        except:
            count +=1
    # predictions that cannot be smoothed or accurate blue score calculated
    print("Predictions ignored : %s" % count)
    return bset,gset
    
def partition(refs,preds):  #partitioning predictions less than 4 words in length and corresponding references
    srefs = dict()
    spreds = dict()
    lrefs = dict()
    lpreds = dict()
    for i in preds:
        if len(preds[i]) <= 3:
            srefs[i] = refs[i]
            spreds[i] = preds[i]
        else:
            lrefs[i] = refs[i]
            lpreds[i] =  preds[i]
    return srefs,spreds,lrefs,lpreds

def intersect(preds1,preds2,refs):
    print("For first model")
    bset,gset = bad_set(refs,preds1)
    print("For second model")
    bset2,gset2 = bad_set(refs,preds2)
    bcount1 = len(bset)
    bcount2 = len(bset2)
    print("For the intersection of bad sets")
    improved = list()
    intersect = list(set(bset).intersection(bset2))
    cintersect = len(intersect)
    for fid in bset:
            if fid not in intersect:
                improved.append(fid)

    improvement = dict()
    improvement = {x:gset2[x] for x in improved if x in gset2.keys()} 
    
    print('Number of bad translations in Predict 1 = %s \n' % (bcount1))
    print('Number of bad translations in Predict 2 = %s \n' % (bcount2))
    print('Number of common bad translations = %s \n' % (cintersect))
    print('improved %s \n' % (len(improvement)))
    return improvement, gset, gset2

def re_0002(i):
    # split camel case and remove special characters
    tmp = i.group(0)
    if len(tmp) > 1:
        if tmp.startswith(' '):
            return tmp
        else:
            return '{} {}'.format(tmp[0], tmp[1])
    else:
        return ' '.format(tmp)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('input', type=str, default=None)
    parser.add_argument('input2', type=str, default=None)
    parser.add_argument('--data', dest='dataprep', type=str, default='./data/')  
    parser.add_argument('--outdir', dest='outdir', type=str, default='./data/outdir')
    parser.add_argument('--challenge', action='store_true', default=False)
    parser.add_argument('--obfuscate', action='store_true', default=False)
    parser.add_argument('--sbt', action='store_true', default=False)
    args = parser.parse_args()
    outdir = args.outdir
    dataprep = args.dataprep
    input_file = args.input
    input_file2 =args.input2
    challenge = args.challenge
    obfuscate = args.obfuscate
    sbt = args.sbt

    if challenge:
        dataprep = '../data/challengeset/output'

    if obfuscate:
        dataprep = '../data/obfuscation/output'

    if sbt:
        dataprep = '../data/sbt/output'

    if input_file is None or input_file2 is None:
        print('Please provide an input file to test with --input')
        exit()
    sys.path.append(dataprep)
    preds1,refs = get_pred_ref(input_file)
    preds2,_ = get_pred_ref(input_file2)
    displaylong = " For comments 4 words or longer \n"
    displayshort = "For comments 3 words or shorter \n"
  
    print('final status')
    srefs1,spreds1,lrefs1,lpreds1 = partition(refs,preds1)
    srefs2,spreds2,lrefs2,lpreds2 = partition(refs,preds2)
    print("For model %s" % args.input)
    print(displaylong + bleu_so_far(lrefs1, lpreds1, "long"))
    print(displayshort + bleu_so_far(srefs1, spreds1,"short"))
    print("For model %s" % args.input2)
    print(displaylong + bleu_so_far(lrefs2, lpreds2, "long"))
    print(displayshort + bleu_so_far(srefs2, spreds2,"short"))
    #print(*full_set(refs,newpreds), sep = "\n")
    improved,gset,gset2 = intersect(preds1,preds2,refs)
    m1 = input_file.split("/")[-1].split(".")[0]+"_bleusen"
    m2 = input_file2.split("/")[-1].split(".")[0]+"_bleusen"

    pickle.dump(gset,open(m1,"wb"))
    pickle.dump(gset2,open(m2,"wb"))
    pickle.dump(improved,open("improved.pkl","wb"))


