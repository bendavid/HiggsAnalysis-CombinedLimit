#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules, getrefcount
from optparse import OptionParser

import numpy as np
import h5py
from HiggsAnalysis.CombinedLimit.h5pyutils import writeFlatInChunks, writeSparse
import math
import gc
from collections import OrderedDict


# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
argv.remove( '-b-' )

from root_numpy import hist2array

#don't add histograms to global lists (otherwise memory cannot be freed)
#ROOT.TH1.AddDirectory(False)

from array import array

from HiggsAnalysis.CombinedLimit.DatacardParser import *
from HiggsAnalysis.CombinedLimit.ModelTools import *
from HiggsAnalysis.CombinedLimit.ShapeTools import *
from HiggsAnalysis.CombinedLimit.PhysicsModel import *


parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
addDatacardParserOptions(parser)
parser.add_option("-P", "--physics-model", dest="physModel", default="HiggsAnalysis.CombinedLimit.PhysicsModel:defaultModel",  type="string", help="Physics model to use. It should be in the form (module name):(object name)")
parser.add_option("--PO", "--physics-option", dest="physOpt", default=[],  type="string", action="append", help="Pass a given option to the physics model (can specify multiple times)")
parser.add_option("", "--dump-datacard", dest="dumpCard", default=False, action='store_true',  help="Print to screen the DataCard as a python config and exit")
parser.add_option("","--allowNegativeExpectation", default=False, action='store_true', help="allow negative expectation")
parser.add_option("","--maskedChan", default=[], type="string",action="append", help="channels to be masked in likelihood but propagated through for later storage/analysis")
parser.add_option("-S","--doSystematics", type=int, default=1, help="enable systematics")
parser.add_option("","--chunkSize", type=int, default=4*1024**2, help="chunk size for hd5fs storage")
parser.add_option("", "--sparse", default=False, action='store_true',  help="Store normalization and systematics arrays as sparse tensors")
parser.add_option("", "--scaleMaskedYields", type=float, default=1.,  help="Scaling factor for yields in masked channels")
parser.add_option("", "--postfix", default="",type="string", help="add _<postfix> to output hdf5 file")
parser.add_option("", "--clipSystVariations", type=float, default=-1.,  help="Clipping of syst variations (all processes)")
parser.add_option("", "--clipSystVariationsSignal", type=float, default=-1.,  help="Clipping of syst variations (signal processes)")
(options, args) = parser.parse_args()

if len(args) == 0:
    parser.print_usage()
    exit(1)

options.fileName = args[0]


if not options.fileName.endswith(".pkl"):
    
    if options.fileName.endswith(".gz"):
        import gzip
        file = gzip.open(options.fileName, "rb")
        options.fileName = options.fileName[:-3]
    else:
        file = open(options.fileName, "r")
    ## Parse text file 
    DC = parseCard(file, options)

else:
    file = open(options.fileName, "rb")
    import pickle
    DC = pickle.load(file)

if options.dumpCard:
    DC.print_structure()
    exit()

print(options)

nproc = len(DC.processes)
nsignals = len(DC.signals)

dtype = 'float64'

MB = ShapeBuilder(DC, options)

#list of processes, signals first
procs = []
for proc in DC.processes:
  if DC.isSignal[proc]:
    procs.append(proc)

for proc in DC.processes:
  if not DC.isSignal[proc]:
    procs.append(proc)    

#list of signals preserving datacard order
signals = []
for proc in DC.processes:
  if DC.isSignal[proc]:
    signals.append(proc)
      
#list of systematic uncertainties (nuisances)
systsd = OrderedDict()
systs = []
systsnoprofile = []
systsnoconstraint = []
if options.doSystematics:
  for syst in DC.systs:
    if not 'NoProfile' in syst[2]:
      systsd[syst[0]] = syst
      systs.append(syst[0])
  for syst in DC.systs:
    if 'NoProfile' in syst[2]:
      systsd[syst[0]] = syst
      systs.append(syst[0])
      systsnoprofile.append(syst[0])
    if 'NoConstraint' in syst[2]:
        systsnoconstraint.append(syst[0])

nsyst = len(systs)

constraintweights = np.ones([nsyst],dtype=dtype)
for syst in systsnoconstraint:
    constraintweights[systs.index(syst)] = 0.
  
#list of groups of systematics (nuisances) and lists of indexes
systgroups = []
systgroupidxs = []
if options.doSystematics:
  for group in DC.groups:
    systgroups.append(group)
    systgroupidx = []
    for syst in DC.groups[group]:
      systgroupidx.append(systs.index(syst))
    systgroupidxs.append(systgroupidx)
    
#list of groups of signal processes by charge
chargegroups = []
chargegroupidxs = []
for group in DC.chargeGroups:
  chargegroups.append(group)
  chargegroupidx = []
  for proc in DC.chargeGroups[group]:
    chargegroupidx.append(procs.index(proc))
  chargegroupidxs.append(chargegroupidx)

#list of groups of signal processes by polarization
polgroups = []
polgroupidxs = []
for group in DC.polGroups:
  polgroups.append(group)
  polgroupidx = []
  for proc in DC.polGroups[group]:
    polgroupidx.append(procs.index(proc))
  polgroupidxs.append(polgroupidx)


#list of groups of signal processes by helicity xsec
helgroups = []
helgroupidxs = []
for group in DC.helGroups:
  helgroups.append(group)
  helgroupidx = []
  for proc in DC.helGroups[group]:
    helgroupidx.append(procs.index(proc))
  helgroupidxs.append(helgroupidx)


#list of groups of signal processes to be summed
sumgroups = []
sumgroupsegmentids = []
sumgroupidxs = []
for igroup,group in enumerate(DC.sumGroups):
  sumgroups.append(group)
  for proc in DC.sumGroups[group]:
    sumgroupsegmentids.append(igroup)
    sumgroupidxs.append(procs.index(proc))
    
#list of groups of signal processes by chargemeta
chargemetagroups = []
chargemetagroupidxs = []
for group in DC.chargeMetaGroups:
  chargemetagroups.append(group)
  chargemetagroupidx = []
  for proc in DC.chargeMetaGroups[group]:
    chargemetagroupidx.append(sumgroups.index(proc))
  chargemetagroupidxs.append(chargemetagroupidx)
  
#list of groups of signal processes by ratiometa
ratiometagroups = []
ratiometagroupidxs = []
for group in DC.ratioMetaGroups:
  ratiometagroups.append(group)
  ratiometagroupidx = []
  for proc in DC.ratioMetaGroups[group]:
    ratiometagroupidx.append(sumgroups.index(proc))
  ratiometagroupidxs.append(ratiometagroupidx)

#list of groups of signal processes by helmeta
helmetagroups = []
helmetagroupidxs = []
for group in DC.helMetaGroups:
  print group, "group"
  helmetagroups.append(group)
  helmetagroupidx = []
  for proc in DC.helMetaGroups[group]:
    print proc, "in group"
    helmetagroupidx.append(sumgroups.index(proc))
  helmetagroupidxs.append(helmetagroupidx)

print len(helmetagroupidxs)
    
#list of groups of signal processes for regularization
reggroups = []
reggroupidxs = []
for igroup,group in enumerate(DC.regGroups):
  reggroups.append(group)
  reggroupidx = []
  for proc in DC.regGroups[group]:
    reggroupidx.append(procs.index(proc))
  reggroupidxs.append(reggroupidx)
  
poly1dreggroups = []
poly1dreggroupfirstorder = []
poly1dreggrouplastorder = []
poly1dreggroupnames = []
poly1dreggroupbincenters = []
for group in DC.poly1DRegGroups:
  poly1dreggroups.append(group)
  poly1dreggroupfirstorder.append(DC.poly1DRegGroups[group]["firstorder"])
  poly1dreggrouplastorder.append(DC.poly1DRegGroups[group]["lastorder"])
  poly1dreggroupnames.append(DC.poly1DRegGroups[group]["names"])
  poly1dreggroupbincenters.append(DC.poly1DRegGroups[group]["bincenters"])
  
poly2dreggroups = []
poly2dreggroupfirstorder = []
poly2dreggrouplastorder = []
poly2dreggroupfullorder = []
poly2dreggroupnames = []
poly2dreggroupbincenters0 = []
poly2dreggroupbincenters1 = []
for group in DC.poly2DRegGroups:
  poly2dreggroups.append(group)
  poly2dreggroupfirstorder.append(DC.poly2DRegGroups[group]["firstorder"])
  poly2dreggrouplastorder.append(DC.poly2DRegGroups[group]["lastorder"])
  poly2dreggroupfullorder.append(DC.poly2DRegGroups[group]["fullorder"])
  poly2dreggroupnames.append(DC.poly2DRegGroups[group]["names"])
  bincenters0 = []
  bincenters1 = []
  for bincenter in DC.poly2DRegGroups[group]["bincenters"]:
    bincenters0.append(bincenter[0])
    bincenters1.append(bincenter[1])
  poly2dreggroupbincenters0.append(bincenters0)
  poly2dreggroupbincenters1.append(bincenters1)
  
#list of groups of systematics to be treated as additional outputs for impacts, etc (aka "nuisances of interest")
noigroups = []
noigroupidxs = []
for group in DC.noiGroups:
  noigroups.append(group)
  for syst in DC.noiGroups[group]:
    noigroupidxs.append(systs.index(syst))

#list of channels, ordered such that masked channels are last
chans = []
for chan in DC.bins:
  if not chan in options.maskedChan:
    chans.append(chan)

maskedchans = []
for chan in DC.bins:
  if chan in options.maskedChan:
    chans.append(chan)
    maskedchans.append(chan)
    
if not len(maskedchans) == len(options.maskedChan):
  raise Exception("Some of the specified masked channels do not actually exist")

#first loop through observed data to determine the total number of bins
nbinsfull = 0
nbins = 0
for chan in chans:
  if chan in options.maskedChan:
    nbinschan = 1
  else:
    #print chan, "looking at this channel"
    data_obs_chan_hist = MB.getShape(chan,"data_obs")
    #exclude overflow/underflow bins
    nbinschan = data_obs_chan_hist.GetNbinsX()*data_obs_chan_hist.GetNbinsY() * data_obs_chan_hist.GetNbinsZ()
    nbins += nbinschan
  
  nbinsfull += nbinschan
  
print("nbins = %d, nbinsfull = %d, nproc = %d, nsyst = %d" % (nbins,nbinsfull,nproc,nsyst))





#fill data, expected yields, and kappas into HDF5 file (with chunked storage and compression)

#fill data, expected yields with information about statistical uncertainties, and kappas into numpy arrays

#n.b data and expected have shape [nbins]
#sumw and sumw2 keep track of total nominal statistical uncertainty per bin and have shape [nbins]

#norm has shape [nbinsfull, nproc] and keeps track of expected normalization

#logk has shape [nbinsfull, nproc, 2, nsyst] and keep track of systematic variations
#per nuisance-parameter, per-process, per-bin
#the second-last dimension, of size 2, indexes "logkavg" and "logkhalfdiff" for asymmetric uncertainties
#where logkavg = 0.5*(logkup + logkdown) and logkhalfdiff = 0.5*(logkup - logkdown)

#n.b, in case of masked channels, nbinsfull includes the masked channels where nbins does not

nentries = 0

data_obs = np.zeros([nbins], dtype)
sumw = np.zeros([nbins], dtype)
sumw2 = np.zeros([nbins], dtype)

if options.sparse:
  maxsparseidx = max(nbinsfull*nproc,2*nsyst)
  
  idxdtype = 'int32'
  
  if maxsparseidx > np.iinfo(idxdtype).max:
    print("sparse array shapes are too large for index datatype, switching to int64")
    idxdtype = 'int64'
  
  norm_sparse_size = 0
  norm_sparse_indices = np.zeros([norm_sparse_size,2],idxdtype)
  norm_sparse_values = np.zeros([norm_sparse_size],dtype)
  
  logk_sparse_size = 0
  logk_sparse_normindices = np.zeros([logk_sparse_size,1],idxdtype)
  logk_sparse_systindices = np.zeros([logk_sparse_size,1],idxdtype)
  logk_sparse_values = np.zeros([logk_sparse_size],dtype)
else:
  norm = np.zeros([nbinsfull,nproc], dtype)
  logk = np.zeros([nbinsfull,nproc,2,nsyst], dtype)

#fill data_obs, norm, and logk
#numerical cutoff in case of zeros in systematic variations
logkepsilon = math.log(1e-3)
#counter to keep track of current bin being written
ibin = 0
for chan in chans:
  
  if not chan in options.maskedChan:
    #get histogram, convert to np array with desired type, and exclude underflow/overflow bins
    data_obs_chan_hist = MB.getShape(chan,"data_obs")
    data_obs_chan = hist2array(data_obs_chan_hist, include_overflow=False).ravel().astype(dtype)
    data_obs_chan_hist.Delete()
    nbinschan = data_obs_chan.shape[0]
    #write to output array
    data_obs[ibin:ibin+nbinschan] = data_obs_chan
    data_obs_chan = None
  else:
    nbinschan = 1
  
  expchan = DC.exp[chan]
  for iproc,proc in enumerate(procs):
    
    hasproc = proc in expchan
    
    if not hasproc:
      continue
    
    #get histogram, convert to np array with desired type, and exclude underflow/overflow bins
    norm_chan_hist = MB.getShape(chan,proc)
    norm_chan = hist2array(norm_chan_hist, include_overflow=False).ravel().astype(dtype)
    if not chan in options.maskedChan:
      if (norm_chan_hist.GetSumw2().GetSize()>0):
        #print("proper uncertainties")
        sumw2_chan_hist = norm_chan_hist.Clone()
        if sumw2_chan_hist.InheritsFrom('TH1F'):
            #work around for the fact GetSumw2 returns a TArrayD but TH1F expects a Float_t *
            from array import array
            sumw2f=[sumw2_chan_hist.GetSumw2()[i] for i in range(sumw2_chan_hist.GetSumw2().GetSize())]
            sumw2f = array('f',sumw2f)
            sumw2_chan_hist.Set(sumw2_chan_hist.GetSumw2().GetSize(), sumw2f)
        else:
            sumw2_chan_hist.Set(sumw2_chan_hist.GetSumw2().GetSize(), sumw2_chan_hist.GetSumw2().GetArray())
        sumw2_chan = hist2array(sumw2_chan_hist, include_overflow=False).ravel().astype(dtype)
        sumw2_chan_hist.Delete()
      else:
        print("Warning: Sumw2 not filled for histograms, using fallback method for computing uncertainties, binByBin uncertainties for templates may not be reliable")
        print(proc, "is problematic")
        print(norm_chan_hist.GetSumw2().GetSize())
        nentries_chan = (norm_chan_hist.GetEntries()/norm_chan_hist.GetSumOfWeights())*norm_chan
        sumw2_chan = nentries_chan*np.square(norm_chan/nentries_chan)
        nentries_chan = None
    norm_chan_hist.Delete()
    
    if norm_chan.shape[0] != nbinschan:
      raise Exception("Mismatch between number of bins in channel for data and template")
    
    if not options.allowNegativeExpectation:
      norm_chan = np.maximum(norm_chan,0.)

    if not chan in options.maskedChan:
      sumw[ibin:ibin+nbinschan] += norm_chan
      sumw2[ibin:ibin+nbinschan] += sumw2_chan
      sumw2chan = None
    
    if chan in options.maskedChan:
      norm_chan_scaled = options.scaleMaskedYields*norm_chan
    else:
      norm_chan_scaled = norm_chan
    
    
    if options.sparse:
      norm_chan_indices = np.transpose(np.nonzero(norm_chan_scaled))
      norm_chan_values = np.reshape(norm_chan_scaled[norm_chan_indices],[-1])
            
      nvals_chan = len(norm_chan_values)
      oldlength = norm_sparse_size
      norm_sparse_size = oldlength + nvals_chan
      norm_sparse_indices.resize([norm_sparse_size,2])
      norm_sparse_values.resize([norm_sparse_size])
      
      out_indices = np.array([[ibin,iproc]]) + np.pad(norm_chan_indices,((0,0),(0,1)),'constant')
      norm_chan_indices = None
      
      norm_sparse_indices[oldlength:norm_sparse_size] = out_indices
      out_indices = None
      
      norm_sparse_values[oldlength:norm_sparse_size] = norm_chan_values
      norm_chan_values = None
      
      norm_chan_idx_map = np.cumsum(np.not_equal(norm_chan_scaled,0.)) - 1 + oldlength

    else:
      #write to (dense) output array
      norm[ibin:ibin+nbinschan,iproc] = norm_chan_scaled
      
    norm_chan_scaled = None
    
    for isyst,(name,syst) in enumerate(systsd.items()):
      stype = syst[2]
        
      if stype in ['lnN','lnNNoProfile','lnNNoConstraint']:
        ksyst = syst[4][chan][proc]
        if type(ksyst) is list:
          ksystup = ksyst[1]
          ksystdown = ksyst[0]
          if ksystup == 0. and ksystdown==0.:
            continue
          if ksystup == 0.:
            ksystup = 1.
          if ksystdown == 0.:
            ksystdown = 1.
          logkup_chan = math.log(ksystup)*np.ones([nbinschan],dtype=dtype)
          logkdown_chan = -math.log(ksystdown)*np.ones([nbinschan],dtype=dtype)
          logkavg_chan = 0.5*(logkup_chan + logkdown_chan)
          logkhalfdiff_chan = 0.5*(logkup_chan - logkdown_chan)
          logkup_chan = None
          logkdown_chan = None
        else:
          if ksyst == 0.:
            continue
          logkavg_chan = math.log(ksyst)*np.ones([nbinschan],dtype=dtype)
          logkhalfdiff_chan = np.zeros([nbinschan],dtype=dtype)
      elif 'shape' in stype:
        kfac = syst[4][chan][proc]
        
        if kfac == 0.:
          continue
        
        systup_chan_hist = MB.getShape(chan,proc,name+"Up")
        systup_chan = hist2array(systup_chan_hist, include_overflow=False).ravel().astype(dtype)
        systup_chan_hist.Delete()
        if systup_chan.shape[0] != nbinschan:
          raise Exception("Mismatch between number of bins in channel for data and systematic variation template")
        logkup_chan = kfac*np.log(systup_chan/norm_chan)
        logkup_chan = np.where(np.equal(np.sign(norm_chan*systup_chan),1), logkup_chan, logkepsilon*np.ones_like(logkup_chan))
        systup_chan = None
        
        systdown_chan_hist = MB.getShape(chan,proc,name+"Down")
        systdown_chan = hist2array(systdown_chan_hist, include_overflow=False).ravel().astype(dtype)
        systdown_chan_hist.Delete()
        if systdown_chan.shape[0] != nbinschan:
          raise Exception("Mismatch between number of bins in channel for data and systematic variation template")
        logkdown_chan = -kfac*np.log(systdown_chan/norm_chan)
        logkdown_chan = np.where(np.equal(np.sign(norm_chan*systdown_chan),1), logkdown_chan, -logkepsilon*np.ones_like(logkdown_chan))
        systdown_chan = None
        
        if options.clipSystVariations>0.:
          cliplow = -np.abs(np.log(options.clipSystVariations))
          cliphigh = np.abs(np.log(options.clipSystVariations))
          logkup_chan = np.clip(logkup_chan,cliplow,cliphigh)
          logkdown_chan = np.clip(logkdown_chan,cliplow,cliphigh)
          
        if options.clipSystVariationsSignal>0. and proc in signals:
          cliplow = -np.abs(np.log(options.clipSystVariationsSignal))
          cliphigh = np.abs(np.log(options.clipSystVariationsSignal))
          logkup_chan = np.clip(logkup_chan,cliplow,cliphigh)
          logkdown_chan = np.clip(logkdown_chan,cliplow,cliphigh)
        
        
        #checks for outliers, one-sided uncertainties, etc
        
        ##check for one-sided uncertainties
        ##onesided = np.all(np.less_equal(np.sign(logkup_chan*logkdown_chan),0))
        #onesided = np.any(np.less(np.sign(logkup_chan*logkdown_chan),0)) and 'Mu' not in name and 'mu' not in name
        ##print("%s,%s,%s" % (chan,proc,name))
        ##print(np.sign(logkup_chan*logkdown_chan))
        ##if onesided:
          ##print("Warning: One-sided uncertainty present for %s,%s,%s" % (chan,proc,name))
          ##print(np.sign(logkup_chan*logkdown_chan))
          ##print(norm_chan)
          ##print(norm_chan*np.less(np.sign(logkup_chan*logkdown_chan),0))
          ##print(logkup_chan)
          ##print(logkdown_chan)
        
        #populated = norm_chan/np.sum(norm_chan) > 1e-5
        #outlierup = populated*(np.abs(logkup_chan)>0.5)
        #outlierdown = populated*(np.abs(logkdown_chan)>0.5)
        ###outlier = np.any(np.abs(logkup_chan)>0.5) or np.any(np.abs(logkdown_chan)>0.5)
        #outlier = np.any(outlierup) or np.any(outlierdown)
        #maxvar = max(np.max(populated*np.abs(logkup_chan)),np.max(populated*np.abs(logkup_chan)))
        #if outlier:
          #print("Warning: outlier with abs(lnk)=%f present for %s,%s,%s" % (maxvar,chan,proc,name))
        
        logkavg_chan = 0.5*(logkup_chan + logkdown_chan)
        logkhalfdiff_chan = 0.5*(logkup_chan - logkdown_chan)
        logkup_chan = None
        logkdown_chan = None
          
      #ensure that systematic tensor is sparse where normalization matrix is sparse
      logkavg_chan = np.where(np.equal(norm_chan,0.), 0., logkavg_chan)
      logkhalfdiff_chan = np.where(np.equal(norm_chan,0.), 0., logkhalfdiff_chan)

      if options.sparse:
        logkavg_chan_indices = np.transpose(np.nonzero(logkavg_chan))
        logkavg_chan_values = np.reshape(logkavg_chan[logkavg_chan_indices],[-1])
        
        nvals_chan = len(logkavg_chan_values)
        oldlength = logk_sparse_size
        logk_sparse_size = oldlength + nvals_chan
        logk_sparse_normindices.resize([logk_sparse_size,1])
        logk_sparse_systindices.resize([logk_sparse_size,1])
        logk_sparse_values.resize([logk_sparse_size])
                
        #first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
        #second dimension is flattened in the [2,nsyst] space, where logkavg corresponds to [0,isyst] flattened to isyst
        #two dimensions are kept in separate arrays for now to reduce the number of copies needed later
        out_normindices = norm_chan_idx_map[logkavg_chan_indices]
        logkavg_chan_indices = None
        
        logk_sparse_normindices[oldlength:logk_sparse_size] = out_normindices
        logk_sparse_systindices[oldlength:logk_sparse_size] = isyst
        out_normindices = None
        
        logk_sparse_values[oldlength:logk_sparse_size] = logkavg_chan_values
        logkavg_chan_values = None
        
        logkhalfdiff_chan_indices = np.transpose(np.nonzero(logkhalfdiff_chan))
        logkhalfdiff_chan_values = np.reshape(logkhalfdiff_chan[logkhalfdiff_chan_indices],[-1])
                
        nvals_chan = len(logkhalfdiff_chan_values)
        oldlength = logk_sparse_size
        logk_sparse_size = oldlength + nvals_chan
        logk_sparse_normindices.resize([logk_sparse_size,1])
        logk_sparse_systindices.resize([logk_sparse_size,1])
        logk_sparse_values.resize([logk_sparse_size])
                
        #out_indices = np.array([[ibin,iproc,isyst,1]]) + np.pad(logkhalfdiff_chan_indices,((0,0),(0,3)),'constant')
        #first dimension of output indices are NOT in the dense [nbin,nproc] space, but rather refer to indices in the norm_sparse vectors
        #second dimension is flattened in the [2,nsyst] space, where logkhalfdiff corresponds to [1,isyst] flattened to nsyst + isyst
        #two dimensions are kept in separate arrays for now to reduce the number of copies needed later
        out_normindices = norm_chan_idx_map[logkhalfdiff_chan_indices]
        logkhalfdiff_chan_indices = None
        
        logk_sparse_normindices[oldlength:logk_sparse_size] = out_normindices
        logk_sparse_systindices[oldlength:logk_sparse_size] = nsyst + isyst
        out_normindices = None
        
        logk_sparse_values[oldlength:logk_sparse_size] = logkhalfdiff_chan_values
        logkhalfdiff_chan_values = None        
      else:
        #write to dense output array
        logk[ibin:ibin+nbinschan,iproc,0,isyst] = logkavg_chan
        logk[ibin:ibin+nbinschan,iproc,1,isyst] = logkhalfdiff_chan
            
      #free memory
      logkavg_chan = None
      logkhalfdiff_chan = None
    
    #free memory
    norm_chan = None
    norm_chan_idx_map = None
    
  
  #increment counter
  ibin += nbinschan



#resize and sort sparse arrays into canonical order
if options.sparse:
  #resize sparse arrays to actual length
  norm_sparse_indices.resize([norm_sparse_size,2])
  norm_sparse_values.resize([norm_sparse_size])
  logk_sparse_normindices.resize([logk_sparse_size,1])
  logk_sparse_systindices.resize([logk_sparse_size,1])
  logk_sparse_values.resize([logk_sparse_size])
  
  #straightforward sorting of norm_sparse into canonical order
  norm_sparse_dense_shape = (nbinsfull, nproc)
  norm_sort_indices = np.argsort(np.ravel_multi_index(np.transpose(norm_sparse_indices),norm_sparse_dense_shape))
  norm_sparse_indices = norm_sparse_indices[norm_sort_indices]
  norm_sparse_values = norm_sparse_values[norm_sort_indices]
    
  #now permute the indices of the first dimension of logk_sparse corresponding to the resorting of norm_sparse
  
  #compute the inverse permutation from the sorting of norm_sparse
  #since the final indices are filled from here, need to ensure it has the correct data type
  logk_permute_indices = np.argsort(norm_sort_indices).astype(idxdtype)
  norm_sort_indices = None
  logk_sparse_normindices = logk_permute_indices[logk_sparse_normindices]
  logk_permute_indices = None
  logk_sparse_indices = np.concatenate([logk_sparse_normindices, logk_sparse_systindices],axis=-1)
  logk_normindices = None

  #now straightforward sorting of logk_sparse into canonical order
  logk_sparse_dense_shape = (norm_sparse_indices.shape[0], 2*nsyst)
  logk_sort_indices = np.argsort(np.ravel_multi_index(np.transpose(logk_sparse_indices),logk_sparse_dense_shape))
  logk_sparse_indices = logk_sparse_indices[logk_sort_indices]
  logk_sparse_values = logk_sparse_values[logk_sort_indices]
  logk_sort_indices = None

#compute poisson parameter for Barlow-Beeston bin-by-bin statistical uncertainties
kstat = np.square(sumw)/sumw2
#numerical protection to avoid poorly defined constraint
kstat = np.where(np.equal(sumw,0.), 1., kstat)

#write results to hdf5 file

procSize = nproc*np.dtype(dtype).itemsize
systSize = 2*nsyst*np.dtype(dtype).itemsize
chunkSize = np.amax([options.chunkSize,procSize,systSize])
if chunkSize > options.chunkSize:
  print("Warning: Maximum chunk size in bytes was increased from %d to %d to align with tensor sizes and allow more efficient reading/writing." % (options.chunkSize, chunkSize))

#create HDF5 file (chunk cache set to the chunk size since we can guarantee fully aligned writes
outfilename = options.out.replace('.root','.hdf5')
if options.sparse:
  outfilename = outfilename.replace('.hdf5','_sparse.hdf5')
if options.postfix:
    outfilename = outfilename.replace('.hdf5','_{pf}.hdf5'.format(pf=options.postfix))
f = h5py.File(outfilename, rdcc_nbytes=chunkSize, mode='w')

#save some lists of strings to the file for later use
hprocs = f.create_dataset("hprocs", [len(procs)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hprocs[...] = procs

hsignals = f.create_dataset("hsignals", [len(signals)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hsignals[...] = signals

hsysts = f.create_dataset("hsysts", [len(systs)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hsysts[...] = systs

hsystsnoprofile = f.create_dataset("hsystsnoprofile", [len(systsnoprofile)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hsystsnoprofile[...] = systsnoprofile

hsystgroups = f.create_dataset("hsystgroups", [len(systgroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hsystgroups[...] = systgroups

hsystgroupidxs = f.create_dataset("hsystgroupidxs", [len(systgroupidxs)], dtype=h5py.special_dtype(vlen=np.dtype('int32')), compression="gzip")
hsystgroupidxs[...] = systgroupidxs

hchargegroups = f.create_dataset("hchargegroups", [len(chargegroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hchargegroups[...] = chargegroups

hchargegroupidxs = f.create_dataset("hchargegroupidxs", [len(chargegroups),2], dtype='int32', compression="gzip")
hchargegroupidxs[...] = chargegroupidxs

hpolgroups = f.create_dataset("hpolgroups", [len(polgroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hpolgroups[...] = polgroups

hpolgroupidxs = f.create_dataset("hpolgroupidxs", [len(polgroups),3], dtype='int32', compression="gzip")
hpolgroupidxs[...] = polgroupidxs

hhelgroups = f.create_dataset("hhelgroups", [len(helgroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hhelgroups[...] = helgroups

hhelgroupidxs = f.create_dataset("hhelgroupidxs", [len(helgroups),6], dtype='int32', compression="gzip")
hhelgroupidxs[...] = helgroupidxs

hsumgroups = f.create_dataset("hsumgroups", [len(sumgroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hsumgroups[...] = sumgroups

hsumgroupsegmentids = f.create_dataset("hsumgroupsegmentids", [len(sumgroupsegmentids)], dtype='int32', compression="gzip")
hsumgroupsegmentids[...] = sumgroupsegmentids

hsumgroupidxs = f.create_dataset("hsumgroupidxs", [len(sumgroupidxs)], dtype='int32', compression="gzip")
hsumgroupidxs[...] = sumgroupidxs

hchargemetagroups = f.create_dataset("hchargemetagroups", [len(chargemetagroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hchargemetagroups[...] = chargemetagroups

hchargemetagroupidxs = f.create_dataset("hchargemetagroupidxs", [len(chargemetagroups),2], dtype='int32', compression="gzip")
hchargemetagroupidxs[...] = chargemetagroupidxs

hratiometagroups = f.create_dataset("hratiometagroups", [len(ratiometagroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hratiometagroups[...] = ratiometagroups

hratiometagroupidxs = f.create_dataset("hratiometagroupidxs", [len(ratiometagroups),2], dtype='int32', compression="gzip")
hratiometagroupidxs[...] = ratiometagroupidxs

hhelmetagroups = f.create_dataset("hhelmetagroups", [len(helmetagroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hhelmetagroups[...] = helmetagroups

hhelmetagroupidxs = f.create_dataset("hhelmetagroupidxs", [len(helmetagroups),6], dtype='int32', compression="gzip")
hhelmetagroupidxs[...] = helmetagroupidxs

hreggroups = f.create_dataset("hreggroups", [len(reggroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hreggroups[...] = reggroups

hreggroupidxs = f.create_dataset("hreggroupidxs", [len(reggroupidxs)], dtype=h5py.special_dtype(vlen=np.dtype('int32')), compression="gzip")
hreggroupidxs[...] = reggroupidxs

hpoly1dreggroups = f.create_dataset("hpoly1dreggroups", [len(poly1dreggroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hpoly1dreggroups[...] = poly1dreggroups

hpoly1dreggroupfirstorder = f.create_dataset("hpoly1dreggroupfirstorder", [len(poly1dreggroupfirstorder)], dtype='int32', compression="gzip")
hpoly1dreggroupfirstorder[...] = poly1dreggroupfirstorder

hpoly1dreggrouplastorder = f.create_dataset("hpoly1dreggrouplastorder", [len(poly1dreggrouplastorder)], dtype='int32', compression="gzip")
hpoly1dreggrouplastorder[...] = poly1dreggrouplastorder

hpoly1dreggroupnames = f.create_dataset("hpoly1dreggroupnames", [len(poly1dreggroupnames)], dtype=h5py.special_dtype(vlen="S256"), compression="gzip")
hpoly1dreggroupnames[...] = poly1dreggroupnames

hpoly1dreggroupbincenters = f.create_dataset("hpoly1dreggroupbincenters", [len(poly1dreggroupbincenters)], dtype=h5py.special_dtype(vlen=np.dtype('float64')), compression="gzip")
hpoly1dreggroupbincenters[...] = poly1dreggroupbincenters

hpoly2dreggroups = f.create_dataset("hpoly2dreggroups", [len(poly2dreggroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hpoly2dreggroups[...] = poly2dreggroups

hpoly2dreggroupfirstorder = f.create_dataset("hpoly2dreggroupfirstorder", [len(poly2dreggroupfirstorder),2], dtype='int32', compression="gzip")
hpoly2dreggroupfirstorder[...] = poly2dreggroupfirstorder

hpoly2dreggrouplastorder = f.create_dataset("hpoly2dreggrouplastorder", [len(poly2dreggrouplastorder),2], dtype='int32', compression="gzip")
hpoly2dreggrouplastorder[...] = poly2dreggrouplastorder

hpoly2dreggroupfullorder = f.create_dataset("hpoly2dreggroupfullorder", [len(poly2dreggroupfullorder),2], dtype='int32', compression="gzip")
hpoly2dreggroupfullorder[...] = poly2dreggroupfullorder

hpoly2dreggroupnames = f.create_dataset("hpoly2dreggroupnames", [len(poly2dreggroupnames)], dtype=h5py.special_dtype(vlen="S256"), compression="gzip")
hpoly2dreggroupnames[...] = poly2dreggroupnames

hpoly2dreggroupbincenters0 = f.create_dataset("hpoly2dreggroupbincenters0", [len(poly2dreggroupbincenters0)], dtype=h5py.special_dtype(vlen=np.dtype('float64')), compression="gzip")
hpoly2dreggroupbincenters0[...] = poly2dreggroupbincenters0

hpoly2dreggroupbincenters1 = f.create_dataset("hpoly2dreggroupbincenters1", [len(poly2dreggroupbincenters1)], dtype=h5py.special_dtype(vlen=np.dtype('float64')), compression="gzip")
hpoly2dreggroupbincenters1[...] = poly2dreggroupbincenters1

hnoigroups = f.create_dataset("hnoigroups", [len(noigroups)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hnoigroups[...] = noigroups

hnoigroupidxs = f.create_dataset("hnoigroupidxs", [len(noigroupidxs)], dtype='int32', compression="gzip")
hnoigroupidxs[...] = noigroupidxs

hmaskedchans = f.create_dataset("hmaskedchans", [len(maskedchans)], dtype=h5py.special_dtype(vlen=str), compression="gzip")
hmaskedchans[...] = maskedchans

#create h5py datasets with optimized chunk shapes
nbytes = 0

nbytes += writeFlatInChunks(constraintweights, f, "hconstraintweights", maxChunkBytes = chunkSize)
constraintweights = None

nbytes += writeFlatInChunks(data_obs, f, "hdata_obs", maxChunkBytes = chunkSize)
data_obs = None

nbytes += writeFlatInChunks(kstat, f, "hkstat", maxChunkBytes = chunkSize)
kstat = None

if options.sparse:
  nbytes += writeSparse(norm_sparse_indices, norm_sparse_values, norm_sparse_dense_shape, f, "hnorm_sparse", maxChunkBytes = chunkSize)
  norm_sparse_indices = None
  norm_sparse_values = None
  nbytes += writeSparse(logk_sparse_indices, logk_sparse_values, logk_sparse_dense_shape, f, "hlogk_sparse", maxChunkBytes = chunkSize)
  logk_sparse_indices = None
  logk_sparse_values = None

else:
  nbytes += writeFlatInChunks(norm, f, "hnorm", maxChunkBytes = chunkSize)
  norm = None
  nbytes += writeFlatInChunks(logk, f, "hlogk", maxChunkBytes = chunkSize)
  logk = None

print("Total raw bytes in arrays = %d" % nbytes)
