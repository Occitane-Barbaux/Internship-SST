### packages
import sys,os

import xarray as xr
import pandas as pd
import numpy as np

import NSSEA as ns
from NSSEA.__tools import matrix_positive_part
from NSSEA.__tools import matrix_squareroot
import scipy.optimize as sco

import re
from itertools import product

from cmdstanpy import CmdStanModel
import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True #No long log


##Functions
def load_obs(pathObs):
	## Load Observations
	dXo = xr.open_dataset(os.path.join( pathObs,"Xo.nc")) #Deja en anomalies
	Xo  = pd.DataFrame( dXo.tas.values.squeeze() , columns = ["Xo"] , index = dXo.time["time.year"].values )

	dYo = xr.open_dataset(os.path.join(pathObs,"Yo.nc"))
	Yo  = pd.DataFrame( dYo.TX.values.squeeze() , columns = ["Yo"] , index = dYo.time["time.year"].values )
	return Xo,Yo #en celsius
	
def import_gcm_multiscenario(pathGCM_X_separated,pathGCM_Y_separated,scenarios=['ssp585']):
    #Maj to ignore directory, only look at files
    ## List of models X
    pathInpX= pathGCM_X_separated
    modelsX = [  "_".join(f.split("/")[-1][:-3].split("_")[:3]) for f in os.listdir(pathInpX) if os.path.isfile(pathInpX+"/"+f) ] #Modele+type+scenario
    modelsX.sort()


    ## List of models Y
    pathInpY= pathGCM_Y_separated
    modelsY =  [  "_".join(f.split("/")[-1][:-3].split("_")[:3]) for f in os.listdir(pathInpY) if os.path.isfile(pathInpY+"/"+f)  ] #Modele+type+scenario
    modelsY.sort()
    models = list(set(modelsX) & set(modelsY)) #Conserve si meme Modele+type+scenario pour tas et tasmax
    models.sort()
    ## Load X and Y
    lX = []
    lY = []
    for m in models:
        model, typeRun, scen=m.split("_")

        ## Load X
        
        df   = xr.open_dataset( os.path.join( pathInpX , "{}_tas_YearMean_Europe.nc".format(m) )  ,decode_times=False )
        time = df.time.values.astype(int)
        X    = pd.DataFrame( df.tas.values.ravel() , columns = [m] , index = time )
        X_miss=X#correct_miss(X)
        X=pd.DataFrame( np.array([X_miss.values.ravel(),
          np.array([model for i in range(len(time))]),
          np.array([typeRun for i in range(len(time))]),
          np.array([scen for i in range(len(time))])]).T, 
          columns = ['tas',"Model", "typeRun", "scenario"] , index = time )
        X['tas']=X['tas'].astype(float)
        lX.append(  X)
    
        ## Load Y
        df   = xr.open_dataset( os.path.join( pathInpY , "{}_tasmax_YearMax.nc".format(m) ) ,decode_times=False  )
        time = df.time.values.astype(int)
        Y    = pd.DataFrame( df.tasmax.values.ravel() , columns = [m] , index = time )
        Y_miss=Y#correct_miss(Y)
        Y=pd.DataFrame( np.array([Y_miss.values.ravel(),
          np.array([model+"_"+typeRun for i in range(len(time))]),
          np.array([scen for i in range(len(time))])]).T, 
          columns = ['tasmax',"Model", "scenario"] , index = time )
        Y['tasmax']=Y['tasmax'].astype(float)
        lY.append( Y)
        
        
    lX_out = pd.concat(lX)
    lX_out.index.name= 'Year'
    lY_out = pd.concat(lY)
    lY_out.index.name= 'Year'
    return lX_out,lY_out,models

def Multi_up_to_FC_Gam(pathOut,lX_out,models,event,time_period,n_sample, ns_law,dt_string,scenarios=['ssp585'],light=True):

    #Preparation
    clim_light = ns.Climatology( event , time_period , models ,n_sample  , ns_law )
    Xebm   = ns.EBM().draw_sample( clim_light.time , n_sample + 1 , fix_first = 0 )
    data={}
    lY={}
    all_models={}
    models_reduced=[]



    for scen in scenarios:
        print(scen)
        lX2= []
        models_scen=[x for x in models if re.match('.*_'+scen,x)]
        for model in models_scen:
            mod=model.split("_")[0]
            u=lX_out.loc[(lX_out['Model']==mod)&
                (lX_out['scenario'].isin([scen, 'historical'])) ,'tas']
            u.name=model
            u=pd.DataFrame(u)
            lX2.append(  u)
        #reload Y in panda format
    
        ## Prior Functions (Light_version is faster)
        ##================================

        clim_light = ns.Climatology( event , time_period , models_scen ,n_sample  , ns_law )
        clim_light =ns.covariates_FC_GAM( clim_light , lX2 ,  Xebm , dof = 7 , verbose = False,light=light )

        data[scen]=clim_light
        #lY[scen]=lY_scen
        all_models[scen]=models_scen
        models_reduced=models_reduced+list(models_scen)
        clim_light.to_netcdf( os.path.join( pathOut , ("clim_light_onlyFCGAM_"+scen+"_"+dt_string+".nc")  ) )
    
    models_reduced=np.unique([m.split("_")[0]+"_"+m.split("_")[1]  for m in models_reduced] ) #Liste models/type
    return data,all_models,models_reduced,clim_light

def create_corresponding_X_Y_multiscenario(scenarios, mod,all_models,lY_out,data, sample="BE",random=False):
    Y = []
    X = []
    for scen in scenarios:
        if (mod+"_"+scen in all_models[scen]):
            Y_scen=lY_out.loc[(lY_out['Model']==mod)&
                (lY_out['scenario'].isin([scen, 'historical'])),'tasmax' ] 
            
            tY    = Y_scen.index
            if random:
                idx = np.random.choice( tY.size , tY.size , replace = True )
                tY = tY.values[idx]
                Y_scen=Y_scen.iloc[idx]
                
            X_scen= data[scen].X.loc[tY,sample,"F",mod+"_"+scen].to_pandas()

            X.append(X_scen)
            Y.append(Y_scen)
            
    X_out = pd.concat(X)
    Y_out = pd.concat(Y)
    return(X_out,Y_out)

class MultiModel:##{{{
	"""
	NSSEA.MultiModel
	================
	Class infering multimodel parameters. Use NSSEA.infer_multi_model to build it
	
	
	Attributes
	----------
	mean   : array
		Multi model mean
	cov    : array
		Multi model covariance matrix
	std    : array
		Square root of multimodel covariance matrix
	"""
	
	def __init__( self ):##{{{
		self.mean = None
		self._cov  = None
		self.std  = None
	##}}}
	
	def _fit( self , mm_matrix ):##{{{
		n_params,n_sample,n_models = mm_matrix.shape
		
		cov_S = np.zeros( (n_params,n_params) )
		for i in range(n_models):
			cov_S += np.cov( mm_matrix[:,1:,i] )
		SSM     = np.cov( mm_matrix[:,0,:] ) * ( n_models - 1 )
		cov_CMU = matrix_positive_part( SSM / ( n_models - 1 ) - cov_S / n_models )
		self.cov  = ( n_models + 1 ) / n_models * cov_CMU + cov_S / n_models**2
	##}}}
	
	def fit( self , mm_matrix ):##{{{
		"""
		Fit Multi model parameters
		
		Parameters
		----------
		mm_matrix: array
			Big matrix containing sample to infer multi model parameters
		method   : str
			Method used, "classic" or "optimal"
		verbose  : bool
			Print (or not) state of execution
		"""
		self.mean = np.mean( mm_matrix[:,0,:] , axis = 1 )
		self._fit(mm_matrix)
	##}}}
	
	def rvs_old(self):##{{{
		"""
		Return a random sample from multi model
		"""
		### Depreciated for scipy 1.11.1
		return self.mean + self.std @ np.random.normal(size = self.mean.size) 
	##}}}
	def rvs(self):##{{{
		"""
		Return a random sample from multi model
		"""
		### Added for scipy 1.11.1
		return np.random.default_rng().multivariate_normal( mean=self.mean, cov = self.cov)
	##}}}
	## Properties {{{
	
	@property
	def n_mm_coef(self):
		return None if self.mean is None else self.mean.size
	
	@property
	def cov(self):
		return self._cov
	
	@cov.setter
	def cov( self , _cov ):
		self._cov = _cov
		self.std = matrix_squareroot(self._cov)
		
def Multi_nsfit_multimodel(clim_light,models_reduced,all_models ,data,lY_out,event, scenarios=['ssp585'],sampling=False,light=True):
	clim=clim_light.copy()
	## Parameters
	models      = models_reduced
	n_models    = len(models_reduced)
	sample      = clim.sample
	n_sample    = clim.n_sample
	
	n_ns_params = clim.ns_law.n_ns_params
	ns_params_names = clim.ns_law.get_params_names()
	
	
	law_coef   = xr.DataArray( np.zeros( (n_ns_params,n_sample + 1,n_models) ) , coords = [ ns_params_names , sample , models ] , dims = ["coef","sample","model"] )
	for mod in models:
		print(mod)
		X_out,Y_out=create_corresponding_X_Y_multiscenario(scenarios, mod,all_models, lY_out,data,sample="BE")
		law = clim.ns_law
		law.fit(Y_out.values,X_out.values)
		law_coef.loc[:,"BE",mod] = law.get_params()
		coef_be = law_coef.loc[:,"BE",mod].values.ravel()
		if not light:
			for s in sample:
			
				fit_is_valid = False
				while not fit_is_valid:
					Xs,Ys = create_corresponding_X_Y_multiscenario(scenarios, mod, sample=s,random=True)
					law.fit(Ys.values,Xs.values, init = coef_be)
					fit_is_valid = True
				law_coef.loc[:,s,mod] = law.get_params()
	n_time    = clim.n_time
	time     = clim.time
	n_alltimes=n_time*len(scenarios) #Times chronicles for all scenarios
	n_sample  = clim.n_sample
	models      = models_reduced
	n_model    = len(models_reduced)
	sample    = clim.sample
	n_coef      = clim.ns_law.n_ns_params
	n_mm_coef = 2 * n_alltimes + n_coef
	
	## Big matrix
	##===========
	prod=list(product(scenarios,clim.time))    
	scen_times =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
	mm_data                        = np.zeros( (n_mm_coef,n_sample + 1,n_model) )
	dX_scen = xr.DataArray( np.empty( (n_alltimes,n_sample + 1,2,n_model) ) , coords = [scen_times , sample , ["F","C"] , models ] , dims = ["time","sample","forcing","model"] )
	dX_scen[:] = np.nan ###### Changé ça pour nan
	for i in range(len(scenarios)):
		print(scenarios[i])
		scen=scenarios[i]
		#if len(all_models[scen])!=len(models_reduced):
		dX = xr.DataArray( np.empty( (n_time,n_sample + 1,2,n_model) ) , coords = [time , sample , ["F","C"] , models ] , dims = ["time","sample","forcing","model"] )
		dX[:] = np.nan ###### Changé ça pour nan
		for mod in models:
			if (mod+"_"+scen) in (all_models[scen]):    
				dX.loc[:,:,:,mod] =data[scen].X.loc[:,:,:,mod+"_"+scen]
				prod=list(product([scen],clim.time))
				time_one_scen=np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
				dX_scen.loc[time_one_scen,:,:,mod]=data[scen].X.loc[:,:,:,mod+"_"+scen].values
		mm_data[n_time*i:n_time*(i+1),:,:]           = dX.loc[:,:,"F",:].values
		mm_data[n_alltimes+n_time*(i):n_alltimes+(i+1)*n_time,:,:] = dX.loc[:,:,"C",:].values   
        
	mm_data[(2)*n_alltimes:,:,:]       = law_coef.values 
	## Multi model parameters inference
	##=================================
	mmodel = MultiModel()
	mm_matrix=mm_data
	n_params,n_sample_tot,n_models = mm_matrix.shape
	cov_S = np.zeros( (n_params,n_params) )
	for i in range(n_models):
		cov_S_i=np.cov( mm_matrix[:,1:,i] )
		cov_S_i[np.isnan(cov_S_i)]=0
		cov_S += cov_S_i
	
	SSM     = pd.DataFrame(mm_matrix[:,0,:].T).cov().T * ( n_models - 1 )
	cov_CMU = matrix_positive_part( SSM / ( n_models - 1 ) - cov_S / n_models )
	covar = ( n_models + 1 ) / n_models * cov_CMU + cov_S / n_models**2
	mmodel.mean=np.nanmean( mm_data [:,0,:] , axis = 1 )
	mmodel.cov=covar

	## Generate sample
	##================
	name = "Multi_Synthesis"
	mm_sample = xr.DataArray( np.zeros( (n_alltimes,n_sample + 1,2,1) ) , coords = [ scen_times , sample , clim.data.forcing , [name] ] , dims = ["time","sample","forcing","model"] )
	mm_params = xr.DataArray( np.zeros( (n_coef,n_sample + 1,1) )   , coords = [ law_coef.coef.values , sample , [name] ]     , dims = ["coef","sample","model"] )
	
	mm_sample.loc[:,"BE","F",name] = mmodel.mean[:n_alltimes]
	mm_sample.loc[:,"BE","C",name] = mmodel.mean[n_alltimes:(2*n_alltimes)]
	mm_params.loc[:,"BE",name]     = mmodel.mean[(2*n_alltimes):]
	
	if sampling:
		draw=np.array(np.random.default_rng().multivariate_normal( mean=mmodel.mean, cov = covar,size=len(sample[1:])))
		mm_sample.loc[:,sample[1:],"F",name] = draw[:,:n_alltimes].T
		mm_sample.loc[:,sample[1:],"C",name] = draw[:,n_alltimes:(2*n_alltimes)].T
		mm_params.loc[:,sample[1:],name]     = draw[:,(2*n_alltimes):].T

	data_mm = xr.Dataset( { "X" : xr.concat( [dX_scen,mm_sample] , dim = "model",coords='minimal'  ) , "law_coef" : xr.concat( [law_coef,mm_params] , dim = "model"  ) } )
	## Add multimodel to xarray, and add to clim
	##==========================================
	index = [ "{}F".format(t) for t in scen_times ] + [ "{}C".format(t) for t in scen_times ] + law_coef.coef.values.tolist()
	dmm_mean  = xr.DataArray( mmodel.mean , dims = ["mm_coef"] , coords = [index] )
	dmm_cov   = xr.DataArray( mmodel.cov  , dims = ["mm_coef","mm_coef"] , coords = [index,index] )
	dataxr3= data_mm.assign( { "mm_mean" : dmm_mean , "mm_cov" : dmm_cov, "anomaly_period" : event.reference } )
	clim.data =dataxr3   
	return clim 

def Multi_constrain_covariate_fast(clim_MM,Xo,scenarios=['ssp585'],assume_good_scale=False,time_reference = np.arange( 1986 , 2016 , 1 , dtype = int )):
	clim = clim_MM.copy()
	assume_good_scale = False
	
	if 'ssp245' in scenarios:
		scen='ssp245'
	else :
		scen=scenarios[-1]
	prod=list(product([scen],time_reference))
	time_reference=np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
	## Parameters
	##===========
	time        = clim.time
	time_Xo     = Xo.index
	prod=list(product([scen],time_Xo))
	time_Xo_Ref=np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
	n_time      = clim.n_time
	n_time_Xo   = time_Xo.size
	n_mm_coef   = clim.data["mm_mean"].size
	n_coef      = clim.n_coef
	n_sample    = clim.n_sample
	sample     = clim.X.sample
	samples     = clim.X.sample
	## Projection matrix H
	##====================
	cx = xr.DataArray( np.zeros( (n_time,n_time) )       , coords = [time,time]       , dims = ["time1","time2"] )
	cy = xr.DataArray( np.zeros( (n_time_Xo,n_time_Xo) ) , coords = [time_Xo_Ref,time_Xo_Ref] , dims = ["time1","time2"] )
	if not assume_good_scale:
		cx.loc[:,time_reference] = 1. / time_reference.size
	if not assume_good_scale:
		cy.loc[:,time_reference] = 1. / time_reference.size
	
	centerX  = np.identity(n_time)    - cx.values
	centerY  = np.identity(n_time_Xo) - cy.values
	extractX = np.hstack( ( np.identity(n_time) , np.zeros( (n_time,n_time) ) , np.zeros( (n_time,n_coef) ) ) )
	H_full   = pd.DataFrame( centerX @ extractX , index = time )	# Centering * extracting
	H        = H_full.loc[time_Xo_Ref,:].values								# Restriction to the observed period
	
	
	# Other inputs : x, SX, y, SY
	##===========================
	X  = clim.data["mm_mean"].values
	SX = clim.data["mm_cov"].values
	Y  = np.ravel(centerY @ Xo)
	SY = centerY @ centerY.T
	
	## Rescale SY
	##===========
	if not assume_good_scale:
		res = Y - H @ X
		K   = H @ SX @ H.T
		
		def fct_to_root(lbda):
			SY_tmp = lbda * SY
			iSig = np.linalg.pinv( K + SY_tmp )
			out = np.sum( np.diag( iSig @ SY_tmp ) ) - res.T @ iSig @ SY_tmp @ iSig @ res
			return out
		
		a,b = 1e-2,1e2
		while fct_to_root(a) * fct_to_root(b) > 0:
			a /= 2
			b *= 2
		
		lbda = sco.brentq( fct_to_root , a = a , b = b )
		SY = lbda * SY
	
	## Apply constraints
	##==================
	
	Sinv = np.linalg.pinv( H @ SX @ H.T + SY )
	K	 = SX @ H.T @ Sinv
	clim.data["mm_mean"].values = X + K @ ( Y - H @ X )
	clim.data["mm_cov"].values  = SX - SX @ H.T @ Sinv @ H @ SX
	
	cx_sample = xr.DataArray( np.zeros( (n_time,n_sample + 1,2) ) , coords = [ clim.X.time , samples , clim.X.forcing ] , dims = ["time","sample","forcing"] )
	mmodelmean=clim.data["mm_mean"].values
	covar=clim.data["mm_cov"].values
	cx_sample.loc[:,"BE","F"] = mmodelmean[:n_time]
	cx_sample.loc[:,"BE","C"] = mmodelmean[n_time:(2*n_time)]
	draw=np.array(np.random.default_rng().multivariate_normal( mean=mmodelmean, cov = covar,size=len(sample[1:])))

	cx_sample.loc[:,sample[1:],"F"] = draw[:,:n_time].T
	cx_sample.loc[:,sample[1:],"C"] = draw[:,n_time:(2*n_time)].T
	for m in clim.model:
		clim.X.loc[:,:,:,m] = cx_sample.values     
	return clim

def stan_constrain_multiscenario(climIn,Yo,scenarios=['ssp585'],stan_file='GEV_non_stationary.stan', **kwargs):
    clim      = climIn.copy()
    if 'ssp245' in scenarios:
        scen='ssp245'
    else :
        scen=scenarios[-1]
    #data
    time_Yo     = Yo.index
    prod=list(product([scen],time_Yo))
    time_Yo_Ref=np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
    X=clim.X.loc[time_Yo_Ref,'BE',"F","Multi_Synthesis"].values.squeeze()
    
    N_X=len(X)
    Y=Yo.values.squeeze()
    N=len(Y)
    p_m   = clim.data["mm_mean"][-clim.n_coef:].values
    p_cov    = clim.data["mm_cov"][-clim.n_coef:,-clim.n_coef:].values

    u,s,v = np.linalg.svd(p_cov)
    p_std = u @ np.sqrt(np.diag(s)) @ v.T
    
    #clim para
    n_features = clim.ns_law.n_ns_params
    ns_params_names = clim.ns_law.get_params_names()
    n_ess = kwargs.get("n_ess")
    if n_ess is None:
        n_ess = 10
    sample_names =[s+"_"+str(i) for i in range(n_ess) for s in clim.sample[1:]]+["BE"]
    law_coef_bay   = xr.DataArray( np.zeros( (n_features,(clim.n_sample)*n_ess + 1,1) ) ,
                                  coords = [ ns_params_names , sample_names , ["Multi_Synthesis"] ] ,
                                  dims = ["coef","sample_MCMC","model"] )
    
    
    
    newDF = pd.DataFrame() #creates a new dataframe that's empty
    samples=clim.X.loc[time_Yo_Ref,:,"F","Multi_Synthesis"].sample.values.squeeze()[1::]

    #compile stan file
    model = CmdStanModel(stan_file=stan_file)

    for s in samples:
        print(s)
        X=clim.X.loc[time_Yo_Ref,s,"F","Multi_Synthesis"].values.squeeze()
        data = {"N":N,
                "Y": Y,
                "N_X": N_X,
                "X": X,
                "p_m":p_m,
                "p_cov":p_std
                }
    
        fit = model.sample(data=data,iter_sampling=int(n_ess/4),show_progress=False)
        df = fit.draws_pd()
        sub_def=df[['para[1]','para[2]','para[3]','para[4]','para[5]']]
        law_coef_bay.loc[:,[s+"_"+str(i) for i in range(n_ess)],"Multi_Synthesis"]=sub_def.T
    clim.law_coef=law_coef_bay
    clim.law_coef.loc[:,"BE",:] = clim.law_coef[:,1:,:].median( dim = "sample_MCMC" )
    clim.BE_is_median = True
    return clim

	
