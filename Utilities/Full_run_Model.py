##### Packages
import sys,os
import numpy as np

print(sys.path)
#import arviz
import NSSEA as ns
import NSSEA.models as nsm
from cmdstanpy import cmdstan_path, set_cmdstan_path
set_cmdstan_path('/home/barbauxo/miniconda3/envs/TestSDFC/bin/cmdstan')
stan_file="Utilities/GEV_non_stationary.stan"


from Functions_run_Model import *


### Set seed
from datetime import datetime
import random
# datetime object containing current date and time
now = datetime.now() 
# dd/mm/YY H:M:S
dt_string = now.strftime("%m%d%H%M")
print("seed is: "+  dt_string)
#dt_string="01151622"

random.seed(dt_string)


#External Arguments

Y_ref=sys.argv[1]
X_ref=sys.argv[2]

work_path=sys.argv[3]

pathObs=sys.argv[4]
pathGCM_X_separated=sys.argv[5]
pathGCM_Y_separated=sys.argv[6]

ref_deb=int(sys.argv[7])
ref_fin=int(sys.argv[8])


#######################Fixed arguments
time_period    = np.arange( 1850 , 2101 , 1 , dtype = int )
time_reference = np.arange( ref_deb , ref_fin , 1 , dtype = int )

ci          = 0.05

#Time period of interest
T=100
T1=2050
T2=2100
deb=1850
fin=2101

ns_law      = nsm.GEV()
event       = ns.Event( "HW19" , 2019 , time_reference , type_ = "value" , variable = "TX3X" , unit = "K" )
n_sample    = 100 #1000?
verbose=False
event.value =0# float(Yo.loc[event.time])
sample_dis=False #If want n sample of each GCM model. For graph only, not used for multisynthesis. Takes a lot of time
light=True
bayes_kwargs = { "n_ess"   : 100}

scenarios=['ssp119','ssp126','ssp245','ssp370','ssp585']

##### Paths
print("Application: "+Y_ref)

basepath=work_path
pathOut=os.path.join(basepath,'./Outputs/'+Y_ref+"_"+X_ref,dt_string)
if not os.path.exists(pathOut):
    os.makedirs(pathOut)

#### Run
#Observations Meteo
dXo = xr.open_dataset(os.path.join( pathObs,"Xo_Ano.nc")) #Deja en anomalies
Xo  = pd.DataFrame( dXo.Xo.values.squeeze() , columns = ["Xo"] , index = dXo.index.values )

dYo =  xr.open_dataset(os.path.join( pathObs,"Yo_Ano.nc"))
Yo  =  pd.DataFrame( dYo.Yo.values.squeeze() , columns = ["Yo"] , index = dYo.index.values )

# Cmip data
lX_out,lY_out,models=import_gcm_multiscenario(pathGCM_X_separated,pathGCM_Y_separated,scenarios)
print(lX_out)

data,all_models,models_reduced,clim_light=Multi_up_to_FC_Gam(pathOut,lX_out,models,event,time_period,n_sample, ns_law,dt_string,scenarios,light)
print(len(models_reduced))

clim=Multi_nsfit_multimodel(clim_light,models_reduced,all_models ,data,lY_out,event,scenarios,sampling=True,light=light)
clim.to_netcdf( os.path.join( pathOut , ("clim_multiscenarios_multisynthesis_allmodels_"+dt_string+".nc")  ) )
clim_light_MM = clim.copy()
clim_light_MM.keep_models( "Multi_Synthesis" )
clim_light_MM.to_netcdf( os.path.join( pathOut , ("clim_multiscenarios_multisynthesis_"+dt_string+".nc")  ) )
print("Multiscenario done")

clim_light_MM=ns.Climatology.from_netcdf(os.path.join( pathOut , ("clim_multiscenarios_multisynthesis_"+dt_string+".nc")  )  , ns_law  )
clim_CX=Multi_constrain_covariate_fast(clim_light_MM,Xo,scenarios,time_reference =time_reference)
clim_CX.to_netcdf( os.path.join( pathOut , ("clim_multiscenarios_CX_"+dt_string+".nc")  ) )
print("CX done")
clim_CX=ns.Climatology.from_netcdf(os.path.join( pathOut , ("clim_multiscenarios_CX_"+dt_string+".nc")  )  , ns_law  )



climCXCB=stan_constrain_multiscenario(clim_CX,Yo,scenarios,stan_file=stan_file, **bayes_kwargs)

climCXCB.to_netcdf( os.path.join( pathOut , ("clim_multiscenarios_CXCB_"+dt_string+".nc")  ) )

sys.stdout.write(dt_string)


