
#####Packages

import sys,os
print(sys.version)
import numpy as np
import pandas as pd
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator
import warnings
warnings.simplefilter("ignore")

#Arguments

Y_ref=sys.argv[1]
X_ref=sys.argv[2]

work_path=sys.argv[3]

path_X_Modeles_Brut=sys.argv[4]
path_Y_Modeles_Brut=sys.argv[5]

path_X_Obs_Brut=sys.argv[6]
path_Y_Obs_Brut=sys.argv[7]

lat=sys.argv[8]
lon=sys.argv[9]

ref_deb=int(sys.argv[10])
ref_fin=int(sys.argv[11])


#Paths 

print("Application: "+Y_ref)
basepath=os.path.join(work_path,"data")
print(basepath)
pathFig=os.path.join(basepath,"Figures/"+Y_ref)
if not os.path.exists(pathFig):
    os.makedirs(pathFig)
    
pathTemp=os.path.join(basepath,"temp")
if not os.path.exists(pathTemp):
    os.makedirs(pathTemp)

    
pathOut_X=os.path.join(basepath,"CMIP6/03_Post_Treatment/TasMean_"+X_ref)
if not os.path.exists(pathOut_X):
    os.makedirs(pathOut_X)
X_files="{}_tas_YearMean_"+X_ref+".nc"

pathOut_Y=os.path.join(basepath,"CMIP6/03_Post_Treatment/TXX_"+Y_ref)
if not os.path.exists(pathOut_Y):
    os.makedirs(pathOut_Y)
Y_files="{}_tasmax_YearMax_"+Y_ref+".nc"
Y_files_input="{}_tasmax_YearMax_TXX_France_Full.nc"

pathOut_Obs=os.path.join(basepath,"Observations/03_Post_Treatment/"+Y_ref+"_"+X_ref)
if not os.path.exists(pathOut_Obs):
    os.makedirs(pathOut_Obs)


    
#Functions
def correct_miss( X , lo =  100 , up = 365 ):##{{{
#	return X
	mod = str(X.columns[0])
	bad = np.logical_or( X < lo , X > up )
	bad = np.logical_or( bad , np.isnan(X) )
	bad = np.logical_or( bad , np.logical_not(np.isfinite(X)) )
	if np.any(bad):
		print("Bad values "+bad.columns[0] +": "+str(bad.sum().values[0]))
		idx,_ = np.where(bad)
		idx_co = np.copy(idx)
		for i in range(idx.size):
			j = 0
			while idx[i] + j in idx:
				j += 1
			idx_co[i] += j
			if (idx_co[i]>=len(X)):
				idx_co[i] =idx_co[i-1] 
		X.iloc[idx] = X.iloc[idx_co].values
	return X

## Corrections Outliers
def is_outlier(Y,log_outliers):
    Y_means=[]
    Y=Y[Y.columns[0]]
    Y.index=pd.to_datetime(Y.index.astype(str), yearfirst=True)

    if sum(Y.index!=Y.sort_index().index)>0:
        print("index issues")
        print(Y.index[(Y.index!=Y.sort_index().index)])
        log_outliers.write(str(Y.index[(Y.index!=Y.sort_index().index)])+"\n")
        Y=Y.sort_index()
        Anomalies=(abs(Y)>4)*0
    else:

        #freq_roll="3650D" #10years
        #freq_roll="7300D" #20 ans
        #freq_roll="1825D" #5 ans
        freq_roll="10950D" #30 ans
        mean_rolling_Y=Y.rolling(freq_roll,min_periods=1, center=True).mean()
        std_rolling_Y=Y.rolling(freq_roll,min_periods=1, center=True).std()
        #std_rolling_Y[std_rolling_Y<1.6]=1.66666
        #Supposant une loi normale car n grand (min 30, voir 25*30)
        #Probabilité 3eq= 99.7%
        #Probabilité 4eq= 99.99%
        #pas de prise en compte de la queue
        Anomalies=(abs(Y-mean_rolling_Y)/std_rolling_Y>4)*1
    Anomalies.index=Anomalies.index.year.astype(int)
    return Anomalies
    
    
#General args
time_period    = np.arange( 1850 , 2101 , 1 , dtype = int )
time_reference = np.arange( ref_deb , ref_fin , 1 , dtype = int )

exclusion=['NorESM2-LM_i1p1f1','UKESM1-0-LL_i1p1f1','UKESM1-0-LL_i1p1f2','UKESM1-0-LL_i1p1f3','HadGEM3-GC31-LL_i1p1f3','HadGEM3-GC31-MM_i1p1f3']

########### Treatment Obs
print("Start treatment of Observation Data")
###For X0, covariate
dXo = xr.open_dataset(path_X_Obs_Brut,decode_times=False) #Deja en anomalies
Xo  = pd.DataFrame( dXo.tas.values.squeeze() , columns = ["Xo"] , index = dXo.time.values )
#Obs Anomalies
Xo_ano =Xo-Xo.loc[time_reference].mean()
#Save obs
Xo_ano.to_xarray().to_netcdf( os.path.join( pathOut_Obs,"Xo_Ano.nc" ) )
print("Example for Xo: "+str(Xo_ano.iloc[-1]))

###For Y0, interest var	
print(Y_ref)
bashCommand = "cdo -selvar,TX -yearmax "+path_Y_Obs_Brut+"_*.nc"+" "+pathTemp+"/Yo_test.nc"
os.system(bashCommand)
dYo = xr.open_dataset(os.path.join(pathTemp,"Yo_test.nc"))
Yo  = pd.DataFrame( dYo.TX.values.squeeze() , columns = ["Yo"] , index = 		dYo.time["time.year"].values )	
Yo=Yo[np.isfinite(Yo.values)]
# kelvin to celsius
if Yo.mean().values>273.15:
	Yo=Yo-273.15	
#Obs Anomalies
bias = { "Multi_Synthesis" : Yo.loc[time_reference].mean().values }
Yo_ano =Yo- bias["Multi_Synthesis"]
print(Yo_ano.index)
print("Observation bias: "+str(bias))
#Save obs and bias Observations
Yo_ano.to_xarray().to_netcdf( os.path.join( pathOut_Obs,"Yo_Ano.nc" ) )
pd.DataFrame(bias).to_xarray().to_netcdf( os.path.join( pathOut_Obs,"bias_Obs.nc" ) )
print("Example for Yo: "+str(Yo_ano.iloc[-1]))

###Figure OBS
fig, ax = plt.subplots()
ax.plot(Yo_ano.index,Yo_ano.values,label="TXX" )
ax.plot(Xo_ano.index,Xo_ano.values,label="TMM" )
xlim = ax.get_xlim()
ax.axvspan(ref_deb, ref_fin,
           alpha=0.03,
           color='blue',
           label="Reference period")

ax.set_xlim(xlim)
plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.1f°C'))
plt.title("Observations series (Anomaly) ")
plt.legend(  )#handles=[]
#plt.show()
plt.savefig(os.path.join( pathFig ,"Anomalies_obs.png"))
plt.close()




########### Treatment CMIP
print("Start treatment of CMIP Data")
#Conserve si meme Modele+type+scenario pour tas et tasmax
## List of models X
modelsX = [  "_".join(f.split("/")[-1][:-3].split("_")[:3]) for f in os.listdir(path_X_Modeles_Brut) ] #Modele+type+scenario
modelsX.sort()

## List of models Y
modelsY =  [  "_".join(f.split("/")[-1][:-3].split("_")[:3]) for f in os.listdir(path_Y_Modeles_Brut) ] #Modele+type+scenario
modelsY.sort()

models = list(set(modelsX) & set(modelsY)) #Conserve si meme Modele+type+scenario pour tas et tasmax
models.sort()


#Only X treatment absolute
lX = []
for m in models:
        model, typeRun, scen=m.split("_")

        ## Load X
        
        df   = xr.open_dataset( os.path.join( path_X_Modeles_Brut , X_files.format(m) )  ,decode_times=False )
        time = df.time.values.astype(int)
        X    = pd.DataFrame( df.tas.values.ravel() , columns = [m] , index = time )
        X_miss=correct_miss(X)
        X=pd.DataFrame( np.array([X_miss.values.ravel(),
          np.array([model for i in range(len(time))]),
          np.array([typeRun for i in range(len(time))]),
          np.array([scen for i in range(len(time))])]).T, 
          columns = ['tas',"Model", "typeRun", "scenario"] , index = time )
        X['tas']=X['tas'].astype(float)
        lX.append(  X)
lX_out = pd.concat(lX)
lX_out.index.name= 'Year'

scenarios=np.unique(lX_out['scenario'])[1:] #liste scenarios
models_reduced=np.unique([m.split("_")[0]+"_"+m.split("_")[1] for m in models]) #Liste models/type 
#Only Y treatment absolute
lY = []
for m in models:
        model, typeRun, scen=m.split("_")
        # prepare
        full_file= os.path.join( path_Y_Modeles_Brut , Y_files_input.format(m))
        temp_file=pathTemp+"/"+Y_files.format(m)

        "cdo remapnn,lon="+str(lon)+"_lat="+str(lat)+" "+ full_file+" "+ temp_file
        bashCommand = "cdo  -remapnn,lon="+str(lon)+"_lat="+str(lat)+" -setmisstonn "+ full_file+" "+ temp_file
        os.system(bashCommand)
        
        ## Load Y
        df   = xr.open_dataset(temp_file  ,decode_times=False  )
        time = df.time.values.astype(int)
        Y    = pd.DataFrame( df.tasmax.values.ravel() , columns = [m] , index = time )
        Y_miss=correct_miss(Y)
        Y=pd.DataFrame( np.array([Y_miss.values.ravel(),
          np.array([model for i in range(len(time))]),
          np.array([typeRun for i in range(len(time))]),
          np.array([scen for i in range(len(time))])]).T, 
          columns = ['tasmax',"Model", "typeRun", "scenario"] , index = time )
        Y['tasmax']=Y['tasmax'].astype(float)
        lY.append( Y)
lY_out = pd.concat(lY)
lY_out.index.name= 'Year'

scenarios=np.unique(lY_out['scenario'])[1:] #liste scenarios
#print(scenarios)
print("CMIP Scenarios available: "+str(scenarios))
models_reduced=np.unique([m.split("_")[0]+"_"+m.split("_")[1] for m in models]) #Liste models/type 



######### Figures
##Analyse pré traitement
#Covariable X, val absolue en  Celcius, all scenarios
for sc in scenarios:
    
    X_means=[]
    for m1 in models_reduced:
        X=lX_out.loc[(lX_out['scenario'].isin(['historical', sc])) &
                 (lX_out['Model']==m1.split("_")[0])&
                 (lX_out['typeRun']==m1.split("_")[1]) ]
        X_means.append(X.groupby('Year').aggregate({X.columns[0]:"mean"})-273.15)
    
    
    fig, ax = plt.subplots()
    period=list(X_means[0].index)
    for i in range(0,len(X_means)):
        ax.plot(X_means[i].index,X_means[i].values )
        #period2=list(lX[i].index)
        #ax.plot(period2,lX[i].values -273.15)
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])

    ax.set_xlim(xlim)
    plt.title("Valeurs absolues sur X - Moyenne annuelle - "+sc)
    plt.legend( handles=[] )
    #plt.show()
    plt.savefig(os.path.join( pathFig ,"02_Tri_Absolutes_X_All_AnnualMean_"+sc+".png"))
    plt.close()
    
    fig, ax = plt.subplots()
    period=list(X_means[0].index)
    for m1 in models_reduced:
        X=lX_out.loc[(lX_out['scenario'].isin(['historical', sc])) &
                 (lX_out['Model']==m1.split("_")[0])&
                 (lX_out['typeRun']==m1.split("_")[1]) ]
        period2=X.index
        ax.plot(period2,X['tas'].values -273.15)
    #ax.plot( Xo_abs.index , Xo_abs.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")

    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    
    ax.set_xlim(xlim)
    plt.title("Valeurs absolues sur X - All Values - "+sc)
    plt.legend( handles=[] )
    #plt.show()
    plt.savefig(os.path.join( pathFig ,"02_Tri_Absolutes_X_All_AllValues_"+sc+".png"))
    plt.close()

#Variable Y, val absolue en  Celcius, all scenarios
for sc in scenarios:
    
    Y_means=[]
    for m1 in models_reduced:
        Y=lY_out.loc[(lY_out['scenario'].isin(['historical', sc])) &
                 (lY_out['Model']==m1.split("_")[0])&
                 (lY_out['typeRun']==m1.split("_")[1]) ]
        Y_means.append(Y.groupby('Year').aggregate({Y.columns[0]:"mean"})-273.15)
    
    
    fig, ax = plt.subplots()
    period=list(Y_means[0].index)
    for i in range(0,len(Y_means)):
        ax.plot(Y_means[i].index,Y_means[i].values )
        #period2=list(lX[i].index)
        #ax.plot(period2,lX[i].values -273.15)
    ax.plot( Yo.index , Yo.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations (réseau variable)")

    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])

    ax.set_xlim(xlim)
    plt.title("Valeurs absolues sur Y - Moyenne annuelle - "+sc)
    plt.legend(handles=[] )
    
    plt.savefig(os.path.join( pathFig ,"02_Tri_Absolutes_Y_All_AnnualMean_"+sc+".png"))
    #plt.show()
    plt.close()
    
    fig, ax = plt.subplots()
    period=list(X_means[0].index)
    for m1 in models_reduced:
        Y=lY_out.loc[(lY_out['scenario'].isin(['historical', sc])) &
                 (lY_out['Model']==m1.split("_")[0])&
                 (lY_out['typeRun']==m1.split("_")[1]) ]
        period2=Y.index
        ax.plot(period2,Y['tasmax'].values -273.15)
    ax.plot( Yo.index , Yo.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations (réseau variable)")

    #ax.plot( Xo_abs.index , Xo_abs.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")

    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    
    ax.set_xlim(xlim)
    plt.title("Valeurs absolues sur Y - All Values - "+sc)
    plt.legend( handles=[])
    
    plt.savefig(os.path.join( pathFig ,"02_Tri_Absolutes_Y_All_AllValues_"+sc+".png"))
    #plt.show()
    plt.close()

#Only modele with scenario (If only historical, no year past 2014 -> exclude)
#Only modele with historical (If only scenario, no year before 2014 -> exclude)

only_2015_Y=lY_out.loc[lY_out.index==2015  ]
only_2003_Y=lY_out.loc[lY_out.index==2003  ]
only_kept=pd.merge(only_2015_Y, only_2003_Y, how='inner', on=['Model', 'typeRun'])
models_kept=np.unique(only_kept['Model'])
typeRun_kept=np.unique(only_kept['typeRun'])
lY_out=lY_out.loc[(lY_out['Model'].isin(models_kept))&
                 (lY_out['typeRun'].isin(typeRun_kept)) ]

lX_out=lX_out.loc[(lX_out['Model'].isin(models_kept))&
                 (lX_out['typeRun'].isin(typeRun_kept)) ]




models_reduced_sce=np.unique([only_kept['Model'].iloc[i]+"_"+only_kept['typeRun'].iloc[i] for i in range(len(only_kept))]) #Liste models/type 
models_reduced_full=models_reduced
models_reduced=models_reduced_sce

main_list = list(set(models_reduced) - set(exclusion))
main_list.sort()
models_reduced=main_list
#print(models_reduced)
print("Number of CMIP Models available: "+str(len(models_reduced)))
print(models_reduced)


#TRansformer en Anomalies for X
print("Start Anomalie transformation")
for m1 in models_reduced:
        X=lX_out.loc[(lX_out['Model']==m1.split("_")[0])&
                 (lX_out['typeRun']==m1.split("_")[1]) ]
        b_X=X['tas'].loc[time_reference].mean()
        lX_out.loc[(lX_out['Model']==m1.split("_")[0])&
                 (lX_out['typeRun']==m1.split("_")[1]) ,'tas']-=b_X
                 
#TRansformer en Anomalies for Y
bias_GCM = {}
for m1 in models_reduced:        
        Y=lY_out.loc[(lY_out['Model']==m1.split("_")[0])&
                 (lY_out['typeRun']==m1.split("_")[1]) ]
        b_Y=Y['tasmax'].loc[time_reference].mean()
        bias_GCM [m1.split("_")[0]] = b_Y- 273.15 #biais en celsius
        lY_out.loc[(lY_out['Model']==m1.split("_")[0])&
                 (lY_out['typeRun']==m1.split("_")[1]) ,'tasmax']-=b_Y



#Figure Anomaly for X
for sc in scenarios:

    
    X_means=[]
    for m1 in models_reduced:
        X=lX_out.loc[(lX_out['scenario'].isin(['historical', sc])) &
                 (lX_out['Model']==m1.split("_")[0])&
                 (lX_out['typeRun']==m1.split("_")[1]) ]
        X_means.append(X.groupby('Year').aggregate({X.columns[0]:"mean"}))
    
    
    fig, ax = plt.subplots()
    period=list(X_means[0].index)
    for i in range(0,len(X_means)):
        ax.plot(X_means[i].index,X_means[i].values )
        #period2=list(lX[i].index)
        #ax.plot(period2,lX[i].values -273.15)
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    ax.plot( Xo_ano.index , Xo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    
    ax.set_xlim(xlim)
    plt.title("Valeurs absolues sur X - Moyenne annuelle - "+sc)
    plt.legend(handles=[] )
    
    plt.savefig(os.path.join( pathFig ,"02_Tri_anomaly_X_All_AnnualMean_"+sc+".png"))
    #plt.show()
    plt.close()
    
    fig, ax = plt.subplots()
    period=list(X_means[0].index)
    for m1 in models_reduced:
        X=lX_out.loc[(lX_out['scenario'].isin(['historical', sc])) &
                 (lX_out['Model']==m1.split("_")[0])&
                 (lX_out['typeRun']==m1.split("_")[1]) ]
        period2=X.index
        ax.plot(period2,X['tas'].values )
    #ax.plot( Xo_abs.index , Xo_abs.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    ax.plot( Xo_ano.index , Xo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    
    ax.set_xlim(xlim)
    plt.title("Valeurs absolues sur X - All Values - "+sc)
    plt.legend(handles=[] )
    
    plt.savefig(os.path.join( pathFig ,"02_Tri_Anomaly_X_All_AllValues_"+sc+".png"))
    #plt.show()  
    plt.close()  

#Figure Anomaly for Y
for sc in scenarios:
    #print(sc)
    
    Y_means=[]
    for m1 in models_reduced:
        Y=lY_out.loc[(lY_out['scenario'].isin(['historical', sc])) &
                 (lY_out['Model']==m1.split("_")[0])&
                 (lY_out['typeRun']==m1.split("_")[1]) ]
        Y_means.append(Y.groupby('Year').aggregate({Y.columns[0]:"mean"}))
    
    
    fig, ax = plt.subplots()
    period=list(Y_means[0].index)
    for i in range(0,len(Y_means)):
        ax.plot(Y_means[i].index,Y_means[i].values )
        #period2=list(lX[i].index)
        #ax.plot(period2,lX[i].values -273.15)
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    ax.plot( Yo_ano.index , Yo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations (Réseau variable)")
    
    ax.set_xlim(xlim)
    plt.title("Valeurs absolues sur Y - Moyenne annuelle - "+sc)
    plt.legend(handles=[] )
    
    plt.savefig(os.path.join( pathFig ,"02_Tri_anomaly_Y_All_AnnualMean_"+sc+".png"))
    #plt.show()
    plt.close()
    
    fig, ax = plt.subplots()
    period=list(Y_means[0].index)
    for m1 in models_reduced:
        Y=lY_out.loc[(lY_out['scenario'].isin(['historical', sc])) &
                 (lY_out['Model']==m1.split("_")[0])&
                 (lY_out['typeRun']==m1.split("_")[1]) ]
        period2=Y.index
        ax.plot(period2,Y['tasmax'].values )
    #ax.plot( Xo_abs.index , Xo_abs.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    ax.plot( Yo_ano.index , Yo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations (Réseau variable)")
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    
    ax.set_xlim(xlim)
    plt.title("Valeurs absolues sur Y - All Values - "+sc)
    plt.legend(handles=[] )
    
    plt.savefig(os.path.join( pathFig ,"02_Tri_Anomaly_Y_All_AllValues_"+sc+".png"))       
    plt.close()

#Delete sup years for X
scen=np.unique(lX_out['scenario'])
#Ssp 245 starting in 2014 for EC-Earth3-Veg
lX_out=lX_out.loc[-((lX_out['scenario'].isin( scen[1:])) &(lX_out.index==2014))]
# Historical up to 2016 for FGOALS-g3
lX_out=lX_out.loc[-((lX_out['scenario'].isin( [scen[0]])) &(lX_out.index>2014)) ]

#Delete sup years for Y
scen=np.unique(lY_out['scenario'])
#Ssp 245 starting in 2014 for EC-Earth3-Veg
lY_out=lY_out.loc[-((lY_out['scenario'].isin( scen[1:])) &(lY_out.index==2014))]
# Historical up to 2016 for FGOALS-g3
lY_out=lY_out.loc[-((lY_out['scenario'].isin( [scen[0]])) &(lY_out.index>2014)) ]



#### Outlier treatment
print("start Outlier treatment")
scen=np.unique(lX_out['scenario'])
log_outliers=open(pathFig+'/X_Outliers_log.txt','a')
lX = []
#        lX.append(  X)
for m1 in models_reduced:
    
    for s in scen:
        X=lX_out.loc[(lX_out['Model']==m1.split("_")[0])&
                 (lX_out['typeRun']==m1.split("_")[1])&(lX_out['scenario']==s) ] 
        log_outliers.write(str(m1.split("_")[0])+"\n")
        

        Outliers=is_outlier(X,log_outliers)
        if sum(Outliers)>0:
            print("For X: "+m1.split("_")[0]+" " +s+" :"+str(sum(Outliers))+" Outliers")
            log_outliers.write(str(X[Outliers!=0].loc[:,['tas','scenario']])+"\n")
            log_outliers.write("\n")
            X=X[Outliers==0]
        lX.append(  X)
log_outliers.close()

lX_out_ano_clean = pd.concat(lX)
lX_out_ano_clean.index.name= 'Year'

scen=np.unique(lY_out['scenario'])
log_outliers=open(pathFig+'/Y_Outliers_log.txt','a')
lY = []
#        lX.append(  X)
for m1 in models_reduced:
    for s in scen:
        Y=lY_out.loc[(lY_out['Model']==m1.split("_")[0])&
                 (lY_out['typeRun']==m1.split("_")[1])&(lY_out['scenario']==s) ] 
        log_outliers.write(str(m1.split("_")[0])+"\n")
        
        Outliers=is_outlier(Y,log_outliers)
        if sum(Outliers)>0:
        #log
        

            print("For Y: "+m1.split("_")[0]+" " +s+" :"+str(sum(Outliers))+" Outliers")
            log_outliers.write(str(Y[Outliers!=0].loc[:,['tasmax','scenario']])+"\n")
            log_outliers.write("\n")
            Y=Y[Outliers==0]
        lY.append(  Y)
log_outliers.close()

lY_out_ano_clean = pd.concat(lY)
lY_out_ano_clean.index.name= 'Year'


for sc in scenarios:
    
    X_means=[]
    for m1 in models_reduced:
        X=lX_out_ano_clean.loc[(lX_out_ano_clean['scenario'].isin(['historical', sc])) &
                 (lX_out_ano_clean['Model']==m1.split("_")[0])&
                 (lX_out_ano_clean['typeRun']==m1.split("_")[1]) ]
        X_means.append(X.groupby('Year').aggregate({X.columns[0]:"mean"}))
    
    
    fig, ax = plt.subplots()
    period=list(X_means[0].index)
    for i in range(0,len(X_means)):
        ax.plot(X_means[i].index,X_means[i].values )
        #period2=list(lX[i].index)
        #ax.plot(period2,lX[i].values -273.15)
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    ax.plot( Xo_ano.index , Xo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    
    ax.set_xlim(xlim)
    plt.title("Valeurs absolues sur X - Moyenne annuelle - "+sc)
    plt.legend( handles=[])
    
    plt.savefig(os.path.join( pathFig ,"Clean_Tri_anomaly_X_All_AnnualMean_"+sc+".png"))
    plt.close()
    
    fig, ax = plt.subplots()
    period=list(X_means[0].index)
    for m1 in models_reduced:
        X=lX_out_ano_clean.loc[(lX_out_ano_clean['scenario'].isin(['historical', sc])) &
                 (lX_out_ano_clean['Model']==m1.split("_")[0])&
                 (lX_out_ano_clean['typeRun']==m1.split("_")[1]) ]
        period2=X.index
        ax.plot(period2,X['tas'].values,linestyle = ""  , marker=  "." ,alpha=0.3 )
    #ax.plot( Xo_abs.index , Xo_abs.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    ax.plot( Xo_ano.index , Xo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    
    ax.set_xlim(xlim)
    plt.title("X - Anomaly for 1986-2015 - All Values - "+sc)
    plt.legend(handles=[] )
    
    plt.savefig(os.path.join( pathFig ,"Clean_Tri_Anomaly_X_All_AllValues_"+sc+".png"))
    plt.close()
    
for sc in scenarios:
    
    Y_means=[]
    for m1 in models_reduced:
        Y=lY_out_ano_clean.loc[(lY_out_ano_clean['scenario'].isin(['historical', sc])) &
                 (lY_out_ano_clean['Model']==m1.split("_")[0])&
                 (lY_out_ano_clean['typeRun']==m1.split("_")[1]) ]
        Y_means.append(Y.groupby('Year').aggregate({Y.columns[0]:"mean"}))
    
    
    fig, ax = plt.subplots()
    period=list(Y_means[0].index)
    for i in range(0,len(Y_means)):
        ax.plot(Y_means[i].index,Y_means[i].values )
        #period2=list(lX[i].index)
        #ax.plot(period2,lX[i].values -273.15)
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    ax.plot(  Yo_ano.index , Yo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations  (Réseau variable)")
    
    ax.set_xlim(xlim)
    plt.title("Y - Anomaly for 1986-2015 - Annual Mean - "+sc)
    plt.legend(handles=[] )
    
    plt.savefig(os.path.join( pathFig ,"Clean_Tri_anomaly_Y_All_AnnualMean_"+sc+".png"))
    plt.close()
    
    fig, ax = plt.subplots()
    period=list(Y_means[0].index)
    for m1 in models_reduced:
        Y=lY_out_ano_clean.loc[(lY_out_ano_clean['scenario'].isin(['historical', sc])) &
                 (lY_out_ano_clean['Model']==m1.split("_")[0])&
                 (lY_out_ano_clean['typeRun']==m1.split("_")[1]) ]
        period2=Y.index
        ax.plot(period2,Y['tasmax'].values ,linestyle = ""  , marker=  "." ,alpha=0.3 )
    #ax.plot( Xo_abs.index , Xo_abs.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    ax.plot( Yo_ano.index , Yo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations  (Réseau variable)")
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    
    ax.set_xlim(xlim)
    plt.title("Y - Anomaly for 1986-2015 - All Values - "+sc)
    plt.legend(handles=[] )
    
    plt.savefig(os.path.join( pathFig ,"Clean_Tri_Anomaly_Y_All_AllValues_"+sc+".png"))
    plt.close()
    
######Save data
directory=os.path.join( pathOut_X,"separated")
if not os.path.exists(directory):
    os.makedirs(directory)
print("Start saving covariate data")
for m1 in models_reduced:
    scen_mod=np.unique(lX_out_ano_clean.loc[(lX_out_ano_clean['Model']==m1.split("_")[0])&
                 (lX_out_ano_clean['typeRun']==m1.split("_")[1]),'scenario'] ) #Only scenarios relevant for this modele
    for s in scen_mod:
        nameX="{}_{}_tas_YearMean_Europe.nc".format(m1,s) 
        X=lX_out_ano_clean.loc[(lX_out_ano_clean['Model']==m1.split("_")[0])&
                 (lX_out_ano_clean['typeRun']==m1.split("_")[1])&(lX_out_ano_clean['scenario']==s),'tas' ] 
        X.index.name='time'
        X.to_xarray().to_netcdf( os.path.join( directory,nameX ) )

directory=os.path.join( pathOut_Y,"separated" )
if not os.path.exists(directory):
    os.makedirs(directory)
    
print("Start saving Y data")
for m1 in models_reduced:
    scen_mod=np.unique(lY_out_ano_clean.loc[(lY_out_ano_clean['Model']==m1.split("_")[0])&
                 (lY_out_ano_clean['typeRun']==m1.split("_")[1]),'scenario'] ) #Only scenarios relevant for this modele
    for s in scen_mod:

    
        nameY="{}_{}_tasmax_YearMax.nc".format(m1,s) 
        Y=lY_out_ano_clean.loc[(lY_out_ano_clean['Model']==m1.split("_")[0])&
                 (lY_out_ano_clean['typeRun']==m1.split("_")[1])&(lY_out_ano_clean['scenario']==s) ,'tasmax'] 
        Y.index.name='time'
        Y.to_xarray().to_netcdf( os.path.join( directory,nameY ) )
        
        
#### Diagnostics
print("Start diagnostics")
print("Extrems values for Y")
print("Models with values over 20:"+np.unique(lY_out_ano_clean[lY_out_ano_clean["tasmax"]>20].Model))
print(lY_out_ano_clean[lY_out_ano_clean["tasmax"]>20].sort_values("tasmax"))



for m1 in models_reduced:
	fig, ax = plt.subplots()
	model="ACCESS-CM2"
	Y=lY_out.loc[(lY_out['scenario'].isin(['historical'])) &
                 (lY_out['Model']==model) ]
	ax.plot(Y.index,Y.tasmax,label=model+": 40 membres",linestyle = ""  , marker=  ".")

	model=m1
	Y=lY_out.loc[(lY_out['scenario'].isin(['historical'])) &
                 (lY_out['Model']==m1.split("_")[0]) ]
	ax.plot(Y.index,Y.tasmax,label=model+": 13 membres",linestyle = ""  , marker=  ".")
	plt.legend(handles=[])
	plt.savefig(os.path.join( pathFig ,"Comparison_Data_"+m1+"_with_ACCESS-CM2.png"))
	plt.close()
	
for m1 in models_reduced:
	fig, ax = plt.subplots()
	model="ACCESS-CM2"
	Y=lY_out_ano_clean.loc[(lY_out_ano_clean['scenario'].isin(['historical'])) &
                 (lY_out_ano_clean['Model']==model) ]
	ax.plot(Y.index,Y.tasmax,label=model+": 40 membres",linestyle = ""  , marker=  ".")

	model=m1
	Y=lY_out_ano_clean.loc[(lY_out_ano_clean['scenario'].isin(['historical'])) &
                 (lY_out_ano_clean['Model']==m1.split("_")[0]) ]
	ax.plot(Y.index,Y.tasmax,label=model,linestyle = ""  , marker=  ".")
	plt.legend()
	plt.savefig(os.path.join( pathFig ,"Comparison_CleanData_"+m1+"_with_ACCESS-CM2.png"))
	plt.close()
	
	
	
#### Figure for the papers


for sc in scenarios:    
    fig, ax = plt.subplots()
    period=list(X_means[0].index)
    for m1 in models_reduced:
        X=lX_out_ano_clean.loc[(lX_out_ano_clean['scenario'].isin(['historical', sc])) &
                 (lX_out_ano_clean['Model']==m1.split("_")[0])&
                 (lX_out_ano_clean['typeRun']==m1.split("_")[1]) ]
        period2=X.index
        ax.plot(period2,X['tas'].values,linestyle = ""  , marker=  "." ,alpha=0.3 )
    #ax.plot( Xo_abs.index , Xo_abs.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    ax.plot( Xo_ano.index , Xo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    
    ax.set_xlim(xlim)
    plt.title("X - Anomaly for 1986-2015 - All Values - "+sc)
    plt.legend()
    
    plt.savefig(os.path.join( pathFig ,"Paper_Anomaly_X_All_AllValues_"+sc+".png"))
    plt.close()
    fig, ax = plt.subplots()
    period=list(Y_means[0].index)
    for m1 in models_reduced:
        Y=lY_out_ano_clean.loc[(lY_out_ano_clean['scenario'].isin(['historical', sc])) &
                 (lY_out_ano_clean['Model']==m1.split("_")[0])&
                 (lY_out_ano_clean['typeRun']==m1.split("_")[1]) ]
        period2=Y.index
        ax.plot(period2,Y['tasmax'].values ,linestyle = ""  , marker=  "." ,alpha=0.3 )
    #ax.plot( Xo_abs.index , Xo_abs.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    ax.plot( Yo_ano.index , Yo_ano.values        , color = "black" , linestyle = ""  , marker=  "." , label="Observations")
    xlim = ax.get_xlim()
    #ax.hlines( 0 , xlim[0] , xlim[1] , color = "grey" )
    #plt.xticks(list(plt.xticks()[0]) + [1986,2016])
    
    ax.set_xlim(xlim)
    plt.title("Y - Anomaly for 1986-2015 - All Values - "+sc)
    plt.legend()
    
    plt.savefig(os.path.join( pathFig ,"Paper_Anomaly_Y_All_AllValues_"+sc+".png"))
    plt.close()





    
