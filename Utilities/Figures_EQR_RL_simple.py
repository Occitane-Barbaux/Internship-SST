##### Packages
import sys,os

import xarray as xr
import pandas as pd
import numpy as np

import NSSEA as ns
import NSSEA.models as nsm
import scipy.stats as sc
from itertools import product


import matplotlib.backends.backend_pdf as mpdf
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mplpatch
from matplotlib.ticker import MultipleLocator,FormatStrFormatter,MaxNLocator


#External Arguments

Y_ref=sys.argv[1]
X_ref=sys.argv[2]

work_path=sys.argv[3]

pathObs=sys.argv[4]
pathGCM_X_separated=sys.argv[5]
pathGCM_Y_separated=sys.argv[6]

dt_string=sys.argv[7]


### Fixed arguments
scenarios=["ssp119","ssp126","ssp245","ssp370","ssp585"]
sample_pred=10
ci          = 0.05
ns_law      = nsm.GEV()

#Time period of interest
T=1000
T1=2050
T2=2100
deb=1850
fin=2101
time_period    = np.arange( 1850 , 2101 , 1 , dtype = int )


Eq_Reliability_T=(1-1/T)**(T2-T1+1)
##### Paths
print("Figures: "+Y_ref)

basepath=work_path
pathFig=os.path.join(basepath,'./Figures/'+Y_ref+"_"+X_ref,dt_string)
if not os.path.exists(pathFig):
    os.makedirs(pathFig)

pathOut=os.path.join(basepath,'./Outputs/'+Y_ref+"_"+X_ref,dt_string)
if not os.path.exists(pathOut):
    os.makedirs(pathOut)

#Observations Meteo
dXo = xr.open_dataset(os.path.join( pathObs,"Xo_Ano.nc")) #Deja en anomalies
Xo  = pd.DataFrame( dXo.Xo.values.squeeze() , columns = ["Xo"] , index = dXo.index.values )

dYo =  xr.open_dataset(os.path.join( pathObs,"Yo_Ano.nc"))
Yo  =  pd.DataFrame( dYo.Yo.values.squeeze() , columns = ["Yo"] , index = dYo.index.values )

dbias_multi=xr.open_dataset(os.path.join( pathObs,"bias_Obs.nc"))
bias_multi=float(dbias_multi["Multi_Synthesis"].values.squeeze())

#### Posterior figures
print("Figure: posterior")

if not os.path.exists(os.path.join( pathOut , ("Parametres_vstime_posterior"+"_"+dt_string+".nc") )):
    print("Calculate Para")
    #corrected version
    climCXCB=ns.Climatology.from_netcdf(os.path.join( pathOut , ("clim_multiscenarios_CXCB_"+dt_string+".nc")  )  , ns_law  )

    

    samples_MCMC = climCXCB.data.sample_MCMC
    n_x=len(climCXCB.X.sample)-1
    n_ess=int((len(climCXCB.law_coef.sample_MCMC)-1)/n_x)

    XF_noBE=np.tile(climCXCB.X.loc[:,:,"F","Multi_Synthesis"][:,1:], (1, n_ess)).T
    XF_data=np.vstack((XF_noBE,climCXCB.X.loc[:,"BE","F","Multi_Synthesis"]))
    XF=xr.DataArray( XF_data.T, coords = [climCXCB.X.time , samples_MCMC  ] , dims = ["time","sample_MCMC"] )

    locF  = climCXCB.law_coef.loc["loc0",:,"Multi_Synthesis"] + XF * climCXCB.law_coef.loc["loc1",:,"Multi_Synthesis"]
    scaleF = np.exp(climCXCB.law_coef.loc["scale0",:,"Multi_Synthesis"] + XF * climCXCB.law_coef.loc["scale1",:,"Multi_Synthesis"])
    shape = climCXCB.law_coef.loc["shape0",:,"Multi_Synthesis"] + xr.zeros_like(locF)
    RL_F=xr.DataArray(sc.genextreme.ppf((1-1/T),  loc = locF , scale = scaleF , c = - shape).T,
                   coords = [climCXCB.X.time , samples_MCMC  ] ,
                  dims = ["time","sample_MCMC"] )
    para=xr.concat([locF, scaleF,shape], pd.Index(["loc","scale","shape"], name='parametre'))
    para.to_netcdf(os.path.join( pathOut , ("Parametres_vstime_posterior"+"_"+dt_string+".nc") ))
    RL_F.to_netcdf(os.path.join( pathOut , ("RL_F_posterior"+"_"+dt_string+".nc") ))

para=xr.open_dataarray(os.path.join( pathOut , ("Parametres_vstime_posterior"+"_"+dt_string+".nc")))

RL_F=xr.open_dataarray(os.path.join( pathOut , ("RL_F_posterior"+"_"+dt_string+".nc") ))

if not os.path.exists(os.path.join( pathOut , ("ssp585"+"_pred_tirages_EQR_posterior_"+str(T1)+"_"+str(T2)+"_"+dt_string+".nc") )):
    
    samples_MCMC = para.sample_MCMC
    prod=list(product(list(range(sample_pred)),samples_MCMC.values)) 
    samples_pred =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
    time_year=list(range(T1, T2+1))
    n_time=len(time_year)

    for scen in scenarios:
        prod=list(product([scen],time_year))  
        scen_times =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
        shape_s=np.tile(para.loc["shape",:,scen_times ], ( sample_pred,1))
        loc_s=np.tile(para.loc["loc",:,scen_times ], ( sample_pred,1))
        scale_s=np.tile(para.loc["scale",:,scen_times ], ( sample_pred,1))
        tirages_pred=xr.DataArray(sc.genextreme.rvs(c=-shape_s,loc=loc_s,
                            scale=scale_s ).T,
                   coords = [scen_times , samples_pred  ] ,
                  dims = ["time","sample_pred"] ).max('time')
        tirages_pred.to_netcdf(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1)+"_"+str(T2)+"_"+dt_string+".nc") ))


#Save CSV
ofile=os.path.join(  pathFig+"/"+str(T)+"_ci"+str(ci)+"_"+str(T1)+"_"+str(T2)+"_Multiscenario_Values.txt" )
out_table=[]
for scen in scenarios:
    pred=xr.open_dataarray(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1)+"_"+str(T2)+"_"+dt_string+".nc") ))

    predict_EQR=float(pred.quantile(Eq_Reliability_T))+bias_multi
    out_table.append(predict_EQR)

df=pd.DataFrame(out_table,index=scenarios,columns=[Y_ref]).T

with open(ofile, 'a') as f:
    f.write(df.to_string())



import matplotlib.colors as mcolors
#Save future life values to CSV
T1_l=2020
T2_l=2100
ofile=os.path.join(  pathFig+"/"+str(T1_l)+"_"+str(T2_l)+"_Future_LifeValues.txt" )
out_table=[]

prob_interest=[0.5,0.25,0.1,0.05,0.01,0.001]


i=0

df=pd.DataFrame(index=scenarios,columns=prob_interest)
for scen in scenarios:
    
    if not os.path.exists(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1_l)+"_"+str(T2_l)+"_"+dt_string+".nc") )):
        para=xr.open_dataarray(os.path.join( pathOut , ("Parametres_vstime_posterior"+"_"+dt_string+".nc")))
        samples_MCMC = para.sample_MCMC
        prod=list(product(list(range(sample_pred)),samples_MCMC.values)) 
        samples_pred =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
        time_year=list(range(T1_l, T2_l+1))
        n_time=len(time_year)

        prod=list(product([scen],time_year))  
        scen_times =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
        shape_s=np.tile(para.loc["shape",:,scen_times ], ( sample_pred,1))
        loc_s=np.tile(para.loc["loc",:,scen_times ], ( sample_pred,1))
        scale_s=np.tile(para.loc["scale",:,scen_times ], ( sample_pred,1))
        tirages_pred=xr.DataArray(sc.genextreme.rvs(c=-shape_s,loc=loc_s,
                            scale=scale_s ).T,
                   coords = [scen_times , samples_pred  ] ,
                  dims = ["time","sample_pred"] ).max('time')
        tirages_pred.to_netcdf(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1_l)+"_"+str(T2_l)+"_"+dt_string+".nc") ))

    pred=xr.open_dataarray(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1_l)+"_"+str(T2_l)+"_"+dt_string+".nc") ))
    out_table=[]
    for p in prob_interest:

        predict_EQR=float(pred.quantile(1-p))+bias_multi
        
        out_table.append(predict_EQR)
    df.loc[scen]=out_table
    i=i+1

#df=pd.DataFrame(out_table,index=scenarios,columns=prob_interest).T

with open(ofile, 'a') as f:
    f.write(df.to_string())
    
df
per=str(T1_l)+"-"+str(T2_l)+" for "
df2=df.apply(pd.to_numeric).style \
    .format(precision=2, thousands=",", decimal=".")\
    .relabel_index([per+"SSP1-1.9",per+"SSP1-2.6",per+"SSP2-4.5",per+"SSP3-7.0",per+"SSP5-8.5"], axis=0)\
    .relabel_index(["50%","25%","10%","5%","1%","0.1%"], axis=1)\
    .background_gradient(axis=None, vmin=df.min().min(), vmax=df.max().max(), cmap="coolwarm")\

df2.to_latex(buf=pathFig+"/"+str(T1_l)+"_"+str(T2_l)+"_Future_LifeValues_table.tex",convert_css=True)  

#Save past life values to CSV
T1_l=1940
T2_l=2020
ofile=os.path.join(  pathFig+"/"+str(T1_l)+"_"+str(T2_l)+"_Future_LifeValues.txt" )
out_table=[]

prob_interest=[0.5,0.25,0.1,0.05,0.01,0.001]



i=0

df=pd.DataFrame(index=scenarios,columns=prob_interest)
for scen in scenarios:
    
    if not os.path.exists(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1_l)+"_"+str(T2_l)+"_"+dt_string+".nc") )):
        para=xr.open_dataarray(os.path.join( pathOut , ("Parametres_vstime_posterior"+"_"+dt_string+".nc")))
        samples_MCMC = para.sample_MCMC
        prod=list(product(list(range(sample_pred)),samples_MCMC.values)) 
        samples_pred =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
        time_year=list(range(T1_l, T2_l+1))
        n_time=len(time_year)

        prod=list(product([scen],time_year))  
        scen_times =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
        shape_s=np.tile(para.loc["shape",:,scen_times ], ( sample_pred,1))
        loc_s=np.tile(para.loc["loc",:,scen_times ], ( sample_pred,1))
        scale_s=np.tile(para.loc["scale",:,scen_times ], ( sample_pred,1))
        tirages_pred=xr.DataArray(sc.genextreme.rvs(c=-shape_s,loc=loc_s,
                            scale=scale_s ).T,
                   coords = [scen_times , samples_pred  ] ,
                  dims = ["time","sample_pred"] ).max('time')
        tirages_pred.to_netcdf(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1_l)+"_"+str(T2_l)+"_"+dt_string+".nc") ))

    pred=xr.open_dataarray(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1_l)+"_"+str(T2_l)+"_"+dt_string+".nc") ))
    out_table=[]
    for p in prob_interest:

        predict_EQR=float(pred.quantile(1-p))+bias_multi
        
        out_table.append(predict_EQR)
    df.loc[scen]=out_table
    i=i+1

#df=pd.DataFrame(out_table,index=scenarios,columns=prob_interest).T

with open(ofile, 'a') as f:
    f.write(df.to_string())
    
df
per=str(T1_l)+"-"+str(T2_l)+" for "

df2=df.apply(pd.to_numeric).style \
    .format(precision=2, thousands=",", decimal=".")\
    .relabel_index([per+"SSP1-1.9",per+"SSP1-2.6",per+"SSP2-4.5",per+"SSP3-7.0",per+"SSP5-8.5"], axis=0)\
    .relabel_index(["50%","25%","10%","5%","1%","0.1%"], axis=1)\
    .background_gradient(axis=None, vmin=df.min().min(), vmax=df.max().max(), cmap="coolwarm")\

df2.to_latex(buf=pathFig+"/"+str(T1_l)+"_"+str(T2_l)+"_Future_LifeValues_table.tex",convert_css=True) 



#Save past life values to CSV
T1_l=1950
T2_l=2000
ofile=os.path.join(  pathFig+"/"+str(T1_l)+"_"+str(T2_l)+"_Future_LifeValues.txt" )
out_table=[]

prob_interest=[0.5,0.25,0.1,0.05,0.01,0.001]



i=0

df=pd.DataFrame(index=scenarios,columns=prob_interest)
for scen in scenarios:
    
    if not os.path.exists(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1_l)+"_"+str(T2_l)+"_"+dt_string+".nc") )):
        para=xr.open_dataarray(os.path.join( pathOut , ("Parametres_vstime_posterior"+"_"+dt_string+".nc")))
        samples_MCMC = para.sample_MCMC
        prod=list(product(list(range(sample_pred)),samples_MCMC.values)) 
        samples_pred =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
        time_year=list(range(T1_l, T2_l+1))
        n_time=len(time_year)

        prod=list(product([scen],time_year))  
        scen_times =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
        shape_s=np.tile(para.loc["shape",:,scen_times ], ( sample_pred,1))
        loc_s=np.tile(para.loc["loc",:,scen_times ], ( sample_pred,1))
        scale_s=np.tile(para.loc["scale",:,scen_times ], ( sample_pred,1))
        tirages_pred=xr.DataArray(sc.genextreme.rvs(c=-shape_s,loc=loc_s,
                            scale=scale_s ).T,
                   coords = [scen_times , samples_pred  ] ,
                  dims = ["time","sample_pred"] ).max('time')
        tirages_pred.to_netcdf(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1_l)+"_"+str(T2_l)+"_"+dt_string+".nc") ))

    pred=xr.open_dataarray(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1_l)+"_"+str(T2_l)+"_"+dt_string+".nc") ))
    out_table=[]
    for p in prob_interest:

        predict_EQR=float(pred.quantile(1-p))+bias_multi
        
        out_table.append(predict_EQR)
    df.loc[scen]=out_table
    i=i+1

#df=pd.DataFrame(out_table,index=scenarios,columns=prob_interest).T

with open(ofile, 'a') as f:
    f.write(df.to_string())
    
df
per=str(T1_l)+"-"+str(T2_l)+" for "

df2=df.apply(pd.to_numeric).style \
    .format(precision=2, thousands=",", decimal=".")\
    .relabel_index([per+"SSP1-1.9",per+"SSP1-2.6",per+"SSP2-4.5",per+"SSP3-7.0",per+"SSP5-8.5"], axis=0)\
    .relabel_index(["50%","25%","10%","5%","1%","0.1%"], axis=1)\
    .background_gradient(axis=None, vmin=df.min().min(), vmax=df.max().max(), cmap="coolwarm")\

df2.to_latex(buf=pathFig+"/"+str(T1_l)+"_"+str(T2_l)+"_Future_LifeValues_table.tex",convert_css=True) 








scenarios=["ssp119","ssp245","ssp585"]
ofile=os.path.join(  pathFig+"/"+str(T)+"_ci"+str(ci)+"_"+str(T1)+"_"+str(T2)+"_Multiscenario_Comp_RL_vrs_EQR_"+str(scenarios[-1])+"_dfast.pdf" )
pdf= mpdf.PdfPages( ofile )
fig = plt.figure( figsize = (12,8) )
sns.set_context("talk")

p=1/T

label = ["Project lifetime",
         "Quantile $z_p(t)$ for p="+"{:.0e}".format(p)+" SSP 1-1.9",
         "Quantile $z_p(t)$ for p="+"{:.0e}".format(p)+" SSP 1-2.6",
        "Quantile $z_p(t)$ for p="+"{:.0e}".format(p)+" SSP 2-4.5",
        "Quantile $z_p(t)$ for p="+"{:.0e}".format(p)+" SSP 3-7.0",
        "Quantile $z_p(t)$ for p="+"{:.0e}".format(p)+" SSP 5-8.5",
         "Observations (ERA5)",
         "PER level for SSP 1-1.9",
         "PER level for SSP 1-2.6",
         "PER level for SSP 2-4.5",
         "PER level for SSP 3-7.0",
         "PER level for SSP 5-8.5"
        ]

color_EQR={"ssp119":'#009999',"ssp126":'#003d99',"ssp245":'#997a00',"ssp370":'#990000',"ssp585":'#4d0000'}
color_RL={"ssp119":"#00c3ff","ssp126":"#0066ff","ssp245":"#ffcc00","ssp370":"#ff5050","ssp585":"#800000"}
ax = fig.add_subplot( 1 , 1 ,  1 )



time_year=list(range(1850, 2101))
n_time=len(time_year)
for scen in scenarios:
    prod=list(product([scen],time_year))  
    scen_times =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])
    Q_1000_quantiles=RL_F.loc[scen_times,:].quantile( [ ci / 2 , 1 - ci / 2 , 0.5 ] , dim = "sample_MCMC" ).assign_coords( quantile = ["ql","qu","BE"] )
    ax.plot( time_period ,
        Q_1000_quantiles.loc["BE",scen_times] +bias_multi,
        color = color_RL[scen] ,
        label="Quantile $z_p(t)$ for p="+"{:.0e}".format(p)+" "+scen.upper())
    ax.fill_between( time_period , 
                Q_1000_quantiles.loc["ql",scen_times] +bias_multi,
                Q_1000_quantiles.loc["qu",scen_times]+bias_multi ,
                color = color_RL[scen] ,
                alpha = 0.3 )
    
    
    pred=xr.open_dataarray(os.path.join( pathOut , (scen+"_pred_tirages_EQR_posterior_"+str(T1)+"_"+str(T2)+"_"+dt_string+".nc") ))

    predict_EQR=float(pred.quantile(Eq_Reliability_T))

    plt.hlines(y=predict_EQR+bias_multi,
           xmin=T1, xmax=T2,
            color=color_EQR[scen],
            linestyle='-',
            label="PER level for "+scen.upper())
    ax.text(T2+2, predict_EQR+bias_multi, '%.1f'%(predict_EQR+bias_multi)+"°C",
        verticalalignment='center', horizontalalignment='left',
        #transform=ax.transAxes,
        color=color_EQR[scen])#, fontsize=15)

    
    
ax.scatter( Yo.index, Yo+bias_multi,
           color = "black" ,
           label=label[6])
ax.axvspan(T1, T2,
           alpha=0.03,
           color='blue',
           label=label[0])




#label = ["Time period","RL 1000y", "RL 2y", "Obs", "EQR"]
legend = [mplpatch.Patch(facecolor = c , edgecolor = c , label = l , alpha = 0.5 ) for c,l in zip(["blue","red","grey","black","green"],label)]

fig.set_tight_layout(True)
ax.set(xlim=(1935,np.max(time_period)))
ax.xaxis.set_major_locator(MultipleLocator(25))


plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%d°C'))
ax.yaxis.tick_right()
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_minor_locator(MultipleLocator(1))

plt.grid(alpha=0.3)

#ax.legend( frameon=False, )#handles = legend ,loc='upper left')
leg=ax.legend( )
leg.get_frame().set_edgecolor('white')

pdf.savefig(fig, bbox_inches = 'tight')
plt.close(fig)
pdf.close()




### Parameter distribution
print("Parameter distribution")

clim_MM=ns.Climatology.from_netcdf( os.path.join( pathOut , ("clim_multiscenarios_multisynthesis_"+dt_string+".nc"))  , ns_law   )
climCXCB=ns.Climatology.from_netcdf(os.path.join( pathOut , ("clim_multiscenarios_CXCB_"+dt_string+".nc")  )  , ns_law  )
clim=ns.Climatology.from_netcdf( os.path.join( pathOut , ("clim_multiscenarios_multisynthesis_allmodels_"+dt_string+".nc"))  , ns_law   )

law = nsm.GEV()
ind=list(set(Yo.index).intersection(Xo.index))
law.fit(Yo.loc[ind].values,Xo.loc[ind].values)
theta_obs=law.get_params()
theta_obs


#Only prior
custom_params = {"axes.spines.bottom": True,"axes.spines.right": True,"axes.spines.left": True, "axes.spines.top":True}
sns.set_theme(style="whitegrid",rc=custom_params)
    
ofile=pathFig+"/"+dt_string+"_Comp_Distributions_Para_prior_posterior.pdf" 
pdf = mpdf.PdfPages( ofile )

qcoefX=climCXCB.law_coef[:,1:,:].quantile( [ci/2,1-ci/2,0.5] , dim = 'sample_MCMC' ).assign_coords( quantile = ["ql","qu","BE"] )

ymin = float( (climCXCB.law_coef ).min())
ymax = float( (climCXCB.law_coef ).max())
delta = 0.1 * (ymax-ymin)
ylim = (ymin-delta,ymax+delta)

kwargs = {  "showmeans" : False , "showextrema" : False , "showmedians" : True }
models=climCXCB.data.model.values
m='Multi_Synthesis'
fig = plt.figure( figsize = ( 16 , 10 ) )

sub_def=climCXCB.law_coef.coef.values#[2:4]
para_names=climCXCB.ns_law.get_params_names(True)#[2:4]
label = ["Prior","Posterior",'Observation ML fit']
legend = [mplpatch.Patch(facecolor = c , edgecolor = c , label = l , alpha = 0.5 ) for c,l in zip(["blue","red","green"],label)]


for i in range(len(sub_def)):
    
    ax = fig.add_subplot(1,len(sub_def),i+1)
   
    vplot = ax.violinplot( ((climCXCB.law_coef) )[:,1:,:].loc[sub_def[i],:,m]  , **kwargs )
    vplotf = ax.violinplot( ((clim_MM.law_coef) )[:,1:,:].loc[sub_def[i],:,m]  , **kwargs )

    
    for pc in vplot["bodies"]:
        pc.set_facecolor("red")
        pc.set_edgecolor("red")
        pc.set_alpha(0.5)
    vplot["cmedians"].set_color("red")  

    for pc in vplotf["bodies"]:
            pc.set_facecolor("blue")
            pc.set_edgecolor("blue")
            pc.set_alpha(0.3)
    vplotf["cmedians"].set_color("blue")
    #ax.hlines( newDF[newDF.columns[i]].quantile( [ci/2,1-ci/2,0.5]  ).values , 1 - 0.1 , 1 + 0.1 , color = "blue",label="pystan" )
    
    #for q in ["BE"]:
    #            a=ax.hlines( qcoefX[:,i,:].loc[q,m] , 1 - 0.1 , 1 + 0.1 , color = "red")#, label= "NSSEA" )
    #            ax.hlines( qcoef[:,i,:].loc[q,m] , 1 - 0.1 , 1 + 0.1 , color = "blue")#,label="No" )
                #ax.hlines( qcoefX_stan[:,i,:].loc[q,m] , 1 - 0.1 , 1 + 0.1 , color = "green",label="Stan" )
    #ax.vlines( 1 , qcoefX[:,i,:].loc["BE",m] , qcoef[:,i,:].loc["BE",m] , color = "grey" )
    ax.set_xticks([1])
    
    xticks = [ para_names[i] ]
    ax.set_xticklabels( xticks , fontsize = 13 )
    if ((clim.law_coef) )[:,1:,:].loc[sub_def[i],:,m].min()*((clim.law_coef) )[:,1:,:].loc[sub_def[i],:,m].max()<0:
        ax.hlines( 0 , 1-0.3 , 1+0.3, color = "black" )
    gcm_para=clim.data.law_coef.loc[sub_def[i],'BE' ,:].values[:-1]

    ax.scatter([1]*len(gcm_para),gcm_para,color="blue")
    #Add obs
    ax.hlines( theta_obs[i] , 1-0.3 , 1+0.3, color = "green",linewidth=4 )
    
    if i ==0:
        ax.legend( handles = legend , fontsize = 20,loc='upper left' )
#ax.legend(labels=["posterior","prior"],fontsize = 20)

fig.set_tight_layout(True)

#fig.suptitle( "With X sampling", fontsize = 25 )
pdf.savefig(fig)
plt.close(fig)
pdf.close()



##QQ Plot
print("QQ plot")
import statsmodels.api as sm

qparamsX =para[:,1:,:].quantile( [ ci / 2 , 1 - ci / 2 , 0.5 ] , dim = "sample_MCMC" ).assign_coords( quantile = ["ql","qu","BE"] )

fig = plt.figure( figsize = (15,5) )
j=1
for scen in scenarios:
	#KS test
	print(scen)
	prod=list(product([scen],Yo.index))  
	scen_times =   np.array([str(prod[i][1])+"_"+str(prod[i][0]) for i in range(len(prod))])

	loc   = qparamsX.loc["BE",'loc',scen_times].values
	scale = qparamsX.loc["BE",'scale',scen_times].values
	shape = qparamsX.loc["BE",'shape',scen_times].values
	Z = ( Yo.values.squeeze() - loc ) / scale
	print(sc.kstest( Z , lambda x : sc.genextreme.cdf( x , loc = 0 , scale = 1 , c = - shape )) )#p-value > 0.05, no rejection.


	ax = fig.add_subplot( 1 , 3, j )
	residuals=[]
	for i in range(0,len(Yo)):
    		shape=qparamsX.loc["BE",'shape',str(Yo.index[i])+"_"+scen].values
    		loc=qparamsX.loc["BE",'loc',str(Yo.index[i])+"_"+scen].values
    		scale=qparamsX.loc["BE",'scale',str(Yo.index[i])+"_"+scen].values
        
    		residuals.append(((Yo.iloc[i].values[0]-loc)/scale))

	sm.qqplot(np.asarray(residuals),dist=sc.genextreme,distargs=(-shape,), line="45",ax=ax)
	ax.set_ylabel("Sample Quantile for Observations" )
	#ax.set_xlabel("Théorique Posterior (GEV)" )
	ax.set_title(scen)
	j=j+1

plt.savefig(os.path.join( pathFig ,X_ref+"_"+Y_ref+"AnnexeFigure6.pdf"))
plt.show()
