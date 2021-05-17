import sys
from numpy import array
import numpy as np
import scipy.stats as sc
import statsmodels.tsa.stattools as stt
import xarray as xr
import pandas as pd

import SBCK as bc
import SBCK.tools as bct
import SBCK.metrics as bcm
import SBCK.datasets as bcd

import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import date
import os, psutil
from datetime import datetime

startTime = datetime.now()
date_created = date.today()

def readin(var, model):
    var_obs = ('../../../CRUJRA/'+var+'/crujra.v2.0.'+var+'.std-ordering.nc')
    var_sim = ('../../../../australia_climate/'+var+'/'+var+'_'+model+
                 '_SSP245_r1i1p1f1_K_1850_2100.nc')

    obs = xr.open_dataset(var_obs)
    sim = xr.open_dataset(var_sim)

    obs = obs.sel(time = slice('1989-01-01','2010-12-31'))
    sim_HIST = sim.sel(time = slice('1989-01-01','2010-12-31'))
    sim_COR = sim.sel(time = slice('1889-01-01','1910-12-31'))

    obs = obs[var]
    sim_HIST = sim_HIST[var]
    sim_COR = sim_COR[var]

    lats = obs.lat.values
    lons = obs.lon.values

    bc_HIST = np.zeros([len(sim_HIST.time.values),
                        len(sim_HIST.lat.values),
                        len(sim_HIST.lon.values)])
    bc_HIST[:] = np.nan

    bc_COR = np.zeros([len(sim_COR.time.values),
                       len(sim_COR.lat.values),
                       len(sim_COR.lon.values)])
    bc_COR[:] = np.nan

    if var == 'prec':
        sim_HIST_values = sim_HIST.values*86400
    else:
        sim_HIST_values = sim_HIST.values

    HIST_dict = {}
    HIST_dict['time'] = sim_HIST.time.values
    HIST_dict['lon'] =  sim_HIST.lon.values
    HIST_dict['lat'] =  sim_HIST.lat.values

    if var == 'prec':
        sim_COR_values = sim_COR.values*86400
    else:
        sim_COR_values = sim_COR.values

    COR_dict = {}
    COR_dict['time'] = sim_COR.time.values
    COR_dict['lon'] =  sim_COR.lon.values
    COR_dict['lat'] =  sim_COR.lat.values

    obs_values = obs.values

    return(obs_values, sim_HIST_values, sim_COR_values, bc_HIST, bc_COR,
           HIST_dict, COR_dict, lats, lons)

def write_netcdf(input, var, model, method, method_long, dic, lats, lons, period):

    dataset = xr.Dataset({var:(('time', 'lat','lon'),
                               input)},
                         coords={'lat': lats,
                                 'lon': lons,
                                 'time':dic['time']})

    dataset['lat'].attrs={'units':'degrees_north',
                          'long_name':'latitude',
                          'standard_name':'latitude',
                          'axis':'Y'}
    dataset['lon'].attrs={'units':'degrees_east',
                          'long_name':'longitude',
                          'standard_name':'longitude',
                          'axis':'X'}

    if var == 'temp':
        dataset[var].attrs={'long_name':'Temperature at 2m',
                            'standard_name':'air_temperature',
                            'units':'K'}
    elif var == 'prec':
        dataset[var].attrs={'long_name':'Precipitation',
                            'standard_name':'precipitation_amount',
                            'units':'kg m-2'}

    elif var == 'insol':
        dataset[var].attrs={'long_name':'Downward solar radiation flux',
                            'standard_name':'surface_downwelling_shortwave_flux',
                            'units':'W m-2'}


    dataset.attrs={'Conventions':'CF-1.6',
                   'Model':model+' CMIP6',
                   'Experiment':'SSP245',
                   'Realisation':'r1i1p1f1',
                   'Correctionmethod': method_long,
                   'Date_Created':str(date_created)}

    dataset.to_netcdf(method+'_'+var+'_'+model+'_'+period+'.nc',
                      encoding={'time':{'dtype': 'double'},
                                'lat':{'dtype': 'double'},
                                'lon':{'dtype': 'double'},
                                var:{'dtype': 'float32'}
                                }
                      )

def Bias_Correction(model, method, method_long):
    temp_obs_values, temp_sim_HIST_values, temp_sim_COR_values, temp_bc_HIST, \
    temp_bc_COR, temp_HIST_dict, temp_COR_dict, lats, lons = readin('temp', model)
    prec_obs_values, prec_sim_HIST_values, prec_sim_COR_values, prec_bc_HIST, \
    prec_bc_COR, prec_HIST_dict, prec_COR_dict, lats, lons = readin('prec', model)
    insol_obs_values, insol_sim_HIST_values, insol_sim_COR_values, insol_bc_HIST, \
    insol_bc_COR, insol_HIST_dict, insol_COR_dict, lats, lons = readin('insol', model)
    #
    # correct_params = []
    for i,lat in enumerate(lats):
        for j,lon in enumerate(lons):
            params_dict = {}
            if (np.isnan(temp_obs_values[0,i,j]) or
                np.isnan(temp_sim_HIST_values[0,i,j]) or
                np.isnan(prec_obs_values[0,i,j]) or
                np.isnan(prec_sim_HIST_values[0,i,j]) or
                np.isnan(insol_obs_values[0,i,j]) or
                np.isnan(insol_sim_HIST_values[0,i,j])):

                temp_bc_HIST[:,i,j] = np.nan
                prec_bc_HIST[:,i,j] = np.nan
                insol_bc_HIST[:,i,j] = np.nan

                temp_bc_COR[:,i,j] = np.nan
                prec_bc_COR[:,i,j] = np.nan
                insol_bc_COR[:,i,j] = np.nan

                params_dict['lat'] = lat
                params_dict['lon'] = lon
                params_dict['params'] = np.nan
            else:

                ### Combine variables to matrix with 3 columns and ntime rows
                OBS = array([temp_obs_values[:,i,j],prec_obs_values[:,i,j],
                             insol_obs_values[:,i,j]]).transpose()
                HIST = array([temp_sim_HIST_values[:,i,j],
                              prec_sim_HIST_values[:,i,j],
                              insol_sim_HIST_values[:,i,j]]).transpose()
                COR = array([temp_sim_COR_values[:,i,j],
                             prec_sim_COR_values[:,i,j],
                             insol_sim_COR_values[:,i,j]]).transpose()

                if method == 'OTC_biv':
                    otc = bc.OTC()
                    otc.fit(OBS, HIST)
                    HIST_BC = otc.predict(HIST)
                    COR_BC = otc.predict(COR)

                elif method == 'dOTC':
                    dotc = bc.dOTC()
                    dotc.fit(OBS, HIST, COR)
                    COR_BC, HIST_BC = dotc.predict(COR, HIST)

                elif method == 'ECBC':
                	irefs = [0]

                	ecbc = bc.ECBC()
                	ecbc.fit(OBS, HIST, COR)
                	COR_BC, HIST_BC = ecbc.predict(COR, HIST)

                elif method == 'QMRS':
                    irefs = [0]
                    qmrs = bc.QMrs(irefs = irefs)
                    qmrs.fit(OBS, HIST)
                    HIST_BC = qmrs.predict(HIST)
                    COR_BC = qmrs.predict(COR)

                elif method == 'R2D2':
                    irefs = [0]
                    r2d2 = bc.R2D2(irefs = irefs)
                    r2d2.fit(OBS, HIST, COR)
                    COR_BC, HIST_BC = r2d2.predict(COR, HIST)

                elif method == 'QDM':
                    qdm = bc.QDM()
                    qdm.fit(OBS, HIST, COR)
                    COR_BC, HIST_BC = qdm.predict(COR, HIST)

                elif method == 'MBCn':
                    mbcn = bc.MBCn()
                    mbcn.fit(OBS, HIST, COR)
                    COR_BC, HIST_BC = mbcn.predict(COR, HIST)

                elif method == 'MRec':
                    mbcn = bc.MRec()
                    mbcn.fit(OBS, HIST, COR)
                    COR_BC, HIST_BC = mbcn.predict(COR, HIST)

                elif method == 'RBC':
                    rbc = bc.RBC()
                    rbc.fit(OBS, HIST, COR)
                    COR_BC, HIST_BC = rbc.predict(COR, HIST)

                ### Write bias corrected values into bias correction matrix
                ### Historical corrected
                temp_bc_HIST[:,i,j] = HIST_BC[:,0].flatten()
                prec_bc_HIST[:,i,j] = HIST_BC[:,1].flatten()
                insol_bc_HIST[:,i,j] = HIST_BC[:,2].flatten()

                ### Projected corrected
                temp_bc_COR[:,i,j] = COR_BC[:,0].flatten()
                prec_bc_COR[:,i,j] = COR_BC[:,1].flatten()
                insol_bc_COR[:,i,j] = COR_BC[:,2].flatten()

                if i%5==0 and j%5==0:
                    print(lat,lon)

    ### Corrected historical temperature to netcdf
    write_netcdf(temp_bc_HIST, 'temp', model, method, method_long,
                 temp_HIST_dict, lats, lons, 'HIST')
    write_netcdf(temp_bc_COR, 'temp', model, method, method_long,
                 temp_COR_dict, lats, lons, 'COR')
    write_netcdf(prec_bc_HIST, 'prec', model, method, method_long,
                 prec_HIST_dict, lats, lons, 'HIST')
    write_netcdf(prec_bc_COR, 'prec', model, method, method_long,
                 prec_COR_dict, lats, lons, 'COR')
    write_netcdf(insol_bc_HIST, 'insol', model, method, method_long,
                 insol_HIST_dict, lats, lons, 'HIST')
    write_netcdf(insol_bc_COR, 'insol', model, method, method_long,
                 insol_COR_dict, lats, lons, 'COR')

# Bias_Correction('CanESM5', 'OTC_biv', 'Optimal Transport bias Corrector')
Bias_Correction('CanESM5', 'dOTC', 'Dynamical Optimal Transport bias Corrector')
# Bias_Correction('CanESM5', 'ECBC', 'Empirical Copula Bias Correction')
# Bias_Correction('CanESM5', 'QMRS', 'Quantile Mapping with multivariate rankshuffle')
# Bias_Correction('CanESM5', 'R2D2', 'Rank Resampling For Distributions and Dependences')
# Bias_Correction('CanESM5', 'QDM', 'Quantile Delta Mapping')
# Bias_Correction('CanESM5', 'MBCn', 'Multivariate Bias Correction with N-dimensional probability density function transform')
# Bias_Correction('CanESM5', 'MRec', 'Matrix Recorrelation')
# Bias_Correction('CanESM5', 'RBC', 'Random Bias Correction')
process = psutil.Process(os.getpid())
print(process.memory_info().rss/(1024 ** 2))
print(datetime.now() - startTime)
