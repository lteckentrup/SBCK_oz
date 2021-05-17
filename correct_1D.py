import sys
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

def Bias_Correction(var, model, method, method_long):
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

    correct_params = []
    for i,lat in enumerate(lats):
        for j,lon in enumerate(lons):
            params_dict = {}
            if (np.isnan(sim_HIST_values[0,i,j]) or
                np.isnan(obs_values[0,i,j])):
                bc_HIST[:,i,j] = np.nan
                params_dict['lat'] = lat
                params_dict['lon'] = lon
                params_dict['params'] = np.nan
            else:
                try:
                    HIST = sim_HIST_values[:,i,j]
                    OBS = obs_values[:,i,j]
                    COR = sim_COR_values[:,i,j]

                    if method == 'QM':
                        qm = bc.QM()
                        qm.fit(OBS,HIST)
                        HIST_BC = qm.predict(HIST)
                        COR_BC = qm.predict(COR)

                    elif method == 'CDFt':
                        cdft = bc.CDFt()
                        cdft.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC  = cdft.predict(COR, HIST)

                    elif method == 'OTC_univ':
                        otc = bc.OTC()
                        otc.fit(OBS, HIST)
                        HIST_BC = otc.predict(HIST)
                        COR_BC = otc.predict(COR)

                    elif method == 'dOTC':
                        dotc = bc.dOTC()
                        dotc.fit(OBS, HIST, COR)
                        COR_BC, HIST_BC = dotc.predict(COR, HIST)

                    bc_HIST[:,i,j] = HIST_BC.flatten()
                    bc_COR[:,i,j] = COR_BC.flatten()

                    if i%5==0 and j%5==0:
                        print(lat,lon)

                    params_dict['lat'] = lat
                    params_dict['lon'] = lon
                    params_dict['params'] = HIST_BC

                except:
                    bc_hist[:,i,j] = np.nan
                    bc_COR[:,i,j] = np.nan

                    params_dict['lat'] = lat
                    params_dict['lon'] = lon
                    params_dict['params'] = np.nan

            correct_params.append(params_dict)

    ds_bc_HIST = xr.Dataset({var:(('time', 'lat','lon'),
                                  bc_HIST)},
                            coords={'lat': lats,
                                    'lon': lons,
                                    'time':HIST_dict['time']})
    ds_bc_COR = xr.Dataset({var:(('time', 'lat','lon'),
                                 bc_COR)},
                           coords={'lat': lats,
                                   'lon': lons,
                                   'time':COR_dict['time']})

    ds_bc_HIST['lat'].attrs={'units':'degrees_north',
                             'long_name':'latitude',
                             'standard_name':'latitude',
                             'axis':'Y'}
    ds_bc_COR['lat'].attrs={'units':'degrees_north',
                            'long_name':'latitude',
                            'standard_name':'latitude',
                            'axis':'Y'}
    ds_bc_HIST['lon'].attrs={'units':'degrees_east',
                             'long_name':'longitude',
                             'standard_name':'longitude',
                             'axis':'X'}
    ds_bc_COR['lon'].attrs={'units':'degrees_east',
                            'long_name':'longitude',
                            'standard_name':'longitude',
                            'axis':'X'}

    if var == 'temp':
        ds_bc_HIST['temp'].attrs={'long_name':'Temperature at 2m',
                                  'standard_name':'air_temperature',
                                  'units':'K'}
        ds_bc_COR['temp'].attrs={'long_name':'Temperature at 2m',
                                 'standard_name':'air_temperature',
                                 'units':'K'}
    elif var == 'prec':
        ds_bc_HIST['prec'].attrs={'long_name':'Precipitation',
                                  'standard_name':'precipitation_amount',
                                  'units':'kg m-2'}
        ds_bc_COR['prec'].attrs={'long_name':'Precipitation',
                                 'standard_name':'precipitation_amount',
                                 'units':'kg m-2'}

    elif var == 'insol':
        ds_bc_HIST['insol'].attrs={'long_name':'Downward solar radiation flux',
                                   'standard_name':'surface_downwelling_shortwave_flux',
                                   'units':'W m-2'}
        ds_bc_COR['insol'].attrs={'long_name':'Downward solar radiation flux',
                                  'standard_name':'surface_downwelling_shortwave_flux',
                                  'units':'W m-2'}

    ds_bc_HIST.attrs={'Conventions':'CF-1.6',
                      'Model':model+' CMIP6',
                      'Experiment':'SSP245',
                      'Realisation':'r1i1p1f1',
                      'Correctionmethod': method_long,
                      'Date_Created':str(date_created)}
    ds_bc_COR.attrs={'Conventions':'CF-1.6',
                     'Model':model+' CMIP6',
                     'Experiment':'SSP245',
                     'Realisation':'r1i1p1f1',
                     'Correctionmethod': method_long,
                     'Date_Created':str(date_created)}

    ds_bc_HIST.to_netcdf(method+'_'+var+'_'+model+'_HIST.nc',
                         encoding={'time':{'dtype': 'double'},
                                   'lat':{'dtype': 'double'},
                                   'lon':{'dtype': 'double'},
                                   var:{'dtype': 'float32'}
                                   }
                         )
    ds_bc_COR.to_netcdf(method+'_'+var+'_'+model+'_COR.nc',
                        encoding={'time':{'dtype': 'double'},
                                  'lat':{'dtype': 'double'},
                                  'lon':{'dtype': 'double'},
                                  var:{'dtype': 'float32'}
                                  }
                        )



model_names = ['CanESM5', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'EC-Earth3',
               'EC-Earth3-Veg', 'GFDL-CM4', 'IITM-ESM', 'INM-CM4-8',
               'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC6',
               'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM',
               'NorESM2-MM']
variables = ['temp', 'prec', 'insol']

for v in variables:
    # Bias_Correction(v, 'CanESM5', 'QM', 'Quantile Mapping')
    # Bias_Correction(v, 'CanESM5', 'CDFt', 'Quantile Mapping bias, taking account '
    #                    'of an evolution of the distribution')
    Bias_Correction(v, 'CanESM5', 'OTC_univ', 'Optimal Transport bias Corrector')

# Bias_Correction('prec', 'CanESM5', 'dOTC', 'Dynamical Optimal Transport'
                                           # 'bias Corrector')

### Ideally it'd be a loop like this

# for mn in model_names:
#     for v in variables:
#         Bias_Correction('prec', 'CanESM5', 'QM', 'Quantile Mapping')
#         Bias_Correction('prec', 'CanESM5', 'CDFt', 'Quantile Mapping bias, '
#                                                    'taking account of an evolution
#                                                    'of the distribution')
#         Bias_Correction('prec', 'CanESM5', 'OTC_univ', 'Optimal Transport bias '
#                                                        'Corrector')
#         Bias_Correction('prec', 'CanESM5', 'dOTC', 'Dynamical Optimal Transport'
#                                                    'bias Corrector')

process = psutil.Process(os.getpid())
print(process.memory_info().rss/(1024 ** 2))
print(datetime.now() - startTime)
