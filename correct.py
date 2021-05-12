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
    observed = ('../../../CRUJRA/'+var+'/crujra.v2.0.'+var+'.std-ordering.nc')
    prcp_hist = ('../../../../australia_climate/'+var+'/'+var+'_'+model+
                 '_SSP245_r1i1p1f1_K_1850_2100.nc')
    prcp_COR = ('../../../../australia_climate/'+var+'/'+var+'_'+model+
                '_SSP245_r1i1p1f1_K_1850_2100.nc')

    observed = xr.open_dataset(observed)
    model_hist = xr.open_dataset(prcp_hist)
    model_COR = xr.open_dataset(prcp_COR)

    # observed = observed.sel(time = slice('1951-01-01','2015-12-31'))
    # model_hist = model_hist.sel(time = slice('1951-01-01','2015-12-31'))
    # model_COR = model_COR.sel(time = slice('1951-01-01','2015-12-31'))

    observed = observed.sel(time = slice('1989-01-01','2010-12-31'))
    model_hist = model_hist.sel(time = slice('1989-01-01','2010-12-31'))
    model_COR = model_COR.sel(time = slice('1989-01-01','2010-12-31'))

    observed = observed[var]
    model_hist = model_hist[var]
    model_COR = model_COR[var]

    lats = observed.lat.values
    lons = observed.lon.values

    bias_corrected_results_hist = np.zeros([len(model_hist.time.values),
                                            len(model_hist.lat.values),
                                            len(model_hist.lon.values)])
    bias_corrected_results_hist[:] = np.nan

    bias_corrected_results_COR = np.zeros([len(model_COR.time.values),
                                           len(model_COR.lat.values),
                                           len(model_COR.lon.values)])
    bias_corrected_results_COR[:] = np.nan

    if var == 'prec':
        model_hist_values = model_hist.values*86400
    else:
        model_hist_values = model_hist.values
    hist_dict = {}
    hist_dict['time'] = model_hist.time.values
    hist_dict['lon'] =  model_hist.lon.values
    hist_dict['lat'] =  model_hist.lat.values

    if var == 'prec':
        modelCOR_values = model_COR.values*86400
    else:
        modelCOR_values = model_COR.values

    COR_dict = {}
    COR_dict['time'] = model_COR.time.values
    COR_dict['lon'] =  model_COR.lon.values
    COR_dict['lat'] =  model_COR.lat.values

    observation_attr_values = observed.values

    correct_params = []
    for i,lat in enumerate(lats):
        for j,lon in enumerate(lons):
            params_dict = {}
            if (np.isnan(model_hist_values[0,i,j]) or
                np.isnan(observation_attr_values[0,i,j])):
                bias_corrected_results_hist[:,i,j] = np.nan
                params_dict['lat'] = lat
                params_dict['lon'] = lon
                params_dict['params'] = np.nan
            else:
                # try:
                X0 = model_hist_values[:,i,j]
                X1 = observation_attr_values[:,i,j]
                Y0 = modelCOR_values[:,i,j]

                if method == 'QM':
                    qm = bc.QM()
                    qm.fit(Y0,X0)
                    Z0 = qm.predict(X0)

                elif method == 'OTC_univ':
                    otc = bc.OTC()
                    otc.fit( Y0 , X0 )
                    Z0 = otc.predict( X0 )

                elif method == 'ECBC':
                    ecbc = bc.ECBC()
                    ecbc.fit( Y0 , X0 , X1 )
                    Z1,Z0 = ecbc.predict(X1,X0)

                elif method == 'QMRS':
                    irefs = [0]
                    qmrs = bc.QMrs( irefs = irefs )
                    qmrs.fit( Y0 , X0 )
                    Z0 = qmrs.predict(X0)

                elif method == 'R2D2':
                    irefs = [0]
                    r2d2 = bc.R2D2( irefs = irefs )
                    r2d2.fit( Y0 , X0 , X1 )
                    Z1 = r2d2.predict(X1)
                    Z  = r2d2.predict(X1,X0)

                elif method == 'QDM':
                	qdm = bc.QDM()
                	qdm.fit( Y0 , X0 , X1 )
                	Z1,Z0 = qdm.predict(X1,X0)

                elif method == 'MBCn':
                	mbcn = bc.MBCn()
                	mbcn.fit( Y0 , X0 , X1 )
                	Z1,Z0 = mbcn.predict(X1,X0)

                elif method == 'MRec':
                	mbcn = bc.MRec()
                	mbcn.fit( Y0 , X0 , X1 )
                	Z1 = mbcn.predict(X1)
                	_,Z0 = mbcn.predict(X1,X0)

                bias_corrected_results_COR[:,i,j] = Z0.flatten()

                if i%5==0 and j%5==0:
                    print(lat,lon)

                params_dict['lat'] = lat
                params_dict['lon'] = lon
                params_dict['params'] = Z0

                # except:
                #     bias_corrected_results_hist[:,i,j] = np.nan
                #     bias_corrected_results_COR[:,i,j] = np.nan
                #
                #     params_dict['lat'] = lat
                #     params_dict['lon'] = lon
                #     params_dict['params'] = np.nan

            correct_params.append(params_dict)

    ds_COR = xr.Dataset({var:(('time', 'lat','lon'),
                                 bias_corrected_results_COR)},
                           coords={'lat': lats,
                                   'lon': lons,
                                   'time':COR_dict['time']})

    ds_COR['lat'].attrs={'units':'degrees_north',
                         'long_name':'latitude',
                         'standard_name':'latitude',
                         'axis':'Y'}
    ds_COR['lon'].attrs={'units':'degrees_east',
                         'long_name':'longitude',
                         'standard_name':'longitude',
                         'axis':'X'}

    if var == 'temp':
        ds_COR['temp'].attrs={'long_name':'Temperature at 2m',
                              'standard_name':'air_temperature',
                              'units':'K'}
    elif var == 'prec':
        ds_COR['prec'].attrs={'long_name':'Precipitation',
                              'standard_name':'precipitation_amount',
                              'units':'kg m-2'}
        # ds_COR['prec']=ds_COR['prec'].where(ds_COR['prec']>0,0)
    elif var == 'insol':
        ds_COR['insol'].attrs={'long_name':'Downward solar radiation flux',
                               'standard_name':'surface_downwelling_shortwave_flux',
                               'units':'W m-2'}
        # ds_COR['insol']=ds_COR['insol'].where(ds_COR['insol']>0,0)

    ds_COR.attrs={'Conventions':'CF-1.6',
                  'Model':model+' CMIP6',
                  'Experiment':'SSP245',
                  'Realisation':'r1i1p1f1',
                  'Correctionmethod': method_long,
                  'Date_Created':str(date_created)}
    ds_COR.to_netcdf(method+'_'+var+'_'+model+'_cor1.nc',
                        encoding={'time':{'dtype': 'double'},
                                  'lat':{'dtype': 'double'},
                                  'lon':{'dtype': 'double'},
                                  var:{'dtype': 'float32'}
                                  }
                        )

Bias_Correction('prec', 'CanESM5', 'QM', 'Quantile Mapping')
Bias_Correction('prec', 'CanESM5', 'OTC_univ', 'Optimal Transport bias Corrector')
Bias_Correction('prec', 'CanESM5', 'ECBC', 'Empirical Copula Bias Correction')
Bias_Correction('prec', 'CanESM5', 'QMRS', 'Quantile Mapping bias corrector with multivariate rankshuffle')
Bias_Correction('prec', 'CanESM5', 'R2D2', 'Non stationnary Quantile Mapping bias corrector with multivariate rankshuffle')
Bias_Correction('prec', 'CanESM5', 'QDM', 'Quantile Delta Mapping')
#Bias_Correction('prec', 'CanESM5', 'MRec', 'MRec Bias correction method')
#Bias_Correction('prec', 'CanESM5', 'MBCn', 'MBCn Bias correction method')


process = psutil.Process(os.getpid())
print(process.memory_info().rss/(1024 ** 2))
print(datetime.now() - startTime)
