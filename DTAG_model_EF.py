'''
@author: bramkaarga
'''

import pandas as pd
import numpy as np
import os
import sys
import shutil
import random
from backports import tempfile
import matplotlib.pyplot as plt
import operator

import metronamica_helper as mh #from the helper folder
import process_raster as pr
import geonamica_model_v01 as gnm
#import QUEFTS_v01 as quefts

class MetronamicaCoupled(object):
    def __init__(self, 
                 start_yr, 
                 end_yr, 
                 lu_map, #initial land-use map
                 geoproj, #metronamica land-use model geoproj file
                 log_option, #.log file to dynamically store the simulated land-use maps
                 fert_map, #fertilizer application map. default: 625kg / cell (based on Fig.3 in doi:10.1016/j.jenvman.2018.03.116.)
                 soil_map, #soil categories map based on HWSD
                 districts_map, 
                 river_acc_map, #NO LONGER USED. river accessibility map for land-use change model
                 flood_coeff_map_all, #regression slope map for annual flood metamodel
                 flood_intercept_map_all, #regression intercept map for annual flood metamodel
                 flood_coeff_map_jul, #regression slope map for july flood metamodel
                 flood_intercept_map_jul, #regression intercept map for july flood metamodel
                 sediment_rate_map, #relative sedimentation potential map (based on Fig.7 in doi:10.5194/hess-18-3033-2014)
                 clim_scen = 1, #for generating hydrograph. 1: rcp4.5, 2: rcp8.5, val:historical
                 lud_scen = 1, #land-use demand scenario. 1: BAU, 2: shift back to 2-rice, 3: heavy urbanization, 4: rising non-rice
                 model_folder = os.getcwd(),
                 coop_discharge=None, #for cooperative upstream discharge policy
                 shallow_thres=0.5, #threshold to differentiate shallow and deep inundation; 1.5m based on doi:10.5194/nhess-18-2859-2018
                 shallow_flood_sediment=0.1, #sedimentation rate in shallow inundation area
                 nutrient_in_sediment=0.067, #for sedimentation, we assume that nutrient content in sediment is 6.7% (based on doi:10.5194/hess-18-3033-2014)
                 TE=0, #trapping efficiency of upstream dam, for dam construction scenarios
                 pol_dike_switch=0, #1: all high dikes after 2020, 2: all low dikes after 2020
                 seed_upgrade=False, #seed upgrade policy
                 pol_fer=0, #1: 375kg to districts far from river, 2: 375kg to poor districts, 3: 50kg to districts far from river
                 pol_discharge=0,
                 pol_seed=0,
                 province_map='Data//DTAG_province.tif',
                 init_nutrient=100000,
                 yield_adjust_SA=0.24,
                 yield_adjust_AW=0.3,
                 cost_scaling_factor=1, #based on the data, fertilizer cost is only 25-40% of total cost
                 val_discharge_reduction=None, #for validation purpose
                 val_sedimentation_reduction=None, #for validation purpose
                 p=True):
        
        random.seed(1)
        np.random.seed(1)
        
        #simulation time horizon
        self.start_yr = start_yr
        self.current_yr = start_yr
        self.end_yr = end_yr
        self.count_yr = 0 
        
        #printing option
        self.p = p
        
        #maps
        self.lu_map = pr.get_landuse_array(lu_map)
        self.fert_map = pr.get_landuse_array(fert_map)         
        self.soil_map = pr.get_landuse_array(soil_map)
        self.districts_map = pr.get_landuse_array(districts_map)
        self.river_acc_map = pr.get_landuse_array(river_acc_map)
        self.sediment_rate_map = pr.get_landuse_array(sediment_rate_map)
        self.yield_map = np.zeros((self.lu_map.shape[0], self.lu_map.shape[1]))
        self.profit_map = np.zeros((self.lu_map.shape[0], self.lu_map.shape[1]))
        self.subsidence_map = np.zeros((self.lu_map.shape[0], self.lu_map.shape[1]))
        self.flood_map_check = np.zeros((self.lu_map.shape[0], self.lu_map.shape[1]))
        self.fertilizer_supply = np.zeros((self.lu_map.shape[0], self.lu_map.shape[1]))
        
        self.flood_coeff_map_all = pr.get_landuse_array(flood_coeff_map_all)
        self.flood_intercept_map_all = pr.get_landuse_array(flood_intercept_map_all)
        self.flood_coeff_map_jul = pr.get_landuse_array(flood_coeff_map_jul)
        self.flood_intercept_map_jul = pr.get_landuse_array(flood_intercept_map_jul)
        
        #Metronamica setup files
        self.geoproj = geoproj
        self.log_option = log_option
        self.model_folder = model_folder
        
        #sedimentation parameters
        self.shallow_thres = shallow_thres
        self.shallow_flood_sediment = shallow_flood_sediment
        self.nutrient_in_sediment = nutrient_in_sediment
        
        #other parameters
        self.init_nutrient = init_nutrient
        self.seed_upgrade = seed_upgrade
        self.yield_adjust_SA = yield_adjust_SA
        self.yield_adjust_AW = yield_adjust_AW
        self.cost_scaling_factor = cost_scaling_factor
        
        #scenario parameters
        self.clim_scen = clim_scen
        self.lud_scen = lud_scen
        self.TE = TE
        
        #percentage of N,P,K from fertilizer, adopted from Table 10.2, then adjusted with Table 10.6 in:
            #Site-specific nutrient management in irrigated rice systems of the Mekong Delta of Vietnam
        self.In_perc = np.random.triangular(0.1013, 0.1513, 0.3613, size=self.lu_map.shape)
        self.Ip_perc = np.random.triangular(0.0142, 0.03753, 0.1042, size=self.lu_map.shape)
        self.Ik_perc = np.random.triangular(0.0695, 0.0912, 0.1645, size=self.lu_map.shape)
               
        #replace maps of nutrients_stock and water with inital values
        src = self.model_folder+'//Data//nutrients_stock_init'
        dst = self.model_folder+'//Data//nutrients_stock'
        shutil.rmtree(dst)
        shutil.copytree(src, dst)
        src = os.getcwd()+'\\Data\\water_init'
        dst = os.getcwd()+'\\Data\\water'
        shutil.rmtree(dst)
        shutil.copytree(src, dst)
        src = self.model_folder+'\\Legends\\'+self.geoproj
        dst = self.model_folder+'\\'+self.geoproj
        shutil.copyfile(src, dst)   
        
        #outcomes
        self.yield_district_end = {}
        self.yield_district_start = {}
        self.yield_district_all = {}
        self.profit_district_end = {}
        self.profit_district_end_not_normalized = {}
        self.profit_district_start = {}
        self.profit_district_all = {}
        self.annual_rice_production = {}
        self.annual_rice_production_0 = {}
        self.total_rice_production = 0
        self.annual_flood_map = {}
        self.annual_potential_sedimentation = {}
        self.average_profit_district = {} #normalized per season
        self.average_profit_district_not_normalized = {}
        self.average_profit_all = {}
        self.worst_flood = np.zeros((self.lu_map.shape[0], self.lu_map.shape[1]))
        
        #validation variables
        self.v_revenue_avrg = {}
        self.v_revenue_double = {}
        self.v_revenue_triple = {}
        self.v_profit_avrg = {}
        self.v_profit_double = {}
        self.v_profit_triple = {}
        self.v_yield_avrg = {}
        self.v_yield_double = {}
        self.v_yield_triple = {}
        self.v_yield_map = {}
        self.province_map = pr.get_landuse_array(province_map)
        self.v_yield_dt = {}
        self.v_yield_ag = {}
        self.v_max_wl_tanchau = {}
        self.v_max_wl_chaudoc = {}
        self.v_sediment_supply = {}
        self.v_sedimentation_rate = {}
        self.v_yield_WSC = {}
        self.v_yield_SAC = {}
        self.v_yield_AWC = {}
        self.v_loses = {}
        self.v_discharge_reduction = val_discharge_reduction
        self.v_sedimentation_reduction = val_sedimentation_reduction
        
        #policy variables
        self.coop_discharge = coop_discharge
        
        self.pol_dike_switch = pol_dike_switch

        self.pol_fer = False
        if pol_fer==1:
            fert_map =  'Data//DTAG_fertilizer_pol1.tif'
        elif pol_fer==2:
            fert_map = 'Data//DTAG_fertilizer_pol2.tif'
        elif pol_fer==3:
            self.pol_fer = True
        self.fert_map = pr.get_landuse_array(fert_map)
            
        if pol_discharge==1:
            self.coop_discharge = 50000
            
        if pol_seed==1:
            self.seed_upgrade = True            
                
    def run(self):
    
        #04: scenarios generation
        self._climate_scenario()
        self._landuse_scenario()
        
        random.seed(1)
        np.random.seed(1)
        
        for n, i in enumerate(range(self.end_yr-self.start_yr+1)):
            
            if self.p:
                print(self.current_yr)
            
            #12: land subsidence
            self._subsidence()
            
            #15: Run QUEFTS
            #rice_yield is expected yield in one season
            #rice_yield is only for land-use class 1,2,3,4. In other land-use classes the yield is 0
            rice_yield = self.run_QUEFTS_(n) #run_QUEFTS_ assume 1x crop season
            
            #17: Generate flood map
            flood_map_all, flood_map_jul = self._generate_flood_map()
            self.annual_flood_map[self.current_yr] = flood_map_all
            
            #19: adjust yield based on flood damage
            self.yield_map = self._adjust_yield(rice_yield, flood_map_all, flood_map_jul) #self.yield_map is now the annual yield_map (not seasonal)
            
            #21: calculate inflow sediment
            self._deposit_sediment3(flood_map_all)
            
            #25: calculate total rice production (ton), with 3/2/1-crop per year based on the LU map
            self._record_yield()
            
            #27: calculate profit of farmers, with 3/2/1-crop per year based on the LU map
            self._calc_profit()
            
            if self.p:
                print('run Metronamica')            
            #50: Run LUCM            
            gnm.run_model(geoproj=self.geoproj,
                          log_option=self.log_option,
                          run_option='Step',
                          save=True,
                          model_folder=self.model_folder)
                          
            #70: Calculate yield per district
            _yield_district = self._calc_yield_per_district()
            _profit_district, _profit_district_not_normalized = self._calc_profit_per_district()
            
            if self.current_yr >= 2025 :
                for key, val in _yield_district.items():
                    try:
                        self.yield_district_end[key].append(val)
                        self.profit_district_end[key].append(_profit_district[key])
                        self.profit_district_end_not_normalized[key].append(_profit_district_not_normalized[key])
                    except:
                        self.yield_district_end[key] = [val]
                        self.profit_district_end[key] = [_profit_district[key]]
                        self.profit_district_end_not_normalized[key] = [_profit_district_not_normalized[key]]
                        
            #61: Update LU map
            try:
                if self.p:
                    print('update LU map')
                self.lu_map = pr.get_landuse_array(self.model_folder+'//Log_cmd//Land_use//Land use map_{}-Jan-01 00_00_00.asc'.format(self.current_yr))
            except:
                if self.p:
                    print('it is year {} already'.format(self.current_yr))
                        

                        
            #99: Tick
            self.count_yr += 1
            self.current_yr += 1
        
        #99: reset geonamica model
        gnm.run_model(geoproj=self.geoproj,
                      log_option=self.log_option,
                      run_option='Reset',
                      save=True,
                      model_folder=self.model_folder)
                      
        #for hackathon back then
        for key, val in self.profit_district_end.items():
            self.average_profit_district[key] = np.mean(val)
        for key, val in self.profit_district_end_not_normalized.items():
            self.average_profit_district_not_normalized[key] = np.mean(val)
            
        self.average_profit_all = np.mean(list(self.average_profit_all.values()))
            
        cumulative_depths = {}
        for key, val in self.annual_flood_map.items():
            cumulative_depths[key] = val.sum().sum()
        max_flood_yr = max(cumulative_depths.items(), key=operator.itemgetter(1))[0]
        self.worst_flood = self.annual_flood_map[max_flood_yr]
        
    def run_QUEFTS_(self, n):
        
        nutrients_map = pr.get_landuse_array(self.model_folder+'//Data//nutrients_stock//nutrients_stock_{}.tif'.format(self.current_yr))
        if n==0:
            nutrients_map = np.full(shape=nutrients_map.shape, fill_value=self.init_nutrient)
        nutrients_map = nutrients_map/3 #nutrients stock assume annual stock, not just seasonal stock
        # fertilizer map is already in application of fertilizer per season (not for the entire year)
        # the median fertilizer application is 625kg/ha (10.1016/j.jenvman.2018.03.116)
        self.fertilizer_supply = self.fert_map * np.random.triangular(0.72, 0.94, 1.1)
        if self.pol_fer:
            if self.current_yr >= 2025:
                fertilizer_supply_current_yr = np.where(((self.river_acc_map < 0.15) & (self.fertilizer_supply > 0)), self.fertilizer_supply+50, self.fertilizer_supply)
            else:
                fertilizer_supply_current_yr = self.fertilizer_supply
        if not self.pol_fer:
            fertilizer_supply_current_yr = self.fertilizer_supply
        #assuming the nutrients_map is the availability of sediment in the soil
        #the ratio of N,P,K supply is based on https://www.hydrol-earth-syst-sci.net/18/3033/2014/
        In_soil = nutrients_map * np.random.triangular(0.039, 0.049, 0.059, size=nutrients_map.shape)
        Ip_soil = nutrients_map * np.random.triangular(0.015, 0.019, 0.023, size=nutrients_map.shape)
        Ik_soil = nutrients_map * np.random.triangular(0.18, 0.225, 0.27, size=nutrients_map.shape)
        
        #percentage from supplementary material 1 of 10.1016/j.jenvman.2018.03.116
        self.In_perc = np.random.uniform(0.16, 0.2, size=self.fertilizer_supply.shape)
        self.Ip_perc = np.random.uniform(0.16, 0.2, size=self.fertilizer_supply.shape)
        self.Ik_perc = np.random.uniform(0.13, 0.15, size=self.fertilizer_supply.shape)
        
        In_fert = fertilizer_supply_current_yr * self.In_perc
        Ip_fert = fertilizer_supply_current_yr * self.Ip_perc
        Ik_fert = fertilizer_supply_current_yr * self.Ik_perc
        
        #input of N,P,K = input fromm soil + input from fertilizer
        In = In_soil + In_fert
        Ip = Ip_soil + Ip_fert
        Ik = Ik_soil + Ik_fert
        
        YU, UptakeN, UptakeP, UptakeK = self._QUEFTS_metamodel(In, Ip, Ik)
        
        In_soil = In_soil - np.maximum(UptakeN-In_fert,0)
        Ip_soil = Ip_soil - np.maximum(UptakeP-Ip_fert,0)
        Ik_soil = Ik_soil - np.maximum(UptakeK-Ik_fert,0)
        
        stock_N = In_soil / 0.049
        stock_P = Ip_soil / 0.019
        stock_K = Ik_soil / 0.225
        
        nutrients_map = np.minimum(np.minimum(stock_N, stock_P), stock_K)
        nutrients_map = np.maximum(nutrients_map, 0)
        
        #regenerate nutrient unless it is urban land-use (urban lu_code=5)
        nutrients_map = np.where(self.lu_map==5, nutrients_map, 
                                 nutrients_map*np.random.triangular(1.05, 1.1, 1.15, size=nutrients_map.shape))
        
        #normalize back to annual nutrients stock from seasonal nutrients stock
        nutrients_map *= 3
        
        projection_template = self.model_folder+'//Data//DTAG_NIAESLU2002_200m_v03a_wgs84utm48n_t05.tif'
        outfile_name = self.model_folder+'//Data//nutrients_stock//nutrients_stock_{}.tif'.format(self.current_yr+1)
        pr.ndarray_to_tiff(nutrients_map, outfile_name, projection_template)
        
        #adjust YU. yield only applies to shrimp-rice, triple rice, double rice, and single rice
        #yield is not normalized, as we are interested in annual potential yield instead of actual yield
        #this is gonna be used later for rice farmers welfare calculation
        YU = np.where(np.isin(self.lu_map, [1,2,3,4]), YU, 0)
        
        return YU
    
    def _QUEFTS_metamodel(self, In, Ip, Ik, calc_uptake=True):
        '''
        based on Witt et al (1999): 
            internal nutrient efficiencies of irrigated lowland rice in tropical and subtropical Asia
            See table 8
        '''
        y = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 7500, 8000, 8500, 9000, 9500, 9800, 9900, 9990]
        x_N = [0, 15, 29, 44, 59, 73, 88, 104, 115, 127, 142, 159, 182, 205, 217, 243]
        x_P = [0, 2.6, 5.2, 7.8, 10.4, 13, 15.6, 18.4, 20.4, 22.6, 25.1, 28.2, 32.2, 36.3, 38.6, 43.1]
        x_K = [0, 15, 29, 43, 58, 72, 87, 103, 114, 126, 140, 157, 180, 203, 215, 240]
        
        Yield_N = np.interp(In, x_N, y)
        Yield_P = np.interp(Ip, x_P, y)
        Yield_K = np.interp(Ik, x_K, y)

        Yield = np.minimum(np.minimum(Yield_N, Yield_P), Yield_K)

        if calc_uptake:
            UptakeN = np.interp(Yield, y, x_N)
            UptakeP = np.interp(Yield, y, x_P)
            UptakeK = np.interp(Yield, y, x_K)
            
            return Yield, UptakeN, UptakeP, UptakeK
        
        else:
            return Yield
        
    
    def _adjust_yield(self, rice_yield, flood_map_all, flood_map_jul):
        '''
        based on:
        Towards risk-based flood management in highly productive paddy rice cultivation in Mekong Delta (2018)
        
        Damages due to flooding are incurred to double and triple rice cropping, but not to single rice cropping.
        WSC: Winter Spring Crop (not affected by flood)
        SAC: Summer Autumn Crop (double+triple rice; affected by July flood)
        AWC: Autumn Winter Crop (triple rice, affected by annual peak flood)
        '''
        
        #reduction of yield for shallow flooding, use s-shape function instead of just a number
        y = [0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.75, 0.85, 0.95, 1] #reduction
        x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] #flood depth
        if self.seed_upgrade:
            y = [0, 0.021, 0.042, 0.07, 0.105, 0.14, 0.21, 0.525, 0.65, 0.8, 1]
        
        reduction_jul = np.interp(flood_map_jul, x, y) #yield reduction in july flood
        reduction_all = np.interp(flood_map_all, x, y) #yield reduction in maximum annual flood
        
        if self.current_yr >= 2015:
            self.annual_rice_production_0[self.current_yr] = rice_yield.sum().sum()
        
        rice_yield_single = np.where(self.lu_map==4, rice_yield, 0)
        rice_yield_double = np.where(np.isin(self.lu_map, [1,3]), rice_yield, 0)
        rice_yield_triple = np.where(self.lu_map==2, rice_yield, 0)
        
        #double rice Summer-autumn crop (SAC) is affected only by july flood
        #double rice Winter-spring crop (WSC) is not affected by flood
        rice_yield_double_WSC = rice_yield_double
        rice_yield_double_SAC = rice_yield_double
        reduced_yield_double_SAC = rice_yield_double * (reduction_jul) * (1-self.yield_adjust_SA) #for validation
        rice_yield_double_SAC = rice_yield_double_SAC * (1-reduction_jul) * (1-self.yield_adjust_SA)
        rice_yield_double_SAC_forValidation = rice_yield_double_SAC * (1-self.yield_adjust_SA)
        rice_yield_double = rice_yield_double_WSC + rice_yield_double_SAC
        
        #triple rice is affected by both july and max annual peak flood
        rice_yield_triple_WSC = rice_yield_triple
        rice_yield_triple_SAC = rice_yield_triple
        rice_yield_triple_AWC = rice_yield_triple
        reduced_yield_triple_SAC = rice_yield_triple * (reduction_jul) * (1-self.yield_adjust_SA) #for validation
        reduced_yield_triple_AWC = rice_yield_triple * (reduction_all) * (1-self.yield_adjust_AW) #for validation
        rice_yield_triple_SAC = rice_yield_triple_SAC * (1-reduction_jul) * (1-self.yield_adjust_SA)
        rice_yield_triple_SAC_forValidation = rice_yield_triple_SAC * (1-self.yield_adjust_SA)
        rice_yield_triple_AWC = rice_yield_triple_AWC * (1-reduction_all) * (1-self.yield_adjust_AW)
        rice_yield_triple_AWC_forValidation = rice_yield_triple_AWC * (1-self.yield_adjust_AW)
        rice_yield_triple = rice_yield_triple_WSC + rice_yield_triple_SAC + rice_yield_triple_AWC
        
        #combine everything again
        rice_yield = rice_yield_single + rice_yield_double + rice_yield_triple
        
        #update validation variables - yield
        rice_yield_avrg = rice_yield_double/2 + rice_yield_triple/3
        v_yield_avrg = np.where(np.isin(self.lu_map, [2,3]), rice_yield_avrg, 0)
        v_yield_angiang = np.where(self.province_map==1, rice_yield_avrg, 0)
        v_yield_dongthap = np.where(self.province_map==2, rice_yield_avrg, 0)
        
        self.v_yield_map[self.current_yr] = v_yield_avrg[v_yield_avrg>0]
        
        v_yield_avrg = np.mean(v_yield_avrg[v_yield_avrg>0])
        v_yield_angiang = np.mean(v_yield_angiang[v_yield_angiang>0])
        v_yield_dongthap = np.mean(v_yield_dongthap[v_yield_dongthap>0])
        
        v_yield_double = np.where(self.lu_map==3, rice_yield_double/2, 0)
        v_yield_double = np.mean(v_yield_double[v_yield_double>0])
        v_yield_triple = np.where(self.lu_map==2, rice_yield_triple/3, 0)
        v_yield_triple = np.mean(v_yield_triple[v_yield_triple>0])
        
        self.v_yield_avrg[self.current_yr] = v_yield_avrg
        self.v_yield_ag[self.current_yr] = v_yield_angiang
        self.v_yield_dt[self.current_yr] = v_yield_dongthap
        self.v_yield_double[self.current_yr] = v_yield_double
        self.v_yield_triple[self.current_yr] = v_yield_triple
        
        v_yield_WSC = rice_yield_triple_WSC + rice_yield_double_WSC
        v_yield_WSC = np.where(np.isin(self.lu_map, [2,3]), v_yield_WSC, 0)
        self.v_yield_WSC[self.current_yr] = v_yield_WSC[v_yield_WSC>0]
        v_yield_SAC = rice_yield_triple_SAC_forValidation + rice_yield_double_SAC_forValidation
        v_yield_SAC = np.where(np.isin(self.lu_map, [2,3]), v_yield_SAC , 0) 
        self.v_yield_SAC[self.current_yr] = v_yield_SAC[v_yield_SAC>0]
        v_yield_AWC = np.where(np.isin(self.lu_map, [2]), rice_yield_triple_AWC_forValidation, 0)
        v_yield_AWC = v_yield_AWC[v_yield_AWC>0]
        self.v_yield_AWC[self.current_yr] = v_yield_AWC
        
        self.v_loses[self.current_yr] = reduced_yield_double_SAC + reduced_yield_triple_SAC + reduced_yield_triple_AWC
        self.v_loses[self.current_yr] = self.v_loses[self.current_yr][self.v_loses[self.current_yr]>0]
        
        return rice_yield
        
    def _record_yield(self):
        
        sum_yield = self.yield_map.sum().sum()
        
        if self.current_yr >= 2025 :
            self.annual_rice_production[self.current_yr] = sum_yield
            self.total_rice_production += sum_yield
    
    def _subsidence(self):
        '''
        based on:
        The relation between land use and subsidence in the Vietnamese Mekong delta (2018)
        '''
        nrow, ncol = self.lu_map.shape[0], self.lu_map.shape[1]
        
        #generate subsidance maps for each land-use class
        other = np.random.normal(loc=0.006, scale=0.006*0.2, size=(nrow, ncol))
        other = np.maximum(other, 0)
        other = np.minimum(other, 0.018)
        
        triple_rice = np.random.normal(loc=0.008, scale=0.008*0.2, size=(nrow, ncol))
        triple_rice = np.maximum(triple_rice, 0)
        triple_rice = np.minimum(triple_rice, 0.021)
        
        double_rice = np.random.normal(loc=0.0105, scale=0.0105*0.2, size=(nrow, ncol))
        double_rice = np.maximum(double_rice, 0)
        double_rice = np.minimum(double_rice, 0.025)
        
        single_rice = np.random.normal(loc=0.013, scale=0.013*0.2, size=(nrow, ncol))
        single_rice = np.maximum(single_rice, 0)
        single_rice = np.minimum(single_rice, 0.03)
        
        urban = np.random.normal(loc=0.019, scale=0.019*0.2, size=(nrow, ncol))
        urban = np.maximum(urban, 0.005)
        urban = np.minimum(urban, 0.03)
        
        orchard = np.random.normal(loc=0.014, scale=0.014*0.2, size=(nrow, ncol))
        orchard = np.maximum(orchard, 0.002)
        orchard = np.minimum(orchard, 0.025)
        
        aquaculture = np.random.normal(loc=0.011, scale=0.011*0.2, size=(nrow, ncol))
        aquaculture = np.maximum(aquaculture, 0)
        aquaculture = np.minimum(aquaculture, 0.03)
        
        #reduce DEM based on land-use class
        self.subsidence_map = np.where(np.isin(self.lu_map, [0,8,9]) , self.subsidence_map + other, self.subsidence_map)
        self.subsidence_map = np.where(self.lu_map==2, self.subsidence_map + triple_rice, self.subsidence_map )
        self.subsidence_map = np.where(np.isin(self.lu_map, [1,3]), self.subsidence_map + double_rice, self.subsidence_map )
        self.subsidence_map = np.where(self.lu_map==4, self.subsidence_map + single_rice, self.subsidence_map )
        self.subsidence_map = np.where(self.lu_map==5, self.subsidence_map + urban, self.subsidence_map )
        self.subsidence_map = np.where(self.lu_map==6, self.subsidence_map + orchard, self.subsidence_map )
        self.subsidence_map = np.where(self.lu_map==7, self.subsidence_map + aquaculture, self.subsidence_map )
        
    def _deposit_sediment3(self, flood_map):
    
        #get water level at Tan Chau
        #visual inspection says that Tan Chau is at [44][210]
        H = flood_map[44][210]
        H = flood_map[94][254]
        H = max(flood_map[44][210], flood_map[94][254])
        H = np.percentile(flood_map[44:54, 210:254], 99.4)
        
        #calculate total sedimentation based on Manh et al (2015), in million tons
        #all statistical metamodel is from Table 3 in the paper above
        if self.TE==0: #no dam construction
            total_sedimentation = 1.99 * (H)**2 - 12.61 * H + 20.44
        elif self.TE==1: #trapping efficiency = 0.12
            total_sedimentation = 1.86 * (H)**2 - 12.12 * H + 20.54
        elif self.TE==2: #trapping efficiency = 0.33
            total_sedimentation = 1.43 * (H)**2 - 9.41 * H + 16.12
        elif self.TE==3: #trapping efficiency = 0.53
            total_sedimentation = 1.06 * (H)**2 - 7.13 * H + 12.53
        elif self.TE==4: #trapping efficiency = 0.74
            total_sedimentation = 0.61 * (H)**2 - 4.18 * H + 7.47
        else: #trapping efficiency = 0.95
            total_sedimentation = 0.12 * (H)**2 - 0.84 * H + 1.51
        
        #update validation variables 
        self.v_sediment_supply[self.current_yr] = total_sedimentation        
        
        if self.current_yr >= 2015:
            self.annual_potential_sedimentation[self.current_yr] = total_sedimentation
        
        q_now = self.q_kratie.loc[self.q_kratie['yr']==self.current_yr]['q'].iloc[0]
        
        if self.v_sedimentation_reduction:
            total_sedimentation = total_sedimentation * (1-self.v_sedimentation_reduction)
        
        #print(self.current_yr, total_sedimentation)
        
        #from a raster analysis, our case study area, DTAG, retains 90.67% of the total sedimentation in Manh et al (2015)
        total_sedimentation *= 0.9067
        #convert from million ton to kg
        total_sedimentation *= 1e9
        
        nutrients_map = pr.get_landuse_array(self.model_folder+'//Data//nutrients_stock//nutrients_stock_{}.tif'.format(self.current_yr+1))
    
        #reduce sedimentation rate in shallow flood
        adjusted_sedimentation_map = np.where(flood_map > self.shallow_thres, self.sediment_rate_map, self.sediment_rate_map*self.shallow_flood_sediment)
        #when there's no flood then there's no sedimentation
        adjusted_sedimentation_map = np.where(flood_map > 0, adjusted_sedimentation_map, 0)
        
        #normalize sedimentation rate map to 0-1
        adjusted_sedimentation_map = adjusted_sedimentation_map / adjusted_sedimentation_map.sum().sum()
        
        #normalize total sedimentation
        deposited_sediment = total_sedimentation * adjusted_sedimentation_map #in kg/4ha
        self.v_sedimentation_rate[self.current_yr] = deposited_sediment
        
        deposited_nutrient = deposited_sediment * self.nutrient_in_sediment
        
        nutrients_map = nutrients_map + deposited_nutrient
        #avoid cumulating nutrients in non-agriculture area
        nutrients_map[nutrients_map>1e6]=1e6
        
        projection_template = self.model_folder+'//Data//DTAG_NIAESLU2002_200m_v03a_wgs84utm48n_t05.tif'
        outfile_name = self.model_folder+'//Data//nutrients_stock//nutrients_stock_{}.tif'.format(self.current_yr+1)
        pr.ndarray_to_tiff(nutrients_map, outfile_name, projection_template)
        
    def _calc_profit(self):
        '''
        fertilizer cost and rice selling price based on Table 3 Tran et al (2018)
            Questioning triple rice intensification............
        '''
        income = self.yield_map * np.random.triangular(4.7, 5, 5.5, size=self.yield_map.shape) #5000 VND / ton = 5 VND / kg
        
        fertilizer_fixed_cost = np.where(self.lu_map==4, self.fertilizer_supply * 10, 0) #single rice
        fertilizer_fixed_cost = np.where(np.isin(self.lu_map, [1,3]), self.fertilizer_supply * 10 * 2, fertilizer_fixed_cost) #double rice
        fertilizer_fixed_cost = np.where(self.lu_map==2, self.fertilizer_supply * 10 * 3, fertilizer_fixed_cost) #triple rice
        
        fertilizer_total_cost = fertilizer_fixed_cost
        
        total_cost = fertilizer_total_cost * self.cost_scaling_factor
        
        self.profit_map = income - total_cost

        profit_all = self.profit_map[self.profit_map>0]
        profit_all = np.mean(profit_all)
        self.average_profit_all[self.current_yr] = profit_all
        
        #update validation variables - revenue
        v_revenue_avrg = np.where(np.isin(self.lu_map, [2,3]), income, 0)
        v_revenue_avrg = np.mean(v_revenue_avrg[v_revenue_avrg>0])
        
        v_revenue_double = np.where(self.lu_map==3, income, 0)
        v_revenue_double = np.mean(v_revenue_double[v_revenue_double>0])
        
        v_revenue_triple = np.where(self.lu_map==2, income, 0)
        v_revenue_triple = np.mean(v_revenue_triple[v_revenue_triple>0])
        
        self.v_revenue_avrg[self.current_yr] = v_revenue_avrg
        self.v_revenue_double[self.current_yr] = v_revenue_double
        self.v_revenue_triple[self.current_yr] = v_revenue_triple
        
        #update validation variables - profit
        v_profit_avrg = np.where(np.isin(self.lu_map, [2,3]), self.profit_map, 0)
        v_profit_avrg = v_profit_avrg[v_profit_avrg>0]
        
        v_profit_double = np.where(self.lu_map==3, self.profit_map, 0)
        v_profit_double = v_profit_double[v_profit_double>0]
        
        v_profit_triple = np.where(self.lu_map==2, self.profit_map, 0)
        v_profit_triple = v_profit_triple[v_profit_triple>0]
        
        self.v_profit_avrg[self.current_yr] = v_profit_avrg
        self.v_profit_double[self.current_yr] = v_profit_double
        self.v_profit_triple[self.current_yr] = v_profit_triple
    
    def _calc_yield_per_district(self):
        yield_district = {}
        
        for d in np.unique(self.districts_map):
            if d >0:
                yield_map_district = np.where(self.districts_map==d, self.yield_map, 0)
                yield_map_district = yield_map_district[yield_map_district>0]
                
                y = np.median(yield_map_district)
                
                yield_district[d] = y
        
        return yield_district
    
    def _calc_profit_per_district(self):
        profit_district = {}
        profit_district_not_normalized = {}
        
        for d in np.unique(self.districts_map):
            if d >0:
                profit_map_district = np.where(self.districts_map==d, self.profit_map, 0)
                profit_map_district = np.where(self.lu_map==2, profit_map_district/3, profit_map_district) #triple rice
                profit_map_district = np.where(np.isin(self.lu_map, [1,3]), profit_map_district/2, profit_map_district) #double rice
                profit_map_district = np.where(self.lu_map>4, 0, profit_map_district) #1,2,3,4 are the only rice land-use classes
                profit_map_district = profit_map_district[profit_map_district>0]
                
                y = np.median(profit_map_district)
                
                profit_district[d] = y
                
                #not normalized
                profit_map_district = np.where(self.districts_map==d, self.profit_map, 0)
                profit_map_district = np.where(np.isin(self.lu_map, [1,2,3,4]), profit_map_district, 0)
                profit_map_district = profit_map_district[profit_map_district>0]
                y = np.median(profit_map_district)
                profit_district_not_normalized[d] = y
        
        return profit_district, profit_district_not_normalized
    
    def _climate_scenario(self):
        
        rcps = {1: 'rcp45', 2: 'rcp85', 'val': 'Data/q_kratie/q_kratie_historical.csv'}
        
        rcp = rcps[self.clim_scen]
        fn = rcp+'_med_v01.csv'
        
        q_hydrograph = 'Data//q_kratie//'+fn
        
        if self.clim_scen == 'val':
            self.q_kratie = pd.read_csv(rcps[self.clim_scen])
        else:
            self.q_kratie = pd.read_csv(q_hydrograph)
            
        #adjust discharge based on dam trapping scenarios
        #based on (10.5194/hess-16-4603-2012), average decrease in annual peak discharge
        #when all dams are built is 12.3%. This becomes the upper bound (TE==5).
        #based on (10.1016/j.gloplacha.2015.01.001), the lower bound (TE==1) is 40% of the upper bound, meaning 4.94%.
        #the rest of TE scenarios is adjusted accordingly, 
        #based on linear transformation of upper and lower bound
        if self.TE==1: #trapping efficiency = 0.12
            self.q_kratie['q'] *= (1-0.049)
        elif self.TE==2: #trapping efficiency = 0.33
            self.q_kratie['q'] *= (1-0.068)
        elif self.TE==3: #trapping efficiency = 0.53
            self.q_kratie['q'] *= (1-0.086)
        elif self.TE==4: #trapping efficiency = 0.74
            self.q_kratie['q'] *= (1-0.105)
        elif self.TE==5: #trapping efficiency = 0.95
            self.q_kratie['q'] *= (1-0.123)
            
        if self.v_discharge_reduction:
            self.q_kratie['q'] *= (1-self.v_discharge_reduction)
        
        
    def _landuse_scenario(self):
        #land-use demand scenario. 1: BAU, 2: shift back to 2-rice, 3: heavy urbanization, 4: rising non-rice
        
        dmd_shrimp = 1500 #np.random.uniform(1200, 1300)
        dmd_triple = 77000 #np.random.uniform(73000, 75000)
        dmd_double = 48000 #np.random.uniform(48000, 52000)
        dmd_single = 200 #np.random.uniform(900, 1100)
        dmd_urban = 3500 #np.random.uniform(3700, 4300)
        dmd_orchard = 26000 #np.random.uniform(28000, 32000)
        dmd_aqua = 2000 #np.random.uniform(1200, 1400)
            
        if self.lud_scen == 2:
            dmd_shrimp = 4500 #np.random.uniform(1800, 2200)
            dmd_triple = 45000 #np.random.uniform(58000, 62000)
            dmd_double = 75000 #np.random.uniform(73000, 77000)
            dmd_aqua = 6000
            
        elif self.lud_scen == 3:
            dmd_triple = np.random.uniform(67000, 69000)
            dmd_double = np.random.uniform(44000, 46000)
            dmd_urban = np.random.uniform(11000, 13000)
            
        elif self.lud_scen == 4:
            dmd_shrimp = np.random.uniform(6500, 7000)
            dmd_triple = np.random.uniform(65000, 67000)
            dmd_double = np.random.uniform(40000, 42000)
            dmd_urban = np.random.uniform(4700, 5300)
            dmd_orchard = np.random.uniform(30000, 32000)
            dmd_aqua = np.random.uniform(6000, 6500)
            
        new_demand = [dmd_shrimp, dmd_triple, dmd_double, dmd_single, dmd_urban, dmd_orchard, dmd_aqua]
        new_demand = [int(i) for i in new_demand]
        
        mh.edit_demand(geoproj_infile=self.model_folder+'//'+self.geoproj,
                       geoproj_outfile=self.model_folder+'//'+self.geoproj,
                       new_demands=new_demand)
        
    def _generate_flood_map(self):
        if self.current_yr < 2012:
            dike_yr = self.current_yr
        else:
            dike_yr = 2011 #latest dike map we have is from 2012, but we decided to use 2011 map as it looks more reliable
            
        dike_map = pr.get_landuse_array(self.model_folder+'//Data//CombinedDikes_Proxy//DTAG_Dikes_proxy_{}.tif'.format(dike_yr))
        
        if self.pol_dike_switch == 1: #all high dikes
            if self.current_yr>=2025:
                lu_2011 = pr.get_landuse_array(self.model_folder+'//Data//DTAG_NIAESLU2011_200m_v03a_wgs84utm48n_t05.tif')
                dike_map = np.where(np.isin(lu_2011, [1,2,3,4]), 2, dike_map)
                
        elif self.pol_dike_switch == 2: #all low dikes
            if self.current_yr>=2025:
                lu_2011 = pr.get_landuse_array(self.model_folder+'//Data//DTAG_NIAESLU2011_200m_v03a_wgs84utm48n_t05.tif')
                dike_map = np.where(np.isin(lu_2011, [1,2,3,4]), 1, dike_map)
        
        #in case dike policies are activated, save the new modified dike map to drive
        #so that it's used by the land-use change module in Metronamica
        if self.current_yr == 2025:
            projection_template = self.model_folder+'//Data//DTAG_NIAESLU2002_200m_v03a_wgs84utm48n_t05.tif'
            outfile_name = self.model_folder+'//Data//CombinedDikes_Proxy//DTAG_Dikes_proxy_2025.tif'
            pr.ndarray_to_tiff(dike_map, outfile_name, projection_template)
        
        q_now = self.q_kratie.loc[self.q_kratie['yr']==self.current_yr]['q'].iloc[0]
        q_now = float(q_now)
        if self.coop_discharge:
            if q_now > self.coop_discharge:
                q_now = self.coop_discharge
        #print('Q_Kratie = ' + str(q_now))
        
        flood_map_all = self.flood_intercept_map_all + (self.flood_coeff_map_all * q_now)
        flood_map_jul = self.flood_intercept_map_jul + (self.flood_coeff_map_jul * q_now)
        
        #the flood simulation model has taken into account dike 2.5m
        #therefore we do not consider dike 2.5m anymore here        
        flood_map_all = np.where(((dike_map==2) & (flood_map_all + self.subsidence_map < 4.5)), 0, flood_map_all)
        flood_map_all = np.where(flood_map_all>0, flood_map_all + self.subsidence_map, flood_map_all)        
        flood_map_all = np.maximum(flood_map_all,0)
        
        flood_map_jul = np.where(((dike_map==2) & (flood_map_jul + self.subsidence_map < 4.5)), 0, flood_map_jul)
        flood_map_jul = np.where(flood_map_jul>0, flood_map_jul + self.subsidence_map, flood_map_jul)        
        flood_map_jul = np.maximum(flood_map_jul,0)
        
        projection_template = self.model_folder+'//Data//DTAG_NIAESLU2002_200m_v03a_wgs84utm48n_t05.tif'
        outfile_name = self.model_folder+'//Data//water//water_recurence_{}.tif'.format(self.current_yr)
        pr.ndarray_to_tiff(flood_map_all, outfile_name, projection_template)
        
        self.flood_map_check = flood_map_all
        
        #Make sure flood depth always >= 0
        flood_map_all[flood_map_all<0] = 0
        flood_map_jul[flood_map_jul<0] = 0
        
        #update validation variables
        #for calculation of the index position of Tan  Chau and Chau Doc hydro station
        self.v_max_wl_tanchau[self.current_yr] = flood_map_all[94, 259] #this index is the position of the Tan Chau hydro station
        #alternative position for Tan Chau is [44,210], as based on the function _deposit_sediment3
        self.v_max_wl_chaudoc[self.current_yr] = flood_map_all[152, 206] #this index is the position of the Chau Doc hydro station
        
        return flood_map_all, flood_map_jul
    
def DTAG_coupledModel_EMA_mp_v02(clim_scen=1, 
                          lud_scen=1,
                          TE=0,
                          pol_fer=0,
                          pol_dike_switch=0,
                          pol_discharge=0,
                          pol_seed=0,
                          model_folder='temporary'):
                          
    if pol_fer==0:
        fert_map = 'Data//DTAG_fertilizer.tif'
    elif pol_fer==1:
        fert_map =  'Data/DTAG_fertilizer_pol1.tif'
    elif pol_fer==2:
        fert_map = 'Data/DTAG_fertilizer_pol2.tif'
        
    if pol_discharge==0:
        coop_discharge = None
    elif pol_discharge==1:
        coop_discharge = 50000
        
    if pol_seed==0:
        seed_upgrade = False
    elif pol_seed==1:
        seed_upgrade = True
    
    if model_folder != os.getcwd():
        #make temporary directory
        tempfolder = tempfile.TemporaryDirectory()
        model_folder = tempfolder.name + "/metronamica_model"
        src = os.getcwd()
        dst = model_folder
        shutil.copytree(src, dst)

    #model run
    print('initiating model')
    CoupledModel = MetronamicaCoupled(start_yr = 2002, 
                                      end_yr = 2050,
                                      lu_map = model_folder+'//Data//DTAG_NIAESLU2002_200m_v03a_wgs84utm48n_t05.tif',
                                      geoproj = 'model_v03a_baj_process_v02.geoproj',
                                      log_option = 'cmd_log.xml',
                                      fert_map = model_folder+'//'+fert_map,
                                      soil_map = model_folder+'//Data//soil_DTAG_HWSD_full_recategorized.tif',
                                      districts_map = model_folder+'//Data//DTAG_districts_recoded.tif',
                                      river_acc_map = model_folder+'//Data//river_accessibility.tif',
                                      flood_coeff_map_all='Data//floodsim//all_s1_coeffs_v01.tif', 
                                      flood_intercept_map_all='Data//floodsim//all_s1_intercepts_v01.tif',
                                      flood_coeff_map_jul='Data//floodsim//jul_s1_coeffs_v01.tif', 
                                      flood_intercept_map_jul='Data//floodsim//jul_s1_intercepts_v01.tif',
                                      sediment_rate_map='Data//fig7_interpolation_distance1.tif',
                                      clim_scen = clim_scen,
                                      lud_scen = lud_scen,
                                      model_folder = model_folder,
                                      coop_discharge=coop_discharge,
                                      pol_dike_switch=pol_dike_switch,
                                      seed_upgrade=seed_upgrade,
                                      TE=TE)

    print('starting simulation')
    CoupledModel.run()
    
    #save results
    result = {}
    for key, val in CoupledModel.profit_district_end.items():
        result['district_{}'.format(int(key))] = np.mean(val)
    result['total_rice_yield'] = CoupledModel.total_rice_production
        
    if model_folder != os.getcwd():
        print('cleanup starting')
        tempfolder.cleanup()
        
    return result