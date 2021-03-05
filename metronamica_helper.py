from lxml import etree
import pandas as pd
import copy
import numpy as np

def _demand_helper(root):
    #read simulation starttime and endtime
    modelBlocks = root.xpath("/GeonamicaSimulation/model/modelSettings/startTime")
    simulation_starttime = modelBlocks[0].text
    modelBlocks = root.xpath("/GeonamicaSimulation/applicationSettings/SimulationPauses//elem")
    simulation_endtime = modelBlocks[0].text
    
    #check if demand is available at simulation endtime
    all_demands_time = []
    modelBlocks = root.xpath("//modelBlock[@name='MB_Land_use_demand']//value")
    for node in modelBlocks:
        all_demands_time.append(node.attrib.values()[0])
    if simulation_endtime in all_demands_time: #if endtime demand is available, use it
        demand_time = simulation_endtime
    else: #otherwise, use starttime
        demand_time = simulation_starttime
        
    return demand_time


def read_demand(geoproj_infile):
    '''
    version 1.0
    - Read demand at end of simulation time
    - If demand has not yet been specified after the Metronamica model is setup, use demand at t=0
    '''

    root = etree.parse(geoproj_infile)
    
    #check if demand is available at simulation endtime
    demand_time = _demand_helper(root)
    
    modelBlocks = root.xpath("//modelBlock[@name='MB_Land_use_demand']//value[@time='{}']//elem".format(demand_time))
    
    demands = []
    for node in modelBlocks:
        demands.append(node.text)
        
    return demands
    
def edit_demand(geoproj_infile, geoproj_outfile, new_demands):
    '''
    version 1.0
    - Modify demand at end of simulation time
    - If demand has not yet been specified after the Metronamica model is setup, create demand at endtime
    '''

    root = etree.parse(geoproj_infile)
    
    #check if demand is available at simulation endtime
    demand_time = _demand_helper(root)
    
    #read actual simulation endtime
    modelBlocks = root.xpath("/GeonamicaSimulation/applicationSettings/SimulationPauses//elem")
    simulation_endtime = modelBlocks[0].text
    
    #if demand for simulation endtime is already available, modify it
    if demand_time == simulation_endtime:
        modelBlocks = root.xpath("//modelBlock[@name='MB_Land_use_demand']//value[@time='{}']//elem".format(demand_time))
        for i, node in enumerate(modelBlocks):
            node.text = str(new_demands[i])
    #else, make new demand at simulation endtime
    else:
        modelBlocks = root.find("//modelBlock[@name='MB_Land_use_demand']/DemandBlock/LanduseDemands")
        
        for c in root.findall("//modelBlock[@name='MB_Land_use_demand']/DemandBlock/LanduseDemands/value"):
            dupe = copy.deepcopy(c)
            modelBlocks.append(dupe)
        
        modelBlocks = root.xpath("//modelBlock[@name='MB_Land_use_demand']/DemandBlock/LanduseDemands/value")
        print(simulation_endtime)
        for i, node in enumerate(modelBlocks):
            
            if i==1:
                node.attrib['time']=simulation_endtime
            print(node.attrib)
                
        modelBlocks = root.xpath("//modelBlock[@name='MB_Land_use_demand']//value[@time='{}']//elem".format(simulation_endtime))
        for i, node in enumerate(modelBlocks):
            node.text = str(new_demands[i])
                
        
    #save file
    f = open(geoproj_outfile, 'w')
    f.write(etree.tostring(root, pretty_print=False, encoding='unicode'))
    f.close()