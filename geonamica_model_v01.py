from lxml import etree
import pandas as pd
import numpy as np
import os
import sys
from subprocess import call
from lxml import etree
import inspect

import shutil
import glob
from backports import tempfile

from tempfile import mkstemp
from os import fdopen, remove

# sys.path.insert(0, "C:\\Program Files (x86)\\Geonamica\\Metronamica")

import metronamica_helper as mh #from the helper folder

def _generate_options(log_option=None, 
                      run_option='Run', 
                      step_nr = 1,
                      save=None,
                      model_folder = os.getcwd()):
    if run_option == 'Step':
        run_option = ['--Step', str(step_nr)]
    elif run_option == 'Reset':
        run_option = ['--Reset']
    else:
        run_option = ['--Run']
    
    if save:
        save_option = ['--Save']
    else:
        save_option = []
        
    if log_option:
        log_option = model_folder+'\\'+log_option
        
        #change the directory where land-use maps are logged
        with open(log_option, 'rb') as xml_file:
            root = etree.parse(xml_file)
        modelBlocks = root.xpath("//Path")
        for node in modelBlocks:
            #node.text='Log_cmd'
            node.text=model_folder+'\\Log_cmd'
        f = open(log_option, 'w')
        f.write(etree.tostring(root, pretty_print=False, encoding='unicode'))
        f.close()
        
        del root
        del f
        
        log_option = [ '--LogSettings', log_option]
    else:
        log_option = []
        
    return run_option, save_option, log_option

def run_model(geoproj, 
              log_option=None, 
              run_option='Run',
              step_nr = 1,
              save=None,
              model_folder = os.getcwd()):
    
    #use this dict to browse through the code
    all_phases: {'phase01': 'arg preparation phase', 
                 'phase99': 'simulation run'} 
    
    
    '''
    phase01: exe's arg preparation phase
    '''
    run_option, save_option, log_option = _generate_options(log_option, run_option, step_nr, save, model_folder)
        
    geoproj = model_folder+'\\'+geoproj
    #geoproj = model_folder+r'/'+geoproj
    
    '''
    phase99: simulation run
    '''
    
    #change directory to Metronamica's directory
    notebook_dir = os.getcwd()
    metronamica_dir = r"C:/Program Files (x86)/Geonamica/Metronamica/"
    assert os.path.isdir(metronamica_dir)
    os.chdir(metronamica_dir)
    
    #compose arguments for the executable
    arg_exe = ['GeonamicaCmd.exe']
    arg_exe.extend(run_option)
    arg_exe.extend(log_option)
    arg_exe.extend(save_option)
    arg_exe.extend([geoproj])
    
    #run the model
    call(arg_exe)
  
    #change back to original directory
    os.chdir(notebook_dir)
    
    '''
    end of function
    '''