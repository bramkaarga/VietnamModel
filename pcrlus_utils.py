try:
    from pcraster import *
except:
    pass
import numpy as np
import glob

def pcrfile2np(infile, nanval):
        
    pcrnp = pcr2numpy(readmap(infile), nanval)
    
    return pcrnp

def pcrfile2ascfile(infile, outfile, nanval, nan2num=False, delimiter=' '):
    
    pcrnp = pcrfile2np(infile, nanval)
    
    #write each row in ndarray
    with open(outfile, 'w') as fh:
        fmt="%.6e"
        fmti="%i"
        for row in pcrnp:
            if nan2num:
                row[np.isnan(row)] = nanval
            line = delimiter.join(str(fmti % value) if value % 1 == 0 else str(fmt % value) for value in row)
            fh.write(line + '\n')

def numpy2asc(ndarray, filename, savedir, xll, yll, cs, ndv, nan2num=False, delimiter=' '):
    #from https://stackoverflow.com/questions/24691755/how-to-format-in-numpy-savetxt-such-that-zeros-are-saved-only-as-0
    with open(savedir+filename, 'w') as fh:
        
        #write header
        nrows, ncols = ndarray.shape
        header='\n'.join(["ncols " + str(ncols), 
                      "nrows " + str(nrows), 
                      'xllcorner ' + str(xll), 
                      'yllcorner ' + str(yll),
                      'cellsize ' + str(cs),
                      'NODATA_value ' + str(ndv),
                        ''])
        fh.write(header)
        
        #write each row in ndarray
        fmt="%.6e"
        fmti="%i"
        for row in ndarray:
            if nan2num:
                row[np.isnan(row)] = ndv
            line = delimiter.join(str(fmti % value) if value % 1 == 0 else str(fmt % value) for value in row)
            fh.write(line + '\n')
#end

def load_result_lus(result_dir, rows=6, opt=1, nan2num=0):
    
    ## OPTION 1: store in a dictionary
    if opt == 1:
        all_results = {}
        for file in glob.glob(result_dir+'*.asc'):
            name = file[len(result_dir):-4]
            all_results[name] = np.loadtxt(file,skiprows=rows)
            if nan2num != 0:
                all_results[name][np.isnan(all_results[name])] = nan2num
        return all_results
    
    elif opt == 2:
        for file in glob.glob(result_dir+'*.asc'):
            name = file[len(result_dir):-4]
            exec("globals()['" + name + "']"+ " = np.loadtxt(file, skiprows=rows)")
            print(name+' has been created as an ndarray')