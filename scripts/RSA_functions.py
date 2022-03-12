import numpy as np
import scipy.stats as stats
from sklearn import linear_model

def RSA(data_1, data_2, sliding_window=False, window_width=None) :

	"""
	Perform representational similarity analysis using two pre-proccessed datasets.
  
	Parameters
	----------
    	data_1, data_2 (numpy.ndarray) :
		The two datasets to be correlated. Data must be two dimensional, with rows
		representing variables (ex: TRs) and columns representing variables (ex: voxel
		responses).
	sliding_window (bool, optional) : 
		If sliding_window is False (default), the comparison of the RSMs will correlate
		the entire upper triangles of the RSMs. Otherwise, the RSMs will be compared with
		sliding window correlations, using a square window whose width is given by the
		window_width parameter.
	window_width (int, optional) : 
		The width in TRs of the sliding window used to correlate the RSMs.  
  
   	Returns
	-------
    	results (tuple) :
		A tuple containing both RSMs as two dimensional ndarrays, the between-RSM
		correlation, the associated p-value, and the local correlations if 
		sliding_window is True.	
	"""

	# Handle errors
	if not(isinstance(data_1, np.ndarray)) or not(isinstance(data_2, np.ndarray)) :
		raise TypeError("Data must be a numpy ndarray")
	
	if data_1.ndim != 2 or data_2.ndim != 2 :
		raise ValueError("Data must be two dimensional (variables x observations)")
	if sliding_window == True and window_width == None :
		raise ValueError("window_width cannot be None for sliding window correlations")
	

	# Create the RSMs for each data set
	RSM_1 = np.corrcoef(data_1)
	RSM_2 = np.corrcoef(data_2)
	

	# Use entire upper triangles for between-RSM correlations
	if (sliding_window==False) :
	
		# Pull out the upper triangles of the RSMs for correlation
		triu_idx = np.triu_indices(RSM_1.shape[0], k=1)
		triu_1 = RSM_1[triu_idx]
		triu_2 = RSM_2[triu_idx]
		# Compute the Pearson correlation and p value
		corr, pval = stats.pearsonr(triu_1, triu_2)
		# Store results in tuple and return
		results = (RSM_1, RSM_2, corr, pval)
		return results
		
	# Otherwise, use sliding windows for between-RSM correlations
	
	# Variables for sliding window correlations
	n_trs = data_1.shape[0]
	window_width = window_width
	n_windows = n_trs - window_width
	sliding_corrs = np.zeros(n_windows)
	sliding_pvals = np.zeros_like(sliding_corrs)
	

	# Check that window_width is less than the length of the dataset
	if window_width >= n_trs :
		raise ValueError("Window width must be less than the length of the dataset")
	

	# Calculate sliding window correlations between the RSMs
	for i in range(n_windows) :
		# Isolate the current window of each RSM
		win_1  = chop(RSM_1, i, i+window_width)
		win_2  = chop(RSM_2, i, i+window_width)
		# Pull out the upper triangles of the RSMs for correlation
		triu_idx = np.triu_indices(window_width, k=1)
		win_1_triu = win_1[triu_idx]
		win_2_triu = win_2[triu_idx]
		# Compute the Pearson correlation and p value
		sliding_corrs[i], sliding_pvals[i]  = stats.pearsonr(win_1_triu, win_2_triu)
	
	# Average the statistics over the windows
	avg_corr, avg_pval = np.mean(sliding_corrs), np.mean(sliding_pvals)

	# Store results in tuple and return
	results = (RSM_1, RSM_2, avg_corr, avg_pval, sliding_corrs)
	return results

	
def corr_prep(run1_data, run2_data, run1_predictor=None, run2_predictor=None, transpose=True, regress=False) :

	"""
	RSA preprocessing for neural data. Averages across subjects, concatenates the runs,
	and then may transpose or regress out part of the neural data based on the parameters
        tranpsose and regress.
  
	Parameters
	----------
	run1_data, run2_data (numpy.ndarray) :
		The two runs of three dimensional data to be processed. Data should be either
		voxels x time x subjects (use transpose=True) or time x voxels x subjects 
		(use transpose=False).
        run1_predictor, run2_predictor (numpy.ndarray) :
                The two runs of three dimensional data to be used as predictors for run1_data 
                and run2_data in a linear regression model.
	transpose (bool, optional) :
		If tranpose is True (default), the returned matrix is transposed. Otherwise,
		the orientation of the first two dimensions is unchanged.
        regress (bool, optional) :
                If regress is True, processed_data is the part of run1_data and run2_data
                that isn't explained by run1_predictor and run2_predictor. Otherwise (default),
                no regression is performed.

        Returns
	-------
	processed_data (numpy.ndarray) :
		The processed data.
	
	"""

	# Make sure we're working with data of the correct type and dimensionality
	if not(isinstance(run1_data, np.ndarray)) or not(isinstance(run2_data, np.ndarray)) :
		raise TypeError("Data must be a numpy ndarray")

	if run1_data.ndim != 3 or run2_data.ndim != 3 :
		raise ValueError("Data must be 3 dimensional (voxels x time x subjects or time x voxels x subjects)")

	# Average each run across subjects
	run1_avg = np.mean(run1_data, axis=2)
	run2_avg = np.mean(run2_data, axis=2)

	# Concatenate the data across time
	processed_data  = np.hstack((run1_avg, run2_avg))

	# Check if the data needs to be transposed
	if (transpose) :
		processed_data = processed_data.T

        # Check if we need to regress out run1_predictor and run2_predictor
        if (regress) :
           
            # Check that predictor data was provided
            if ((run1_predictor is None) or (run2_predictor is None)) : 
                raise ValueError("run1_predictor and run2_predictor must not be None when using regress=True")

            # Average over subjects, concatenate runs of predicting data
            predictor_data = corr_pre(run1_predictor, run2_predictor)

            # Create and fit the linear model
            reg = linear_model.LinearRegression()
            reg.fit(predictor_data, processed_data)

            # Subtract the part of processed_data that can be explained by predictor_data
	    processed_data = processed_data - np.dot(reg.coef_, predictor_data) - reg.intercept_[:, np.newaxis]

	return processed_data



def chop(mat, idx_lo, idx_hi) : 

	"""
	Chop off desired values from a square matrix symmetric about its diagonal.
  
	Parameters
	----------
	mat (numpy.ndarray) :
		The matrix to be chopped. Must be two dimensional.
	idx_low, idx_hi (int) : 
		The indices describing the range of desired elements in the both dimensions of the matrix.    	
   	Returns
	-------
	chopped_mat (nump.ndarray) :
		The chopped matrix.    	
	"""

	chopped_mat = mat[idx_lo:idx_hi, idx_lo:idx_hi]
	return chopped_mat
