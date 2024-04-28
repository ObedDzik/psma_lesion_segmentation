'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT License.
'''
#%%
import os 

PSMA_SEGMENTATION_FOLDER = '/data/blobfuse/PSMA_PCA_LESIONS_SEGMENTATION/data_resampled_results/'

DATA_FOLDER = os.path.join(PSMA_SEGMENTATION_FOLDER, 'data')
RESULTS_FOLDER = os.path.join(PSMA_SEGMENTATION_FOLDER, 'results')
os.makedirs(RESULTS_FOLDER, exist_ok=True)
WORKING_FOLDER = os.path.dirname(os.path.abspath(__file__))
