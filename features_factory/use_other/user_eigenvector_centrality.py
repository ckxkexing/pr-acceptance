'''
        
'''
import os
import re
import json
from .utils import Timer
from .utils import write_csv_data
from tqdm import tqdm

def cal_user_eigenvector_centrality():
    ''' 
        two steps 
        1. use commit data to build a network of *source code* files: connect files in same commit
        2. sum of eigenvector of files included in newcommer's commits. one sum  
    '''

    pass