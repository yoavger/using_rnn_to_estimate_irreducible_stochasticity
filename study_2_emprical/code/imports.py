#!/usr/bin/env python
# coding: utf-8

# In[5]:

import sys
import os

module_path = os.path.abspath('models/hybrid/')
if module_path not in sys.path:
    sys.path.append(module_path+"")
    
module_path = os.path.abspath('models/data_driven')
if module_path not in sys.path:
    sys.path.append(module_path+"")  


