# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 16:52:47 2020

@author: Rajan
"""

import predictor as myapp

# This is just a simple wrapper for gunicorn to find your app.
# If you want to change the algorithm file, simply change "predictor" above to the
# new file.

app = myapp.app