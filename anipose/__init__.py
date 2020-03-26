#!/usr/bin/env python3

import sys
import aniposelib as lib

__version__ = '0.8.0'
VERSION = __version__

sys.modules['anipose.lib'] = sys.modules['aniposelib']
