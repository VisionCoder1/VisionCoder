import sys
import os
sys.path.append(os.path.dirname(__file__))
from .base_communication import BaseCommunication
from .TL2ML import TL2ML
from .ML2FC import ML2FC
from .FC2DG import FC2DG
from .DG import DG