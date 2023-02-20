import colorama
import torch
import pdb
import traceback
from colorama import Fore, Back, Style
from torch import autograd
colorama.init()

class BetterAnomalyDetection(autograd.set_detect_anomaly):

    def __init__(self, set_value):
      super(BetterAnomalyDetection, self).__init__(set_value)

    def __enter__(self):
        super(BetterAnomalyDetection, self).__enter__()
        return self

    def __exit__(self, type, value, trace):
        super(BetterAnomalyDetection, self).__exit__()
        if isinstance(value, RuntimeError):
            traceback.print_tb(trace)
            halt(str(value))

def halt(msg):
  print (Fore.RED + "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
  print (Fore.RED + "┃ Software Failure. Press left mouse button to continue ┃")
  print (Fore.RED + "┃        Guru Meditation 00000004, 0000AAC0             ┃")
  print (Fore.RED + "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
  print(Style.RESET_ALL)
  print (msg)
  pdb.set_trace()