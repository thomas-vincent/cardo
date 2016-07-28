import logging
import sys

from cardo.version import __version__

from _cardo import *
from cardo import tree
from cardo import graphics
from cardo import commands

logging.basicConfig(stream=sys.stdout)
