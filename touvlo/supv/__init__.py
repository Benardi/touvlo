import pbr.version

from touvlo.supv import lgx_rg
from touvlo.supv import lin_rg
from touvlo.supv import nn_clsf

__version__ = pbr.version.VersionInfo('touvlo').version_string()
__all__ = ['lgx_rg', 'lin_rg', 'nn_clsf']
