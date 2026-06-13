"""utils/chemical_space.py -- 화학공간 시각화 stubs"""
import logging
logger = logging.getLogger("chemical_space")

def _warn(fn_name): logger.info(f"{fn_name}: not implemented -- skipping")

def plot_tsne_chemical_space(*a, **kw): _warn("plot_tsne_chemical_space")
def plot_chemical_space_tsne(*a, **kw):  _warn("plot_chemical_space_tsne")   # step8 alias
def plot_chemical_space_pca(*a, **kw):   _warn("plot_chemical_space_pca")    # step8 alias
def plot_ad_overlay(*a, **kw): _warn("plot_ad_overlay")
def plot_qm_correlation_heatmap(*a, **kw): _warn("plot_qm_correlation_heatmap")
def plot_qm_space(*a, **kw): _warn("plot_qm_space")
