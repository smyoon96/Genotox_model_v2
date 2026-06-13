"""utils/viz_advanced.py -- 고급 시각화 stubs (step8 호환)"""
import logging
from pathlib import Path
logger = logging.getLogger("viz_advanced")

def _warn(fn_name):
    logger.info(f"{fn_name}: not implemented -- skipping")

def plot_detailed_confusion_matrix(*a, **kw): _warn("plot_detailed_confusion_matrix")
def plot_multi_model_roc_pr(*a, **kw): _warn("plot_multi_model_roc_pr")
def plot_metric_radar(*a, **kw): _warn("plot_metric_radar")
def plot_feature_block_importance(*a, **kw): _warn("plot_feature_block_importance")
def plot_cv_fold_stability(*a, **kw): _warn("plot_cv_fold_stability")
def plot_endpoint_summary_panel(*a, **kw): _warn("plot_endpoint_summary_panel")
def plot_hyperparam_landscape(*a, **kw): _warn("plot_hyperparam_landscape")
def save_comprehensive_plots(*a, **kw): _warn("save_comprehensive_plots")
