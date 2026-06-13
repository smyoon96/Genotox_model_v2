"""utils/qm_descriptors.py — QM 기술자 stubs"""
import logging
logger = logging.getLogger("qm_descriptors")

QM_RELEVANCE = {}

def add_electronic_descriptors(df, smiles_col, prefix="qm_"):
    logger.info("add_electronic_descriptors: skipped (COMPUTE_ELECTRONIC=False)")
    return df

def merge_external_qm(df, qm_file, merge_key="No", qm_cols=None, prefix="extqm_"):
    return df, {"status": "skipped"}

def qm_descriptor_summary(*a, **kw):
    return {}
