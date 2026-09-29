#!/usr/bin/env python
"""
dag_builder.py -- Hard-coded MLP & Attention DAGs, DAG evaluation.

Run:  python dag_builder.py

Outputs:
    mlp_empty_dag.png
    attn_empty_dag.png
    baseline_inserted_dag.png
    baseline_level_dag.png

Naming convention:
    MLP nodes end with "_mlp"; Attention nodes end with "_attn";
    inserted Bootstrap / Modswitch nodes carry the same block suffix.

Model:
    NODES = [(name, op, level_cost, desc)]
    EDGES = [(from, to, #cipher)]
    Bootstrap1 / Bootstrap2 must each have exactly one input edge (u, b, #ct):
        Bootstrap1 -> l_out = L_max,      #boot = #ct
        Bootstrap2 -> l_out = L_max - 1,  #boot = #ct / 2
    Modswitch: l_out = l_in - level_cost (level_cost = level-drop).
    Other nodes: l_out = l_in - level_cost.
    Bootstrap1 / Bootstrap2 / Modswitch contribute 0 to the depth.

Public generic functions:
    evaluate_dag(NODES, EDGES, l_init, l_max, latency=None)
    depth_of(NODES, EDGES)
    budget_of(NODES, EDGES, l_init, l_max)
    print_level_dag_report(NODES, EDGES, l_init, l_max)
    visualize_empty_dag(nodes, edges, title, filename, ...)
"""
import math
from collections import defaultdict, deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# =========================================================================== #
#  1. Latency functions
# =========================================================================== #
def latency_bootstrap1(l):  return 1040.15
def latency_bootstrap2(l):  return 1293.13
def latency_pcmm1(l):       return 366.20 * l + 1797.71
def latency_pcmm2(l):       return 104.64 * l + 2262.12
def latency_pcmm3(l):       return 443.12 * l + 1581.48
def latency_pcmm4(l):       return 377.74 * l + 2095.11
def latency_ccmm1(l):       return 529.58 * l - 159.50
def latency_ccmm2(l):       return 98.67 * l + 1566.84
def latency_polyeval1(l):   return 142.64 * l - 455.72
def latency_polyeval2(l):   return 145.59 * l - 409.08
def latency_polyeval3(l):   return 562.27 * l + 819.37
def latency_polyeval4(l):   return 460.34 * l + 1311.36
def latency_mul(l):         return 0.0663 * l + 0.4192
def latency_add(l):         return 0.0034 * l + 0.0379
def latency_conj(l):        return 0.16 * l + 1.67
def latency_conj_back(l):   return 0.60 * l + 2.02
def latency_sum(l):         return 4.06
def latency_identity(l):    return 0.0

LATENCY = {
    "PCMM1": latency_pcmm1, "PCMM2": latency_pcmm2,
    "PCMM3": latency_pcmm3, "PCMM4": latency_pcmm4,
    "CCMM1": latency_ccmm1, "CCMM2": latency_ccmm2,
    "PolyEval1": latency_polyeval1, "PolyEval2": latency_polyeval2,
    "PolyEval3": latency_polyeval3, "PolyEval4": latency_polyeval4,
    "Mul": latency_mul, "Add": latency_add,
    "Conjugate": latency_conj, "Conjugate back": latency_conj_back,
    "Sum": latency_sum,
    "Bootstrap1": latency_bootstrap1, "Bootstrap2": latency_bootstrap2,
    "Modswitch": latency_identity, "Rotate": latency_identity,
    "Identity": latency_identity,
}


# =========================================================================== #
#  2. DAG definitions
# =========================================================================== #
def get_mlp_empty():
    """MLP block, empty DAG.

    The module names are the ones `get_baseline()` uses for the same modules, so
    that the empty block and the inserted block read the same.
    """
    NODES = [
        ("X_in_mlp",         "Identity",  0, "X"),
        ("Conj_X_mlp",       "Conjugate", 0, "Conjugate X"),
        ("Mul_mask_mlp",     "Mul",       1, "X = mask · X"),
        ("PCMM3_mlp",        "PCMM3",     2, "X = X · Wup"),
        ("Mul_affine_mlp",   "Mul",       1, "X = X·64 & affine X"),
        ("GELU_mlp",         "PolyEval4", 8, "X = GELU(X), deg=127"),
        ("PCMM4_mlp",        "PCMM4",     2, "x = X · Wdo"),
        ("Add_res_mlp",      "Add",       0, "x = x + X"),
        ("x_mask2_mlp",      "Mul",       1, "x = mask · x"),
        ("sum_mu_mlp",       "Sum",       0, "μ = Σ(x)"),
        ("mask_mu_mlp",      "Mul",       1, "μ = mask · μ"),
        ("div_768_mlp",      "Mul",       1, "μ = μ / 768"),
        ("add_z_mlp",        "Add",       0, "z = x − μ"),
        ("mul_z_sq_mlp",     "Mul",       1, "y = z²"),
        ("sum_y_mlp",        "Sum",       0, "s_y = Σ(y)"),
        ("mul_var_mlp",      "Mul",       1, "var = mask·s_y & affine var"),
        ("invsqrt_var_mlp",  "PolyEval2", 5, "λ = Invsqrt(var), deg=15"),
        ("mask_lam_mlp",     "Mul",       1, "λ = mask · λ"),
        ("mul_z_gamma_mlp",  "Mul",       1, "z = z · γ"),
        ("mul_X_out_mlp",    "Mul",       1, "X = z · λ"),
        ("X_out_mlp",        "Identity",  0, "X"),
    ]
    EDGES = [
        ("X_in_mlp",         "Conj_X_mlp",       8),
        ("Conj_X_mlp",       "Mul_mask_mlp",     4),
        ("Mul_mask_mlp",     "PCMM3_mlp",        64),
        ("PCMM3_mlp",        "Mul_affine_mlp",   16),
        ("Mul_affine_mlp",   "GELU_mlp",         16),
        ("GELU_mlp",         "PCMM4_mlp",        16),
        ("PCMM4_mlp",        "Add_res_mlp",      8),
        ("X_in_mlp",         "Add_res_mlp",      8),
        ("Add_res_mlp",      "x_mask2_mlp",      8),
        ("x_mask2_mlp",      "sum_mu_mlp",       8),
        ("x_mask2_mlp",      "add_z_mlp",        8),
        ("sum_mu_mlp",       "mask_mu_mlp",      1),
        ("mask_mu_mlp",      "div_768_mlp",      1),
        ("div_768_mlp",      "add_z_mlp",        1),
        ("add_z_mlp",        "mul_z_sq_mlp",     8),
        ("add_z_mlp",        "mul_z_gamma_mlp",  8),
        ("mul_z_sq_mlp",     "sum_y_mlp",        8),
        ("sum_y_mlp",        "mul_var_mlp",      1),
        ("mul_var_mlp",      "invsqrt_var_mlp",  1),
        ("invsqrt_var_mlp",  "mask_lam_mlp",     1),
        ("mask_lam_mlp",     "mul_X_out_mlp",    1),
        ("mul_z_gamma_mlp",  "mul_X_out_mlp",    8),
        ("mul_X_out_mlp",    "X_out_mlp",        8),
    ]
    return NODES, EDGES


def get_attn_empty():
    """Attention block, empty DAG."""
    NODES = [
        ("X_in_attn",           "Identity",       0, "X"),
        ("Conj_back_X_attn",    "Conjugate back", 0, "Conjugate back X"),
        ("X_div2_attn",         "Mul",            1, "X = X / 2"),
        ("V_Wv_attn",           "PCMM1",          2, "V = X · Wv"),
        ("Conj_V_attn",         "Conjugate",      0, "Conjugate V"),
        ("Q_Wq_attn",           "PCMM1",          2, "Q = X · Wq"),
        ("K_Wk_attn",           "PCMM1",          2, "K = X · Wk"),
        ("QK_mul_attn",         "CCMM1",          4, "QK = Q · Kᵀ / √d"),
        ("qk_div2_attn",        "Mul",            1, "qk = QK / 2"),
        ("qk_affine_attn",      "Mul",            1, "y = qk/2ᵏ & affine y"),
        ("exp_y_attn",          "PolyEval1",      5, "y = Exp(y), deg=15"),
        ("mask_y_attn",         "Mul",            1, "y = mask · y"),
        ("iter1_y_sq_attn",     "Mul",            1, "Iter1: y²"),
        ("iter1_sum_s_attn",    "Sum",            0, "Iter1: s = Σ(y²)"),
        ("iter1_affine_s_attn", "Mul",            1, "Iter1: affine s"),
        ("iter1_invsqrt_attn",  "PolyEval3",      8, "λ = Invsqrt(s), deg=127"),
        ("iter1_ylam_attn",     "Mul",            1, "Iter1: y · λ"),
        ("iter1_y_sq2_attn",    "Mul",            1, "Iter1: y = (yλ)²"),
        ("iter2_y_sq_attn",     "Mul",            1, "Iter2: y²"),
        ("iter2_sum_s_attn",    "Sum",            0, "Iter2: s = Σ(y²)"),
        ("iter2_affine_s_attn", "Mul",            1, "Iter2: affine s"),
        ("iter2_invsqrt_attn",  "PolyEval2",      5, "λ = Invsqrt(s), deg=15"),
        ("iter2_ylam_attn",     "Mul",            1, "Iter2: y · λ"),
        ("iter2_y_sq2_attn",    "Mul",            1, "Iter2: y = (yλ)²"),
        ("iter3_y_sq_attn",     "Mul",            1, "Iter3: y²"),
        ("iter3_sum_s_attn",    "Sum",            0, "Iter3: s = Σ(y²)"),
        ("iter3_affine_s_attn", "Mul",            1, "Iter3: affine s"),
        ("iter3_invsqrt_attn",  "PolyEval2",      5, "λ = Invsqrt(s), deg=15"),
        ("iter3_ylam_attn",     "Mul",            1, "Iter3: y · λ"),
        ("iter3_S_attn",        "Mul",            1, "Iter3: S = (yλ)²"),
        ("conj_S_attn",         "Conjugate",      0, "Conjugate S"),
        ("S_mask_attn",         "Mul",            1, "S = mask · S"),
        ("conj_back_S_attn",    "Conjugate back", 0, "Conjugate back S"),
        ("SV_mul_attn",         "CCMM2",          2, "SV = S · V"),
        ("rotate_SV_attn",      "Rotate",         0, "Rotate SV"),
        ("x_SV_Wo_attn",        "PCMM2",          2, "x = SV · Wo"),
        ("x_mask_attn",         "Mul",            1, "x = mask · x"),
        ("add_out_attn",        "Add",            0, "x = x + X"),
        ("x_mask2_attn",        "Mul",            1, "x = mask · x"),
        ("sum_mu_attn",         "Sum",            0, "μ = Σ(x)"),
        ("mask_mu_attn",        "Mul",            1, "μ = mask · μ"),
        ("div_768_attn",        "Mul",            1, "μ = μ / 768"),
        ("add_z_attn",          "Add",            0, "z = x − μ"),
        ("mul_z_sq_attn",       "Mul",            1, "y = z²"),
        ("sum_y_attn",          "Sum",            0, "s_y = Σ(y)"),
        ("mul_var_attn",        "Mul",            1, "var = mask·s_y & affine var"),
        ("invsqrt_var_attn",    "PolyEval2",      5, "λ = Invsqrt(var), deg=15"),
        ("mask_lam_attn",       "Mul",            1, "λ = Mask(λ)"),
        ("mul_z_gamma_attn",    "Mul",            1, "z = z · γ"),
        ("mul_X_out_attn",      "Mul",            1, "X = z · λ"),
        ("X_out_attn",          "Identity",       0, "X"),
    ]
    EDGES = [
        ("X_in_attn",           "Conj_back_X_attn",    4),
        ("X_in_attn",           "V_Wv_attn",           4),
        ("X_in_attn",           "Q_Wq_attn",           4),
        ("X_in_attn",           "K_Wk_attn",           4),
        ("Conj_back_X_attn",    "X_div2_attn",         8),
        ("V_Wv_attn",           "Conj_V_attn",         4),
        ("Q_Wq_attn",           "QK_mul_attn",         4),
        ("K_Wk_attn",           "QK_mul_attn",         4),
        ("Conj_V_attn",         "SV_mul_attn",         2),
        ("QK_mul_attn",         "qk_div2_attn",        8),
        ("qk_div2_attn",        "qk_affine_attn",      8),
        ("qk_affine_attn",      "exp_y_attn",          8),
        ("exp_y_attn",          "mask_y_attn",         8),
        ("mask_y_attn",         "iter1_y_sq_attn",     8),
        ("mask_y_attn",         "iter1_ylam_attn",     8),
        ("iter1_y_sq_attn",     "iter1_sum_s_attn",    8),
        ("iter1_sum_s_attn",    "iter1_affine_s_attn", 1),
        ("iter1_affine_s_attn", "iter1_invsqrt_attn",  1),
        ("iter1_invsqrt_attn",  "iter1_ylam_attn",     1),
        ("iter1_ylam_attn",     "iter1_y_sq2_attn",    8),
        ("iter1_y_sq2_attn",    "iter2_y_sq_attn",     8),
        ("iter1_y_sq2_attn",    "iter2_ylam_attn",     8),
        ("iter2_y_sq_attn",     "iter2_sum_s_attn",    8),
        ("iter2_sum_s_attn",    "iter2_affine_s_attn", 1),
        ("iter2_affine_s_attn", "iter2_invsqrt_attn",  1),
        ("iter2_invsqrt_attn",  "iter2_ylam_attn",     1),
        ("iter2_ylam_attn",     "iter2_y_sq2_attn",    8),
        ("iter2_y_sq2_attn",    "iter3_y_sq_attn",     8),
        ("iter2_y_sq2_attn",    "iter3_ylam_attn",     8),
        ("iter3_y_sq_attn",     "iter3_sum_s_attn",    8),
        ("iter3_sum_s_attn",    "iter3_affine_s_attn", 1),
        ("iter3_affine_s_attn", "iter3_invsqrt_attn",  1),
        ("iter3_invsqrt_attn",  "iter3_ylam_attn",     1),
        ("iter3_ylam_attn",     "iter3_S_attn",        8),
        ("iter3_S_attn",        "conj_S_attn",         8),
        ("conj_S_attn",         "S_mask_attn",         4),
        ("S_mask_attn",         "conj_back_S_attn",    64),
        ("conj_back_S_attn",    "SV_mul_attn",         128),
        ("SV_mul_attn",         "rotate_SV_attn",      2),
        ("rotate_SV_attn",      "x_SV_Wo_attn",        32),
        ("x_SV_Wo_attn",        "x_mask_attn",         8),
        ("x_mask_attn",         "add_out_attn",        8),
        ("X_div2_attn",         "add_out_attn",        8),
        ("add_out_attn",        "x_mask2_attn",        8),
        ("x_mask2_attn",        "sum_mu_attn",         8),
        ("x_mask2_attn",        "add_z_attn",          8),
        ("sum_mu_attn",         "mask_mu_attn",        1),
        ("mask_mu_attn",        "div_768_attn",        1),
        ("div_768_attn",        "add_z_attn",          1),
        ("add_z_attn",          "mul_z_sq_attn",       8),
        ("add_z_attn",          "mul_z_gamma_attn",    8),
        ("mul_z_sq_attn",       "sum_y_attn",          8),
        ("sum_y_attn",          "mul_var_attn",        1),
        ("mul_var_attn",        "invsqrt_var_attn",    1),
        ("invsqrt_var_attn",    "mask_lam_attn",       1),
        ("mask_lam_attn",       "mul_X_out_attn",      1),
        ("mul_z_gamma_attn",    "mul_X_out_attn",      8),
        ("mul_X_out_attn",      "X_out_attn",          8),
    ]
    return NODES, EDGES


def get_baseline():
    """Connected Attention + MLP baseline DAG (Bootstrap / Modswitch already inserted)."""
    NODES = [
        # Attention
        ("X_in_attn",           "Identity",       0, "X (in, L=9)"),
        ("Boot1_X_attn",        "Bootstrap1",     0, "Boot1(X)"),
        ("ModSwitch_down8_attn","Modswitch",      8, "Level down 8"),
        ("V_Wv_attn",           "PCMM1",          2, "V = X · Wv"),
        ("Q_Wq_attn",           "PCMM1",          2, "Q = X · Wq"),
        ("K_Wk_attn",           "PCMM1",          2, "K = X · Wk"),
        ("Conj_back_X_attn",    "Conjugate back", 0, "Conjugate back X"),
        ("X_div2_attn",         "Mul",            1, "X = X / 2"),
        ("Conj_V_attn",         "Conjugate",      0, "Conjugate V"),
        ("QK_mul_attn",         "CCMM1",          4, "QK = Q · Kᵀ / √d"),
        ("Boot2_QK_attn",       "Bootstrap2",     0, "Boot2(QK)"),
        ("qk_div2_attn",        "Mul",            1, "qk = QK / 2"),
        ("qk_affine_attn",      "Mul",            1, "y = qk/2ᵏ & affine y"),
        ("exp_y_attn",          "PolyEval1",      5, "y = Exp(y), deg=15"),
        ("mask_y_attn",         "Mul",            1, "y = mask · y"),
        ("iter1_y_sq_attn",     "Mul",            1, "Iter1: y²"),
        ("iter1_sum_s_attn",    "Sum",            0, "Iter1: s = Σ(y²)"),
        ("iter1_affine_s_attn", "Mul",            1, "Iter1: affine s"),
        ("Boot1_s_iter1_attn",  "Bootstrap1",     0, "Boot1(s)"),
        ("iter1_invsqrt_attn",  "PolyEval3",      8, "λ = Invsqrt(s), deg=127"),
        ("iter1_ylam_attn",     "Mul",            1, "Iter1: y · λ"),
        ("iter1_y_sq2_attn",    "Mul",            1, "Iter1: y = (yλ)²"),
        ("iter2_y_sq_attn",     "Mul",            1, "Iter2: y²"),
        ("iter2_sum_s_attn",    "Sum",            0, "Iter2: s = Σ(y²)"),
        ("iter2_affine_s_attn", "Mul",            1, "Iter2: affine s"),
        ("Boot1_s_iter2_attn",  "Bootstrap1",     0, "Boot1(s)"),
        ("iter2_invsqrt_attn",  "PolyEval2",      5, "λ = Invsqrt(s), deg=15"),
        ("iter2_ylam_attn",     "Mul",            1, "Iter2: y · λ"),
        ("iter2_y_sq2_attn",    "Mul",            1, "Iter2: y = (yλ)²"),
        ("Boot2_y_attn",        "Bootstrap2",     0, "Boot2(y)"),
        ("iter3_y_sq_attn",     "Mul",            1, "Iter3: y²"),
        ("iter3_sum_s_attn",    "Sum",            0, "Iter3: s = Σ(y²)"),
        ("iter3_affine_s_attn", "Mul",            1, "Iter3: affine s"),
        ("iter3_invsqrt_attn",  "PolyEval2",      5, "λ = Invsqrt(s), deg=15"),
        ("iter3_ylam_attn",     "Mul",            1, "Iter3: y · λ"),
        ("iter3_S_attn",        "Mul",            1, "Iter3: S = (yλ)²"),
        ("conj_S_attn",         "Conjugate",      0, "Conjugate S"),
        ("S_mask_attn",         "Mul",            1, "S = mask · S"),
        ("conj_back_S_attn",    "Conjugate back", 0, "Conjugate back S"),
        ("SV_mul_attn",         "CCMM2",          2, "SV = S · V"),
        ("Boot1_SV_attn",       "Bootstrap1",     0, "Boot1(SV)"),
        ("rotate_SV_attn",      "Rotate",         0, "Rotate SV"),
        ("x_SV_Wo_attn",        "PCMM2",          2, "x = SV · Wo"),
        ("x_mask_attn",         "Mul",            1, "x = mask · x"),
        ("add_out_attn",        "Add",            0, "x = x + X"),
        ("x_mask2_attn",        "Mul",            1, "x = mask · x"),
        ("sum_mu_attn",         "Sum",            0, "μ = Σ(x)"),
        ("mask_mu_attn",        "Mul",            1, "μ = mask · μ"),
        ("div_768_attn",        "Mul",            1, "μ = μ / 768"),
        ("add_z_attn",          "Add",            0, "z = x − μ"),
        ("mul_z_sq_attn",       "Mul",            1, "y = z²"),
        ("sum_y_attn",          "Sum",            0, "s_y = Σ(y)"),
        ("mul_var_attn",        "Mul",            1, "var = mask·s_y & affine var"),
        ("Boot1_var_attn",      "Bootstrap1",     0, "Boot1(var)"),
        ("invsqrt_var_attn",    "PolyEval2",      5, "λ = Invsqrt(var), deg=15"),
        ("mask_lam_attn",       "Mul",            1, "λ = Mask(λ)"),
        ("mul_z_gamma_attn",    "Mul",            1, "z = z · γ"),
        ("mul_X_out_attn",      "Mul",            1, "X = z · λ"),
        ("X_out_attn",          "Identity",       0, "X (attn out)"),

        # MLP
        ("Conj_X_mlp",          "Conjugate",      0, "Conjugate X"),
        ("Boot1_X_mlp",         "Bootstrap1",     0, "Boot1(X)"),
        ("Mul_mask_mlp",        "Mul",            1, "X = mask · X"),
        ("PCMM3_mlp",           "PCMM3",          2, "X = X · Wup"),
        ("Mul_affine_mlp",      "Mul",            1, "X = X·64 & affine X"),
        ("GELU_mlp",            "PolyEval4",      8, "X = GELU(X), deg=127"),
        ("PCMM4_mlp",           "PCMM4",          2, "x = X · Wdo"),
        ("Add_res_mlp",         "Add",            0, "x = x + X"),
        ("Boot2_X_mlp",         "Bootstrap2",     0, "Boot2(X)"),
        ("ModSwitch_down3_mlp", "Modswitch",      3, "Level down 3"),
        ("x_mask2_mlp",         "Mul",            1, "x = mask · x"),
        ("sum_mu_mlp",          "Sum",            0, "μ = Σ(x)"),
        ("mask_mu_mlp",         "Mul",            1, "μ = mask · μ"),
        ("div_768_mlp",         "Mul",            1, "μ = μ / 768"),
        ("add_z_mlp",           "Add",            0, "z = x − μ"),
        ("mul_z_sq_mlp",        "Mul",            1, "y = z²"),
        ("sum_y_mlp",           "Sum",            0, "s_y = Σ(y)"),
        ("mul_var_mlp",         "Mul",            1, "var = mask·s_y & affine var"),
        ("Boot1_var_mlp",       "Bootstrap1",     0, "Boot1(var)"),
        ("invsqrt_var_mlp",     "PolyEval2",      5, "λ = Invsqrt(var), deg=15"),
        ("mask_lam_mlp",        "Mul",            1, "λ = mask · λ"),
        ("mul_z_gamma_mlp",     "Mul",            1, "z = z · γ"),
        ("mul_X_out_mlp",       "Mul",            1, "X = z · λ"),
        ("X_out_mlp",           "Identity",       0, "X (out)"),
    ]
    EDGES = [
        # Attention
        ("X_in_attn",           "Boot1_X_attn",        4),
        ("Boot1_X_attn",        "ModSwitch_down8_attn",4),
        ("Boot1_X_attn",        "Conj_back_X_attn",    4),
        ("ModSwitch_down8_attn","V_Wv_attn",           4),
        ("ModSwitch_down8_attn","Q_Wq_attn",           4),
        ("ModSwitch_down8_attn","K_Wk_attn",           4),
        ("Conj_back_X_attn",    "X_div2_attn",         8),
        ("V_Wv_attn",           "Conj_V_attn",         4),
        ("Q_Wq_attn",           "QK_mul_attn",         4),
        ("K_Wk_attn",           "QK_mul_attn",         4),
        ("QK_mul_attn",         "Boot2_QK_attn",       8),
        ("Boot2_QK_attn",       "qk_div2_attn",        8),
        ("qk_div2_attn",        "qk_affine_attn",      8),
        ("qk_affine_attn",      "exp_y_attn",          8),
        ("exp_y_attn",          "mask_y_attn",         8),
        ("mask_y_attn",         "iter1_y_sq_attn",     8),
        ("mask_y_attn",         "iter1_ylam_attn",     8),
        ("iter1_y_sq_attn",     "iter1_sum_s_attn",    8),
        ("iter1_sum_s_attn",    "iter1_affine_s_attn", 1),
        ("iter1_affine_s_attn", "Boot1_s_iter1_attn",  1),
        ("Boot1_s_iter1_attn",  "iter1_invsqrt_attn",  1),
        ("iter1_invsqrt_attn",  "iter1_ylam_attn",     1),
        ("iter1_ylam_attn",     "iter1_y_sq2_attn",    8),
        ("iter1_y_sq2_attn",    "iter2_y_sq_attn",     8),
        ("iter1_y_sq2_attn",    "iter2_ylam_attn",     8),
        ("iter2_y_sq_attn",     "iter2_sum_s_attn",    8),
        ("iter2_sum_s_attn",    "iter2_affine_s_attn", 1),
        ("iter2_affine_s_attn", "Boot1_s_iter2_attn",  1),
        ("Boot1_s_iter2_attn",  "iter2_invsqrt_attn",  1),
        ("iter2_invsqrt_attn",  "iter2_ylam_attn",     1),
        ("iter2_ylam_attn",     "iter2_y_sq2_attn",    8),
        ("iter2_y_sq2_attn",    "Boot2_y_attn",        8),
        ("Boot2_y_attn",        "iter3_y_sq_attn",     8),
        ("Boot2_y_attn",        "iter3_ylam_attn",     8),
        ("iter3_y_sq_attn",     "iter3_sum_s_attn",    8),
        ("iter3_sum_s_attn",    "iter3_affine_s_attn", 1),
        ("iter3_affine_s_attn", "iter3_invsqrt_attn",  1),
        ("iter3_invsqrt_attn",  "iter3_ylam_attn",     1),
        ("iter3_ylam_attn",     "iter3_S_attn",        8),
        ("iter3_S_attn",        "conj_S_attn",         8),
        ("conj_S_attn",         "S_mask_attn",         4),
        ("S_mask_attn",         "conj_back_S_attn",    64),
        ("conj_back_S_attn",    "SV_mul_attn",         128),
        ("Conj_V_attn",         "SV_mul_attn",         2),
        ("SV_mul_attn",         "Boot1_SV_attn",       2),
        ("Boot1_SV_attn",       "rotate_SV_attn",      2),
        ("rotate_SV_attn",      "x_SV_Wo_attn",        32),
        ("x_SV_Wo_attn",        "x_mask_attn",         8),
        ("x_mask_attn",         "add_out_attn",        8),
        ("X_div2_attn",         "add_out_attn",        8),
        ("add_out_attn",        "x_mask2_attn",        8),
        ("x_mask2_attn",        "sum_mu_attn",         8),
        ("x_mask2_attn",        "add_z_attn",          8),
        ("sum_mu_attn",         "mask_mu_attn",        1),
        ("mask_mu_attn",        "div_768_attn",        1),
        ("div_768_attn",        "add_z_attn",          1),
        ("add_z_attn",          "mul_z_sq_attn",       8),
        ("add_z_attn",          "mul_z_gamma_attn",    8),
        ("mul_z_sq_attn",       "sum_y_attn",          8),
        ("sum_y_attn",          "mul_var_attn",        1),
        ("mul_var_attn",        "Boot1_var_attn",      1),
        ("Boot1_var_attn",      "invsqrt_var_attn",    1),
        ("invsqrt_var_attn",    "mask_lam_attn",       1),
        ("mask_lam_attn",       "mul_X_out_attn",      1),
        ("mul_z_gamma_attn",    "mul_X_out_attn",      8),
        ("mul_X_out_attn",      "X_out_attn",          8),
        # Connection
        ("X_out_attn",          "Conj_X_mlp",          8),
        # MLP
        ("Conj_X_mlp",          "Boot1_X_mlp",         4),
        ("Boot1_X_mlp",         "Mul_mask_mlp",        4),
        ("Mul_mask_mlp",        "PCMM3_mlp",           64),
        ("PCMM3_mlp",           "Mul_affine_mlp",      16),
        ("Mul_affine_mlp",      "GELU_mlp",            16),
        ("GELU_mlp",            "PCMM4_mlp",           16),
        ("PCMM4_mlp",           "Add_res_mlp",         8),
        ("X_out_attn",          "Add_res_mlp",         8),
        ("Add_res_mlp",         "Boot2_X_mlp",         8),
        ("Boot2_X_mlp",         "ModSwitch_down3_mlp", 8),
        ("ModSwitch_down3_mlp", "x_mask2_mlp",         8),
        ("x_mask2_mlp",         "sum_mu_mlp",          8),
        ("x_mask2_mlp",         "add_z_mlp",           8),
        ("sum_mu_mlp",          "mask_mu_mlp",         1),
        ("mask_mu_mlp",         "div_768_mlp",         1),
        ("div_768_mlp",         "add_z_mlp",           1),
        ("add_z_mlp",           "mul_z_sq_mlp",        8),
        ("add_z_mlp",           "mul_z_gamma_mlp",     8),
        ("mul_z_sq_mlp",        "sum_y_mlp",           8),
        ("sum_y_mlp",           "mul_var_mlp",         1),
        ("mul_var_mlp",         "Boot1_var_mlp",       1),
        ("Boot1_var_mlp",       "invsqrt_var_mlp",     1),
        ("invsqrt_var_mlp",     "mask_lam_mlp",        1),
        ("mask_lam_mlp",        "mul_X_out_mlp",       1),
        ("mul_z_gamma_mlp",     "mul_X_out_mlp",       8),
        ("mul_X_out_mlp",       "X_out_mlp",           8),
    ]
    return NODES, EDGES


def get_attn_baseline_inserted():
    """Attention block alone, baseline (Boot1/Boot2/ModSwitch) already inserted.

    Cut out of the connected baseline.  The edges that leave the block for the
    MLP block keep their tensor in `get_mlp_baseline_inserted`; here they have
    no target, so X_out_attn is a sink: the block ends where its output is
    produced.
    """
    nodes, edges = get_baseline()
    keep = {n for n, *_ in nodes if n.endswith("_attn")}
    return ([x for x in nodes if x[0] in keep],
            [x for x in edges if x[0] in keep and x[1] in keep])


def get_mlp_baseline_inserted():
    """MLP block alone, baseline (Boot1/Boot2/ModSwitch) already inserted.

    Cut out of the connected baseline, plus a copy of the node that produces the
    block input -- X_out_attn, renamed X_in_mlp, the same Identity node the MLP
    block has on its own in `get_mlp_empty()`.  The edges that enter the block
    (X_out_attn -> Conj_X_mlp and X_out_attn -> Add_res_mlp, both #cipher 8)
    become its incoming edges, so the block keeps both of them and reads L_init
    from X_in_mlp.
    """
    nodes, edges = get_baseline()
    keep = {n for n, *_ in nodes if n.endswith("_mlp")}
    block = [x for x in nodes if x[0] in keep]
    inner = [x for x in edges if x[0] in keep and x[1] in keep]
    entering = [x for x in edges if x[0] not in keep and x[1] in keep]

    block.insert(0, ("X_in_mlp", "Identity", 0, "X"))
    return block, inner + [("X_in_mlp", v, c) for (_, v, c) in entering]


# =========================================================================== #
#  3. Style
# =========================================================================== #
INK = "#1F2933"
GREY_FILL, GREY_EDGE   = "#EEF1F5", "#B9C2CC"
BLUE_FILL, BLUE_EDGE   = "#D7E7F7", "#6E9BC5"
GREEN_FILL, GREEN_EDGE = "#E3EFD9", "#86A96A"
RED_FILL, RED_EDGE     = "#F6DEDE", "#C08080"
AMBER_FILL, AMBER_EDGE = "#FBEBC8", "#B8860B"
AMBER_TXT = "#8A6400"
SKIP_COLOR = "#B8860B"

STYLE_MAP = {
    "PCMM1": (BLUE_FILL, BLUE_EDGE), "PCMM2": (BLUE_FILL, BLUE_EDGE),
    "PCMM3": (BLUE_FILL, BLUE_EDGE), "PCMM4": (BLUE_FILL, BLUE_EDGE),
    "CCMM1": (BLUE_FILL, BLUE_EDGE), "CCMM2": (BLUE_FILL, BLUE_EDGE),
    "Mul":   (GREEN_FILL, GREEN_EDGE),
    "PolyEval1": (RED_FILL, RED_EDGE), "PolyEval2": (RED_FILL, RED_EDGE),
    "PolyEval3": (RED_FILL, RED_EDGE), "PolyEval4": (RED_FILL, RED_EDGE),
    "Bootstrap1": (AMBER_FILL, AMBER_EDGE), "Bootstrap2": (AMBER_FILL, AMBER_EDGE),
    "Identity": (GREY_FILL, GREY_EDGE), "Add": (GREY_FILL, GREY_EDGE),
    "Sum": (GREY_FILL, GREY_EDGE), "Conjugate": (GREY_FILL, GREY_EDGE),
    "Conjugate back": (GREY_FILL, GREY_EDGE),
    "Modswitch": (GREY_FILL, GREY_EDGE), "Rotate": (GREY_FILL, GREY_EDGE),
}


# =========================================================================== #
#  4. Layout
# =========================================================================== #
def _vertical_layout(nodes, edges, x_spacing, y_spacing):
    """Layer = longest path from a source; y grows downward."""
    succ, pred = defaultdict(list), defaultdict(list)
    indeg = {n: 0 for n, *_ in nodes}
    for e in edges:
        u, v = e[0], e[1]
        succ[u].append(v); pred[v].append(u); indeg[v] += 1

    layer = {n: 0 for n, *_ in nodes}
    q = deque([n for n in indeg if indeg[n] == 0])
    while q:
        n = q.popleft()
        for m in succ[n]:
            layer[m] = max(layer[m], layer[n] + 1)
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)

    layers = defaultdict(list)
    for n, *_ in nodes:
        layers[layer[n]].append(n)

    x = {}
    for L in sorted(layers):
        names = layers[L]
        if L == 0:
            order = sorted(names)
        else:
            def k(n):
                xs = [x[p] for p in pred[n] if p in x]
                return sum(xs) / len(xs) if xs else 0.0
            order = sorted(names, key=k)
        cnt = len(order)
        for i, n in enumerate(order):
            x[n] = i - (cnt - 1) / 2.0

    pos = {n: (x[n] * x_spacing, -layer[n] * y_spacing) for n, *_ in nodes}
    return pos, layer


# =========================================================================== #
#  5. Visualization
# =========================================================================== #
def visualize_empty_dag(nodes, edges, title, filename,
                        x_spacing=5.0, y_spacing=3.1,
                        show_ct=True, show_level=True, levels=None):
    """Draw a vertical DAG.

    Normal edges: straight.  Cross-layer (residual) edges: arc3 curves with
    distinct rad values.  Each edge is labelled with #ct (and level L if
    `levels` is provided).
    """
    pos, layer = _vertical_layout(nodes, edges, x_spacing, y_spacing)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    bw, bh = 4.4, 1.45

    cross_edges = [e for e in edges if layer[e[1]] - layer[e[0]] > 1]
    cross_sorted = sorted(cross_edges, key=lambda e: -pos[e[0]][1])
    rad_map = {id(e): 0.20 + 0.10 * i for i, e in enumerate(cross_sorted)}

    x_min = min(xs) - bw / 2 - 0.5
    x_max = max(xs) + bw / 2 + 0.5
    y_min = min(ys) - bh / 2 - 0.8
    y_max = max(ys) + bh / 2 + 0.8

    span_x, span_y = x_max - x_min, y_max - y_min
    aspect = span_y / span_x

    fig_w = max(12, span_x * 0.55)
    fig_h = max(16, fig_w * aspect)
    if fig_h > 48:
        fig_h = 48
        fig_w = fig_h / aspect
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_title(title, fontsize=14, color=INK, pad=15)
    ax.axis("off")

    for e in edges:
        u, v, ct = e[0], e[1], e[2]
        x0, y0 = pos[u]; x1, y1 = pos[v]

        parts = []
        if show_ct:
            parts.append(f"#ct={ct}")
        if show_level and levels and (u, v) in levels:
            parts.append(f"L={levels[(u, v)]}")
        lbl = ", ".join(parts)

        if layer[v] - layer[u] > 1:
            rad = rad_map[id(e)]
            sx, sy = x0, y0 - bh / 2
            ex, ey = x1, y1 + bh / 2
            ax.add_patch(FancyArrowPatch(
                (sx, sy), (ex, ey),
                arrowstyle="-|>", mutation_scale=12,
                lw=1.4, color=SKIP_COLOR, zorder=1,
                connectionstyle=f"arc3,rad={rad}",
                clip_on=False))
            dx, dy = ex - sx, ey - sy
            chord = math.hypot(dx, dy) or 1.0
            px, py = -dy / chord, dx / chord
            mid_x = (sx + ex) / 2 + rad * chord * px * 0.5
            mid_y = (sy + ey) / 2 + rad * chord * py * 0.5
            if lbl:
                ax.text(mid_x, mid_y, lbl,
                        ha="center", va="center", fontsize=7.0,
                        color=AMBER_TXT, fontweight="bold", zorder=2,
                        bbox=dict(boxstyle="round,pad=0.10",
                                  facecolor="white", edgecolor=SKIP_COLOR,
                                  alpha=0.95))
        else:
            sx, sy = x0, y0 - bh / 2
            ex, ey = x1, y1 + bh / 2
            ax.add_patch(FancyArrowPatch(
                (sx, sy), (ex, ey),
                arrowstyle="-|>", mutation_scale=12,
                lw=1.2, color=INK, zorder=1))
            if lbl:
                mx, my = (sx + ex) / 2, (sy + ey) / 2
                ax.text(mx + 0.18, my, lbl,
                        ha="left", va="center", fontsize=7.0,
                        color=AMBER_TXT, fontweight="bold", zorder=2,
                        bbox=dict(boxstyle="round,pad=0.10",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.92))

    for name, op, d, desc in nodes:
        x, y = pos[name]
        fill, edge = STYLE_MAP.get(op, (GREY_FILL, GREY_EDGE))
        ax.add_patch(FancyBboxPatch(
            (x - bw / 2, y - bh / 2), bw, bh,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            lw=1.4, facecolor=fill, edgecolor=edge, zorder=3))
        ax.text(x, y + 0.46, name, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=INK, zorder=4)
        ax.text(x, y, desc, ha="center", va="center",
                fontsize=7.2, color=INK, zorder=4, style="italic")
        ax.text(x, y - 0.46, f"{op}  (d={d})", ha="center", va="center",
                fontsize=7.0, color="#5A6672", zorder=4)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(filename, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[ok] wrote {filename}")


# =========================================================================== #
#  6. Generic DAG evaluation
# =========================================================================== #
def _topological_order(NODES, EDGES):
    """Raises ValueError on a cycle."""
    succ, pred = defaultdict(list), defaultdict(list)
    indeg = {n: 0 for n, *_ in NODES}
    for e in EDGES:
        u, v = e[0], e[1]
        succ[u].append(v); pred[v].append(u); indeg[v] += 1
    q = deque([n for n in indeg if indeg[n] == 0])
    topo = []
    while q:
        n = q.popleft(); topo.append(n)
        for m in succ[n]:
            indeg[m] -= 1
            if indeg[m] == 0:
                q.append(m)
    if len(topo) != len(NODES):
        raise ValueError("The graph contains a cycle.")
    return topo, succ, pred


def _bootstrap_cipher_from_edges(NODES, EDGES):
    """Map each Bootstrap node to the #cipher on its unique input edge.

    Raises ValueError if the node has zero or multiple input edges.
    """
    bs_names = {n for n, op, *_ in NODES
                if op in ("Bootstrap1", "Bootstrap2")}
    cts = {n: [] for n in bs_names}
    for u, v, ct in EDGES:
        if v in bs_names:
            cts[v].append(ct)

    result = {}
    for n, lst in cts.items():
        if len(lst) != 1:
            raise ValueError(
                f"Bootstrap node '{n}' must have exactly one input edge "
                f"(found {len(lst)})."
            )
        result[n] = lst[0]
    return result


def evaluate_dag(NODES, EDGES, l_init, l_max, latency=None):
    """Push levels through the DAG.

    Rules:
      source node:      l_in = l_init
      multi-input node: l_in = min(l_out[preds])
      Bootstrap1:       l_out = l_max
      Bootstrap2:       l_out = l_max - 1
      Modswitch:        l_out = l_in - d_v   (d_v = level-drop)
      other:            l_out = l_in - d_v

    `latency` optionally replaces the module-level LATENCY table with another
    {op: f(level)} dict, for a DAG whose operators are not the ones above.
    A latency function is called with the input level of its node; the
    per-ciphertext factor of a Bootstrap is applied here either way.

    Every operator of NODES must have a latency function in the table.  A
    missing one raises ValueError -- it is a typo or an operator that was never
    measured, not a free operator; write a function returning 0.0 for an
    operator that is genuinely free.

    Returns {'l_in', 'l_out', 'edge_lvl', 'latency'}, or None on level underflow.
    """
    topo, succ, pred = _topological_order(NODES, EDGES)

    op_of = {n: op for n, op, *_ in NODES}
    d_of  = {n: d  for n, _, d, *_ in NODES}
    bs_ct = _bootstrap_cipher_from_edges(NODES, EDGES)
    table = LATENCY if latency is None else latency

    missing = sorted({op for _, op, *_ in NODES if op not in table})
    if missing:
        raise ValueError(
            f"No latency function for operator(s): {', '.join(missing)}.  "
            f"Every operator of NODES needs one -- a missing entry is not a "
            f"free operator."
        )

    l_in, l_out = {}, {}
    total = 0.0                   # not `latency`: that name is the table argument

    for name in topo:
        lin = l_init if not pred[name] else min(l_out[p] for p in pred[name])
        l_in[name] = lin

        op = op_of[name]
        if op == "Bootstrap1":
            lout = l_max
        elif op == "Bootstrap2":
            lout = l_max - 1
        else:
            lout = lin - d_of[name]
        l_out[name] = lout
        if lout < 0:
            return None

        c = table[op](lin)
        if op == "Bootstrap1":
            c *= bs_ct[name]
        elif op == "Bootstrap2":
            c *= bs_ct[name] // 2
        total += c

    edge_lvl = {(u, v): l_out[u] for (u, v, _) in EDGES}
    return {"l_in": l_in, "l_out": l_out, "edge_lvl": edge_lvl, "latency": total}


# =========================================================================== #
#  7. Depth / budget / utilization
# =========================================================================== #
def depth_of(NODES, EDGES):
    """Level-cost depth; Bootstrap1/Bootstrap2/Modswitch contribute 0."""
    topo, _, pred = _topological_order(NODES, EDGES)

    op_of = {n: op for n, op, *_ in NODES}
    d_of  = {n: d  for n, _, d, *_ in NODES}

    def eff_d(name):
        return 0 if op_of[name] in ("Bootstrap1", "Bootstrap2", "Modswitch") \
               else d_of[name]

    depth_v = {}
    for name in topo:
        base = 0 if not pred[name] else max(depth_v[p] for p in pred[name])
        depth_v[name] = base + eff_d(name)
    return max(depth_v.values())


def budget_of(NODES, EDGES, l_init, l_max):
    """budget = #B1 * L_max + #B2 * (L_max - 1) + L_init.

    #B1 = sum of #cipher on Bootstrap1 input edges;
    #B2 = sum of #cipher/2 on Bootstrap2 input edges.
    """
    bs_ct = _bootstrap_cipher_from_edges(NODES, EDGES)

    n_b1 = 0
    n_b2 = 0
    for name, op, *_ in NODES:
        if op == "Bootstrap1":
            n_b1 += bs_ct[name]
        elif op == "Bootstrap2":
            n_b2 += bs_ct[name] // 2

    budget = n_b1 * l_max + n_b2 * (l_max - 1) + l_init
    return n_b1, n_b2, budget


# =========================================================================== #
#  8. Report
# =========================================================================== #
def print_level_dag_report(NODES, EDGES, l_init, l_max):
    """Print edge/node level tables and aggregate metrics for any DAG.

    Returns the dict from evaluate_dag(), or None if infeasible.
    """
    res = evaluate_dag(NODES, EDGES, l_init, l_max)

    print("=" * 72)
    print(f"DAG evaluation    L_init = {l_init}    L_max = {l_max}")
    print("=" * 72)

    if res is None:
        print("  INFEASIBLE  (some node would run below its level cost)")
        return None

    l_in, l_out = res["l_in"], res["l_out"]
    edge_lvl    = res["edge_lvl"]

    print(f"\n--- Edge levels ({len(EDGES)} edges) ---")
    print(f"  {'from':<24} {'to':<24} {'#ct':>4} {'L':>4}")
    print("  " + "-" * 60)
    for (u, v, ct) in EDGES:
        print(f"  {u:<24} {v:<24} {ct:>4} {edge_lvl[(u, v)]:>4}")

    print(f"\n--- Node levels ({len(NODES)} nodes) ---")
    print(f"  {'node':<24} {'op':<14} {'d_v':>4} {'l_in':>5} {'l_out':>6}")
    print("  " + "-" * 60)
    for (name, op, d, *_) in NODES:
        print(f"  {name:<24} {op:<14} {d:>4} "
              f"{l_in[name]:>5} {l_out[name]:>6}")

    depth = depth_of(NODES, EDGES)
    n_b1, n_b2, budget = budget_of(NODES, EDGES, l_init, l_max)
    utilization = depth / budget if budget > 0 else 0.0

    print()
    print(f"  total latency    = {res['latency']:.4f}")
    print(f"  #B1 (per ct)     = {n_b1}")
    print(f"  #B2 (per ct)     = {n_b2}")
    print(f"  #B  = #B1 + #B2  = {n_b1 + n_b2}")
    print(f"  depth            = {depth}   "
          f"(Bootstrap1/Bootstrap2/Modswitch contribute 0 to the depth)")
    print(f"  budget           = {n_b1}*{l_max} + {n_b2}*{l_max - 1} "
          f"+ {l_init}  =  {budget}")
    print(f"  utilization      = depth / budget = "
          f"{depth} / {budget} = {utilization:.6f}")

    return res


# =========================================================================== #
#  9. main
# =========================================================================== #
def main():
    nodes, edges = get_mlp_empty()
    visualize_empty_dag(
        nodes, edges,
        title="MLP Block -- empty DAG",
        filename="mlp_empty_dag.png",
        x_spacing=9.0, y_spacing=2.4,
    )

    nodes, edges = get_attn_empty()
    visualize_empty_dag(
        nodes, edges,
        title="Attention Block -- empty DAG",
        filename="attn_empty_dag.png",
        x_spacing=9.0, y_spacing=2.4,
    )

    nodes, edges = get_baseline()
    visualize_empty_dag(
        nodes, edges,
        title="Baseline Attention + MLP block -- (Boot1/Boot2/ModSwitch) inserted DAG",
        filename="baseline_inserted_dag.png",
        x_spacing=9.0, y_spacing=2.4,
    )

    L_INIT, L_MAX = 9, 14
    res = print_level_dag_report(nodes, edges, L_INIT, L_MAX)

    if res is not None:
        visualize_empty_dag(
            nodes, edges,
            title=f"Baseline level DAG -- L_init = {L_INIT}, L_max = {L_MAX}",
            filename="baseline_level_dag.png",
            x_spacing=9.0, y_spacing=2.4,
            show_level=True,
            levels=res["edge_lvl"],
        )

    # Each block on its own is a problem of its own, with the same baseline
    # inserted nodes.  The Attention block starts where the block starts, at
    # L_INIT = 9; the MLP block starts at the level X_out_attn hands over in the
    # connected baseline above, so its L_init = 6.  The MLP block is the first
    # task (Section 4.1), the Attention block the second one (Section 4.2).
    for get_block, block, block_l_init, filename in (
        (get_mlp_baseline_inserted,  "MLP block alone",       6, "mlp_baseline_level_dag.png"),
        (get_attn_baseline_inserted, "Attention block alone", 9, "attn_baseline_level_dag.png"),
    ):
        nodes, edges = get_block()
        res = print_level_dag_report(nodes, edges, block_l_init, L_MAX)
        if res is not None:
            visualize_empty_dag(
                nodes, edges,
                title=f"{block} -- baseline inserted DAG, "
                      f"L_init = {block_l_init}, L_max = {L_MAX}",
                filename=filename,
                x_spacing=9.0, y_spacing=2.4,
                show_level=True,
                levels=res["edge_lvl"],
            )


if __name__ == "__main__":
    main()