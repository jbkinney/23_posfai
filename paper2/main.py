# Test whole pipeline
import numpy as np
import posfai2 as my
import time
import sys

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Specify run parameters
    L = 1000; alphabet='ACGT'; alpha=len(alphabet); n_order=3; n_adj=5;
    #L, alpha = 6, 4; n_order = L; n_adj = L
    #L, alpha, n_order = 20, 4, 4; n_adj = L
    #L, alpha, n_order = 300, 4, 5; n_adj = n_order
    #L, alpha, n_order = 300, 20, 2; n_adj = L
    ohe_spec_str = my.get_ohe_spec_str(L, n_order=n_order, n_adj=n_adj)

    T, info_dict = my.compute_T(ohe_spec=ohe_spec_str,
                                alpha=alpha,
                                compute_T_inv=True,
                                verbose=True,
                                get_other_info=True)

    timing_dict = info_dict['timing_dict']
    for key, val in timing_dict.items():
        print(f"\t{key}: {val:.4f} sec.")
