# Test whole pipeline
import numpy as np
import posfai2 as my
import time
import sys

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Specify run parameters
    #L, alpha = 6, 4; n_order = L; n_adj = L
    L, alpha, n_order = 20, 4, 4; n_adj = L
    #L, alpha, n_order = 300, 4, 5; n_adj = n_order
    #L, alpha, n_order = 300, 20, 2; n_adj = L
    ohe_spec_str = my.get_ohe_spec_str(L, n_order=n_order, n_adj=n_adj)

    timing_dict = {}

    ### Start timing
    start_time = time.perf_counter()

    # Get transformation matrix
    print('ohe_spec_to_T...')
    t0 = time.perf_counter()
    T = my.ohe_spec_to_T(ohe_spec_str, alpha=alpha, compute_inv=False)
    timing_dict['ohe_spec_to_T'] =  time.perf_counter() - t0

    # Get reordering matrix
    print('ohe_spec_to_B...')
    t0 = time.perf_counter()
    B, B_inv = my.ohe_spec_to_B(ohe_spec_str, alpha=alpha)
    timing_dict['ohe_spec_to_B'] = time.perf_counter() - t0

    # Get sim_spec
    print('ohe_to_sim_spec...')
    t0 = time.perf_counter()
    sim_spec_str = my.ohe_to_sim_spec(ohe_spec_str)
    timing_dict['ohe_to_sim_spec'] = time.perf_counter() - t0

    # Get thinning matrix
    print('get_thinning_matrix...')
    t0 = time.perf_counter()
    A, A_inv = my.get_thinning_matrix(sim_spec_str, alpha=alpha)
    timing_dict['get_thinning_matrix'] = time.perf_counter() - t0

    # Get distilling matrix
    print('get_distilling_matrix...')
    t0 = time.perf_counter()
    D, D_inv, gamma = my.get_distilling_matrix(sim_spec_str, alpha=alpha)
    timing_dict['get_distilling_matrix'] = time.perf_counter() - t0

    # Get gauge basis
    print('E computation...')
    t0 = time.perf_counter()
    E = D @ A @ B @ T
    gauge_basis = E[-gamma:, :].T
    timing_dict['E computation'] = time.perf_counter() - t0

    ### End timming
    end_time = time.perf_counter()

    # Display results:
    elapsed_time = end_time - start_time
    print(f'---------------------------\nResults')
    print(f'L: {L}')
    print(f'alpha: {alpha}')
    print(f'n_order: {n_order}')
    print(f'n_adj: {n_adj}')
    print(f'ohe_spec_str: {ohe_spec_str[:80]}...')
    print(f'sim_spec_str: {sim_spec_str[:80]}...')

    M = T.shape[0]
    print(f'M: {M:,d}')
    print(f'gamma: {gamma:,d}')
    print('Unique elements of gauge basis', np.unique(gauge_basis.data))

    print(f"The code took {elapsed_time:.3f} sec.")
    for key, val in timing_dict.items():
        print(f"\t{key}: {val:.4f} sec.")

    # Objects of interest:
    objs_dict = {'    T':T,
                 '    A':A,
                 '    B':B,
                 '    D':D,
                 '    E':E,
                 'gauge_basis':gauge_basis}
    for key, val in objs_dict.items():
        size = val.data.nbytes
        pct = 100*size/(M*M)
        print(f"\t{key}: {size:10,d} bytes, {pct:.3f}% dense.")