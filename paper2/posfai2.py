###
### Start again from here
###
import itertools

import numpy as np
import scipy.sparse as sp
from itertools import product, chain, combinations, islice
import pdb

# # Define useful quantities
# x_triv = np.array([1])
#
# x_ohe_A = np.array([1, 0, 0, 0])
# x_ohe_C = np.array([0, 1, 0, 0])
# x_ohe_G = np.array([0, 0, 1, 0])
# x_ohe_T = np.array([0, 0, 0, 1])
#
# char_to_ohe_dict = {
#     'A': x_ohe_A,
#     'C': x_ohe_C,
#     'G': x_ohe_G,
#     'T': x_ohe_T
# }
#
# x_sim_A = np.array([1, 0, 0])
# x_sim_C = np.array([0, 1, 0])
# x_sim_G = np.array([0, 0, 1])
# x_sim_T = np.array([-1, -1, -1])
#
# char_to_sim_dict = {
#     'A': x_sim_A,
#     'C': x_sim_C,
#     'G': x_sim_G,
#     'T': x_sim_T
# }
#
# # Trivial matrix
# T_triv = sp.csr_array([[1]])
#
# T_ohe = sp.csr_array([
#     [1, 1, 1, 1],
#     [1, 0, 0, -1],
#     [0, 1, 0, -1],
#     [0, 0, 1, -1]
# ])
# T_ohe_inv = sp.csr_matrix(np.linalg.inv(T_ohe.todense()))

def make_all_seqs(L, max_num_seqs=1000):
    iterable = product(['A','C','G','T'], repeat=L)
    return [''.join(x) for x in islice(iterable, max_num_seqs)]

def make_random_seqs(L, N, alphabet='ACGT'):
    return [''.join(row) for row in
            np.random.choice(a=list(alphabet), size=[N,L])]

# def L_to_spec_str_constant():
#     return '.'
#
# def L_to_spec_str_additive(L):
#     parts = []
#     parts.append(L_to_spec_str_constant())
#     for i in range(L):
#         parts.append(f'{i:d}')
#     return '+'.join(parts)
#
# def L_to_spec_str_neighbor(L):
#     parts = []
#     parts.append(L_to_spec_str_additive(L))
#     for i in range(L-1):
#         parts.append(f'{i:d}x{i+1:d}')
#     return '+'.join(parts)
#
# def L_to_spec_str_pairwise(L):
#     parts = []
#     parts.append(L_to_spec_str_additive(L))
#     for i in range(L-1):
#         for j in range(i+1,L):
#             parts.append(f'{i:d}x{j:d}')
#     return '+'.join(parts)
#
# def L_to_spec_str_3adjacent(L):
#     parts = []
#     parts.append(L_to_spec_str_neighbor(L))
#     for i in range(L-2):
#         parts.append(f'{i:d}x{i+1:d}x{i+2}')
#     return '+'.join(parts)
#
# def L_to_spec_str_3order(L):
#     parts = []
#     parts.append(L_to_spec_str_pairwise(L))
#     for i in range(L-2):
#         for j in range(i+1,L-1):
#             for k in range(j+1,L):
#                 parts.append(f'{i:d}x{j:d}x{k:d}')
#     return '+'.join(parts)

def seq_to_x_ohe(seq, ohe_spec_str, alphabet='ACGT', sparse=True):
    '''
    inputs: seq (DNA), ohe_spec_str
    returns: x, a one-hot encoding
    '''
    L = len(seq)
    x_components = []

    x_triv = sp.coo_array([1])
    char_to_ohe_dict = get_char_to_ohe_dict(alphabet=alphabet)

    ohe_spec_str_parts = ohe_spec_str.split('+')
    for part in ohe_spec_str_parts:

        # Add in trivial component
        if part == '.':
            x_components.append(x_triv)
        else:
            positions = [int(p) for p in part.split('x')]
            assert len(positions) > 0
            x_irr = x_triv
            while len(positions) > 0:
                pos = positions.pop(-1)
                c = seq[pos]
                x_l = sp.coo_array(char_to_ohe_dict[c])
                x_irr = sp.kron(x_irr, x_l, format='coo')
            x_components.append(x_irr)

    # Create x
    #pdb.set_trace()
    x = sp.hstack(x_components, format='csr').T

    if not sparse:
        x = x.todense()

    return x


def ohe_spec_to_T(ohe_spec_str, alpha=4, compute_inv=False):
    '''
    input: ohe_spec_str
    output: T (s.t. T x = x_factored)
    '''

    # Split ohe_spec_str into parts
    parts = ohe_spec_str.split('+')

    # Get maximum order
    max_order = np.max([len(part.split('x')) for part in parts])

    # Get single-position T blocks
    T_ohe, T_ohe_inv = get_single_position_T_and_T_inv(alpha=alpha)
    T_triv = sp.csr_array([[1]])

    if not compute_inv:
        T_ohe_inv = sp.csr_matrix(np.zeros(shape=(alpha-1,alpha-1)))

    # Build blocks for order up to maximum order
    order_to_block_dict = {}
    T_part = T_triv
    T_part_inv = T_triv
    for k in range(max_order + 1):
        order_to_block_dict[k] = (T_part.copy(), T_part_inv.copy())
        T_part = sp.kron(T_part, T_ohe)
        T_part_inv = sp.kron(T_ohe_inv, T_part_inv)

    # Build block matrix
    diag_mats = []
    diag_mats_inv = []
    for part in parts:
        if part == '.':
            order = 0
        else:
            order = len(part.split('x'))
        T_part, T_part_inv = order_to_block_dict[order]
        diag_mats.append(T_part)
        diag_mats_inv.append(T_part_inv)

    T = sp.block_diag(diag_mats, format='csr')
    T_inv = sp.block_diag(diag_mats_inv, format='csr')

    if compute_inv:
        return T, T_inv
    else:
        return T

# ### Define and test ohe_spec_to_T function
# def ohe_spec_to_T(ohe_spec_str, alpha=4):
#     '''
#     input: ohe_spec_str
#     output: T (s.t. T x = x_factored)
#     '''
#     parts = ohe_spec_str.split('+')
#     diag_mats = []
#     diag_mats_inv = []
#
#     T_triv = sp.csr_array([[1]])
#     T_ohe, T_ohe_inv = get_single_position_T_and_T_inv(alpha=alpha)
#
#     for part in parts:
#         if part == '.':
#             T_part = T_triv
#             T_part_inv = T_triv
#         else:
#             #bits = part.split('x')
#             T_part = T_triv
#             T_part_inv = T_triv
#             for bit in part.split('x'):
#                 m = T_part.shape[0]
#
#                 # Compute kronecker product
#                 T_part = sp.kron(T_part, T_ohe)
#                 T_part_inv = sp.kron(T_ohe_inv,T_part_inv)
#         diag_mats.append(T_part)
#         diag_mats_inv.append(T_part_inv)
#     T = sp.block_diag(diag_mats, format='csr')
#     T_inv = sp.block_diag(diag_mats_inv, format='csr')
#     return T, T_inv

### Define and test ohe_spec_to_T function
# def ohe_spec_to_B(ohe_spec_str, alpha=4):
#     '''
#     input: ohe_spec_str
#     output: T (s.t. T x = x_factored)
#     '''
#     parts = ohe_spec_str.split('+')
#     diag_mats = []
#     diag_mats_inv = []
#     I_triv = np.eye(1, dtype=np.int64)
#     I_ohe = np.eye(alpha, dtype=np.int64)
#     for part in parts:
#         if part == '.':
#             B_part = I_triv
#             B_part_inv = I_triv
#         else:
#             bits = part.split('x')
#             B_part = I_triv
#             B_part_inv = I_triv
#             for bit in part.split('x'):
#                 m = B_part.shape[0]
#
#                 # Compute kronecker product
#                 B_part = sp.kron(B_part, I_ohe)
#                 B_part_inv = sp.kron(B_part_inv, I_ohe)
#
#                 # Fix up row orderd (# Is this where this goes?)
#                 i_vals = list(range(m*alpha))
#                 j_vals = [alpha*i for i in range(m)] + [i-m+1+(i-m)//(alpha-1) for i in range(m,m*alpha)]
#                 data = m*alpha*[1]
#                 new_B = sp.coo_array((data, (i_vals, j_vals)), shape=(alpha*m,alpha*m)).tocsr()
#                 B_part = new_B@B_part
#                 B_part_inv = B_part_inv@(new_B.T)
#         diag_mats.append(B_part)
#         diag_mats_inv.append(B_part_inv)
#     B = sp.block_diag(diag_mats, format='csr')
#     B_inv = sp.block_diag(diag_mats_inv, format='csr')
#     return B, B_inv

def ohe_spec_to_B(ohe_spec_str, alpha=4):
    '''
    input: ohe_spec_str
    output: T (s.t. T x = x_factored)
    '''
    # Split ohe_spec_str into parts
    parts = ohe_spec_str.split('+')

    # Get maximum order
    max_order = np.max([len(part.split('x')) for part in parts])

    # Build blocks for order up to maximum order
    I_triv = sp.eye(1, dtype=np.int64)
    I_ohe = sp.eye(alpha, dtype=np.int64)

    order_to_block_dict = {}
    B_part = I_triv
    B_part_inv = I_triv
    for k in range(max_order + 1):
        order_to_block_dict[k] = (B_part.copy(), B_part_inv.copy())

        # Fix up row order
        m = alpha ** k
        B_part = sp.kron(B_part, I_ohe)
        B_part_inv = sp.kron(B_part_inv, I_ohe)

        i_vals = list(range(m * alpha))
        j_vals = [alpha * i for i in range(m)] + [
            i - m + 1 + (i - m) // (alpha - 1) for i in range(m, m * alpha)]
        data = m * alpha * [1]
        new_B = sp.coo_array((data, (i_vals, j_vals)),
                             shape=(alpha * m, alpha * m)).tocsr()
        # pdb.set_trace()
        B_part = new_B @ B_part
        B_part_inv = B_part_inv @ (new_B.T)

    # Build block matrix
    diag_mats = []
    diag_mats_inv = []
    for part in parts:
        if part == '.':
            order = 0
        else:
            order = len(part.split('x'))
        B_part, B_part_inv = order_to_block_dict[order]
        diag_mats.append(B_part)
        diag_mats_inv.append(B_part_inv)

    B = sp.block_diag(diag_mats, format='csr')
    B_inv = sp.block_diag(diag_mats_inv, format='csr')

    return B, B_inv

def my_expand(x):
    """
    Expands a list of lists. Simulates product expansion
    """
    if len(x) >= 1:
        a = x[0]
        b = x[1:]
        b_exp = my_expand(b)
        c = [[y]+z for y in a for z in b_exp]
        return c
    else:
        return [x]

### Convert OHE to SIM spec
def ohe_to_sim_spec(ohe_spec_str):
    a = ohe_spec_str.split('+')
    b = [z.split('x') for z in a]
    for i in range(len(b)):
        for j in range(len(b[i])):
            z = b[i][j]
            if z != '.':
                b[i][j] = ['.', z]

    # Recursive expansion
    c = []
    for i, b_el in enumerate(b):
        if isinstance(b_el, str):
            c.append([b_el])
        elif isinstance(b_el, list) and len(b_el) >= 1:
            c.extend(my_expand(b_el))

    # Remove redundant factors of '.'
    sim_spec_list = []
    for x in c:
        y = [z for z in x if z != '.']
        if len(y) == 0:
            y = ['.']
        sim_spec_list.append(y)
    sim_spec_str = '+'.join(['x'.join(z) for z in sim_spec_list])
    return sim_spec_str


# Compute starting positions for each entry in the sim_spec_str
def get_shifts_and_sizes(spec_str, encoding_size):
    """ inputs spec list. outputs a list of (spec, size, shift) """
    spec_list = [x.split('x') for x in spec_str.split('+')]
    specs = []
    shift = 0
    for x in spec_list:
        if len(x)==1 and x[0]=='.':
            size = 1
        else:
            size = encoding_size**len(x)
        specs.append(('x'.join(x),size,shift))
        shift += size
    M = shift
    return specs, M

# TODO: Fix this function. I'm getting duplicate offsets.
# Should instead define A as coo anyway.
# def get_thinning_matrix(sim_spec_str, alpha=4):
#     # Build zeroing-out matrix
#     component_dict = {}
#     diag_vecs = []
#     inv_diag_vecs = []
#     diag_offsets = []
#
#     # Get specs list
#     specs, M = get_shifts_and_sizes(sim_spec_str, encoding_size=alpha-1)
#
#     # Add in main diagonal
#     diag_vecs.append(M * [1])
#     inv_diag_vecs.append(M * [1])
#     diag_offsets.append(0)
#     for spec in specs:
#         key = spec[0]
#         m = spec[1]
#         offset = spec[2]
#         if key not in component_dict:
#             component_dict[key] = (m, offset)
#         else:
#             m1, offset1 = component_dict[key]
#             try:
#                 assert m1 == m
#             except:
#                 print('m1:', m1)
#                 print('m:', m)
#                 pdb.set_trace()
#             diag_vec = offset1 * [0] + m * [-1] + (M - (offset + m)) * [0]
#             inv_diag_vec = offset1 * [0] + m * [1] + (M - (offset + m)) * [0]
#             diag_offset = -(offset - offset1)
#             diag_vecs.append(diag_vec)
#             inv_diag_vecs.append(inv_diag_vec)
#             diag_offsets.append(diag_offset)
#     A = sp.diags(diag_vecs, diag_offsets)
#     A_inv = sp.diags(inv_diag_vecs, diag_offsets)
#     return A, A_inv

def get_thinning_matrix(sim_spec_str, alpha=4):
    # Build zeroing-out matrix
    component_dict = {}
    diag_vecs = []
    inv_diag_vecs = []
    diag_offsets = []

    # Get specs list
    specs, M = get_shifts_and_sizes(sim_spec_str, encoding_size=alpha - 1)

    i_vals = list(range(M))
    j_vals = list(range(M))
    data = M * [1]
    data_inv = M * [1]
    for spec in specs:
        key = spec[0]
        m = spec[1]
        offset = spec[2]
        if key not in component_dict:
            component_dict[key] = (m, offset)
        else:
            m1, offset1 = component_dict[key]
            try:
                assert m1 == m
            except:
                print('m1:', m1)
                print('m:', m)
                pdb.set_trace()

            i_start = offset
            j_start = offset1
            i_vals.extend(list(range(i_start, i_start + m)))
            j_vals.extend(list(range(j_start, j_start + m)))
            data.extend(m * [-1])
            data_inv.extend(m * [1])
    A = sp.coo_array((data, (i_vals, j_vals)), shape=(M, M)).tocsr()
    A_inv = sp.coo_array((data_inv, (i_vals, j_vals)), shape=(M, M)).tocsr()
    return A, A_inv

def get_x_to_test_thinning_matrix(sim_spec_str, alpha=4):
    """
    input: sim_spec_str
    return: x_test
    """
    # Get shifts and sizes
    specs, M = get_shifts_and_sizes(sim_spec_str, encoding_size=alpha-1)

    # Get unique labels
    labels = []
    for spec in specs:
        key = spec[0]
        if not key in labels:
            labels.append(key)

    # Create labels dict
    labels_dict = {}
    counter = 1
    for label in labels:
        labels_dict[label] = .1 + .9 * counter / len(labels)
        counter += 1

        # Build x_test
    x_test = np.zeros(M)
    for spec in specs:
        key = spec[0]
        m = spec[1]
        offset = spec[2]
        r = labels_dict[key]
        x_test[offset:offset + m] = r

    return x_test


def seq_to_desired_BTx(seq, sim_spec_str, alphabet='ACGT'):
    '''
    inputs: seq, ohe_spec_str, alphabet
    returns: x, a one-hot encoding
    '''
    L = len(seq)
    x_components = []
    x_triv = np.array([1])
    char_to_sim_dict = get_char_to_sim_dict(alphabet=alphabet)

    sim_spec_str_parts = sim_spec_str.split('+')
    for part in sim_spec_str_parts:

        # Add in trivial component
        if part == '.':
            x_components.append(x_triv)
        else:
            positions = [int(p) for p in part.split('x')]
            assert len(positions) > 0
            x_irr = x_triv
            while len(positions) > 0:
                pos = positions.pop(-1)
                c = seq[pos]
                x_l = char_to_sim_dict[c]
                x_irr = np.kron(x_irr, x_l)
            x_components.append(x_irr)

    # Create x
    x = np.concatenate(x_components)
    return x


def get_distilling_matrix(sim_spec_str, alpha=4):
    # Get specs list
    specs, M = get_shifts_and_sizes(sim_spec_str, encoding_size=alpha-1)

    # Lists to hold i and j values
    component_dict = {}
    nonzero_j_vals = []
    zero_j_vals = []
    next_nonzero_j = 0
    next_zero_j = 0
    next_j = 0
    next_i = 0
    beta = 0
    for spec in specs:
        key = spec[0]
        m = spec[1]
        offset = spec[2]
        next_js = list(range(next_j, next_j + m))
        if key not in component_dict:
            component_dict[key] = (m, offset)
            nonzero_j_vals += next_js
            beta += m
        else:
            zero_j_vals += next_js
        next_j += m
    j_vals = nonzero_j_vals + zero_j_vals
    i_vals = list(range(M))

    data = M * [1]
    D = sp.coo_array((data, (i_vals, j_vals)), shape=(M, M)).tocsr()
    D_inv = D.T
    gamma = M - beta
    return D, D_inv, gamma


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def get_ohe_spec_str(L, n_order, n_adj=None):
    """
    Function to get ohe_spec_str for any order and num adjacent
    :param L: Length of sequence (int > 0)
    :param n_order: Highest order of interaction (int >= 0)
    :param n_adj: Maximum span of interaction across positions (int >= 0)
    :return: ohe_spec_str: String specification of one-hot encoding
    """

    # If n_adj is not specified, assume use L
    if n_adj is None:
        n_adj = L

    # Create list of all iterables
    its = []
    for i in range(L - n_adj + 1):
        subset = range(i, i + n_adj)
        for m in range(n_order+1):
            p = list(itertools.combinations(subset,m))
            its.extend(p)
        #p = [s for s in powerset(subset) if len(s) <= n_order]
    its = sorted(set(its), key=lambda x: (len(x), *x))
    its = [[f'{i:d}' for i in it] for it in its]
    spec_seq = '.' + '+'.join(['x'.join(it) for it in its])

    return spec_seq

#TODO: Generalize everything over multiple alpha

def get_char_to_ohe_dict(alphabet):
    # Make sure all characters in alphabet are unique
    assert len(alphabet) == len(set(alphabet))
    alpha = len(alphabet)
    char_to_ohe_dict = {}
    for i, c in enumerate(alphabet):
        x = np.zeros(alpha)
        x[i] = 1
        char_to_ohe_dict[c] = x
    return char_to_ohe_dict


def get_char_to_sim_dict(alphabet):
    # Make sure all characters in alphabet are unique
    assert len(alphabet) == len(set(alphabet))
    alpha = len(alphabet)
    char_to_sim_dict = {}
    for i, c in enumerate(alphabet[:-1]):
        x = np.zeros(alpha - 1)
        x[i] = 1
        char_to_sim_dict[c] = x
    char_to_sim_dict[alphabet[-1]] = -np.ones(alpha - 1)
    return char_to_sim_dict


def get_single_position_T_and_T_inv(alpha=4):
    M_right = -1 * np.ones((alpha - 1, 1), dtype=np.int64)
    M_top = np.ones([1, alpha], dtype=np.int64)
    M_bulk = np.eye(alpha - 1, dtype=np.int64)
    M_bot = np.concatenate([M_bulk, M_right], axis=1)
    T = np.concatenate([M_top, M_bot], axis=0)

    M_bulk = alpha * np.eye(alpha - 1) - 1
    M_bot = -np.ones([1, alpha - 1])
    M_left = np.ones([alpha, 1])
    M_right = np.concatenate([M_bulk, M_bot], axis=0)
    T_inv = (1 / alpha) * np.concatenate([M_left, M_right], axis=1)

    return sp.csr_array(T), sp.csr_array(T_inv)