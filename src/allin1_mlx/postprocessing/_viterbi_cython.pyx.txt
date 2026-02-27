# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized Viterbi algorithm for DBN downbeat tracking.

This module provides a Cython-compiled version of the Viterbi loop,
aiming to match the performance of the original madmom Cython implementation.

Expected performance: ~13ms for 500 frames (matching original)
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport INFINITY

# NumPy array type definitions
ctypedef np.float32_t FLOAT32_t
ctypedef np.int32_t INT32_t


def viterbi_loop_cython(
    np.ndarray[FLOAT32_t, ndim=2] om_densities,
    np.ndarray[INT32_t, ndim=1] tm_states,
    np.ndarray[INT32_t, ndim=1] tm_pointers,
    np.ndarray[FLOAT32_t, ndim=1] tm_log_probs,
    np.ndarray[INT32_t, ndim=1] om_pointers,
    np.ndarray[FLOAT32_t, ndim=1] initial_dist_log
):
    """
    Cython-optimized Viterbi forward pass for HMM decoding.

    Args:
        om_densities: [num_observations, num_states] observation log-densities
        tm_states: Flattened predecessor state indices (CSR format)
        tm_pointers: Pointers into tm_states for each state (CSR format)
        tm_log_probs: Flattened transition log-probabilities (CSR format)
        om_pointers: Pointers into tm_states for observation model
        initial_dist_log: [num_states] initial state log-distribution

    Returns:
        bt_pointers: [num_observations, num_states] backtracking pointers
        current_viterbi: [num_states] final Viterbi scores
    """
    cdef int num_observations = om_densities.shape[0]
    cdef int num_states = len(initial_dist_log)  # Number of HMM states

    # Allocate output arrays
    cdef np.ndarray[INT32_t, ndim=2] bt_pointers = np.zeros(
        (num_observations, num_states), dtype=np.int32
    )
    cdef np.ndarray[FLOAT32_t, ndim=1] previous_viterbi = np.copy(initial_dist_log)
    cdef np.ndarray[FLOAT32_t, ndim=1] current_viterbi = np.empty(num_states, dtype=np.float32)

    # C-level variables for tight loops
    cdef int frame, state, i, start_idx, end_idx, num_prev
    cdef int prev_state, best_prev_state
    cdef FLOAT32_t density, val, best_val
    cdef FLOAT32_t* prev_viterbi_ptr = <FLOAT32_t*>previous_viterbi.data
    cdef FLOAT32_t* curr_viterbi_ptr = <FLOAT32_t*>current_viterbi.data
    cdef INT32_t* tm_states_ptr = <INT32_t*>tm_states.data
    cdef INT32_t* om_pointers_ptr = <INT32_t*>om_pointers.data
    cdef INT32_t* tm_pointers_ptr = <INT32_t*>tm_pointers.data
    cdef FLOAT32_t* tm_log_probs_ptr = <FLOAT32_t*>tm_log_probs.data
    cdef FLOAT32_t* om_densities_ptr = <FLOAT32_t*>om_densities.data
    cdef INT32_t* bt_pointers_ptr = <INT32_t*>bt_pointers.data

    # Main Viterbi loop
    for frame in range(num_observations):
        for state in range(num_states):
            # Get observation density for this state
            # om_densities has shape [num_observations, 3] (no_beat, beat, downbeat)
            # om_pointers[state] tells us which column to use
            density = om_densities_ptr[frame * 3 + om_pointers_ptr[state]]

            # Get predecessor states for this state (sparse CSR format)
            start_idx = tm_pointers_ptr[state]
            end_idx = tm_pointers_ptr[state + 1]
            num_prev = end_idx - start_idx

            if num_prev == 0:
                # No predecessors - should not happen in well-formed HMM
                curr_viterbi_ptr[state] = -INFINITY
                bt_pointers_ptr[frame * num_states + state] = -1
                continue

            # Find best predecessor
            best_val = -INFINITY
            best_prev_state = -1

            for i in range(num_prev):
                prev_state = tm_states_ptr[start_idx + i]
                val = (prev_viterbi_ptr[prev_state] +
                       tm_log_probs_ptr[start_idx + i] +
                       density)

                if val > best_val:
                    best_val = val
                    best_prev_state = prev_state  # Store actual state, not index

            # Store results
            curr_viterbi_ptr[state] = best_val
            bt_pointers_ptr[frame * num_states + state] = best_prev_state

        # Swap buffers for next iteration
        previous_viterbi, current_viterbi = current_viterbi, previous_viterbi
        prev_viterbi_ptr = <FLOAT32_t*>previous_viterbi.data
        curr_viterbi_ptr = <FLOAT32_t*>current_viterbi.data

    # After all frames, previous_viterbi contains the final scores
    return bt_pointers, previous_viterbi
