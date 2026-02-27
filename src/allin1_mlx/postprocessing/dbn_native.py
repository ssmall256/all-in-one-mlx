from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

try:
    import numba as _numba
except Exception:
    _numba = None

# Try to import Cython-optimized Viterbi
try:
    from ._viterbi_cython import viterbi_loop_cython
    _CYTHON_AVAILABLE = True
except ImportError:
    _CYTHON_AVAILABLE = False


def threshold_activations(activations: np.ndarray, threshold: float) -> Tuple[np.ndarray, int]:
    first = last = 0
    idx = np.nonzero(activations >= threshold)[0]
    if idx.size:
        first = max(first, int(idx.min()))
        last = min(len(activations), int(idx.max()) + 1)
    return activations[first:last], first


class BeatStateSpace:
    def __init__(self, min_interval: float, max_interval: float, num_intervals: int | None = None):
        intervals = np.arange(np.round(min_interval), np.round(max_interval) + 1)
        if num_intervals is not None and num_intervals < len(intervals):
            num_log_intervals = num_intervals
            intervals = []
            while len(intervals) < num_intervals:
                intervals = np.logspace(
                    np.log2(min_interval),
                    np.log2(max_interval),
                    num_log_intervals,
                    base=2,
                )
                intervals = np.unique(np.round(intervals))
                num_log_intervals += 1
        self.intervals = np.ascontiguousarray(intervals, dtype=int)
        self.num_states = int(np.sum(intervals))
        self.num_intervals = len(intervals)
        first_states = np.cumsum(np.r_[0, self.intervals[:-1]])
        self.first_states = first_states.astype(int)
        self.last_states = np.cumsum(self.intervals) - 1
        self.state_positions = np.empty(self.num_states)
        self.state_intervals = np.empty(self.num_states, dtype=int)
        idx = 0
        for interval in self.intervals:
            self.state_positions[idx : idx + interval] = np.linspace(0, 1, interval, endpoint=False)
            self.state_intervals[idx : idx + interval] = interval
            idx += interval


class BarStateSpace:
    def __init__(self, num_beats: int, min_interval: float, max_interval: float, num_intervals: int | None = None):
        self.num_beats = int(num_beats)
        self.state_positions = np.empty(0)
        self.state_intervals = np.empty(0, dtype=int)
        self.num_states = 0
        self.first_states = []
        self.last_states = []
        bss = BeatStateSpace(min_interval, max_interval, num_intervals)
        for beat in range(self.num_beats):
            self.state_positions = np.hstack((self.state_positions, bss.state_positions + beat))
            self.state_intervals = np.hstack((self.state_intervals, bss.state_intervals))
            self.first_states.append(bss.first_states + self.num_states)
            self.last_states.append(bss.last_states + self.num_states)
            self.num_states += bss.num_states


def exponential_transition(
    from_intervals: np.ndarray,
    to_intervals: np.ndarray,
    transition_lambda: float | None,
    threshold: float = np.spacing(1),
    norm: bool = True,
) -> np.ndarray:
    if transition_lambda is None:
        return np.diag(np.diag(np.ones((len(from_intervals), len(to_intervals)))))
    ratio = to_intervals.astype(float) / from_intervals.astype(float)[:, np.newaxis]
    prob = np.exp(-transition_lambda * np.abs(ratio - 1.0))
    prob[prob <= threshold] = 0.0
    if norm:
        prob_sum = np.sum(prob, axis=1)[:, np.newaxis]
        prob_sum[prob_sum == 0] = 1.0
        prob /= prob_sum
    return prob


@dataclass
class TransitionModel:
    state_space: object
    pointers: np.ndarray
    prev_states: np.ndarray
    probabilities: np.ndarray

    @staticmethod
    def make_sparse(
        num_states: int,
        states: Iterable[int],
        prev_states: Iterable[int],
        probabilities: Iterable[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pairs: dict[tuple[int, int], float] = {}
        for s, p, prob in zip(states, prev_states, probabilities):
            key = (int(s), int(p))
            pairs[key] = pairs.get(key, 0.0) + float(prob)
        if not pairs:
            raise ValueError("No transitions found.")
        states_arr = np.fromiter((k[0] for k in pairs.keys()), dtype=int)
        prev_arr = np.fromiter((k[1] for k in pairs.keys()), dtype=int)
        prob_arr = np.fromiter(pairs.values(), dtype=float)

        order = np.lexsort((prev_arr, states_arr))
        states_arr = states_arr[order]
        prev_arr = prev_arr[order]
        prob_arr = prob_arr[order]

        counts = np.bincount(states_arr, minlength=num_states)
        pointers = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
        return pointers, prev_arr.astype(np.int32), prob_arr.astype(np.float32)


class BarTransitionModel(TransitionModel):
    def __init__(self, state_space: BarStateSpace, transition_lambda: float | list[float]):
        if not isinstance(transition_lambda, list):
            transition_lambda = [transition_lambda] * state_space.num_beats
        if state_space.num_beats != len(transition_lambda):
            raise ValueError("transition_lambda length must match num_beats")
        states = np.arange(state_space.num_states, dtype=np.uint32)
        states = np.setdiff1d(states, state_space.first_states)
        prev_states = states - 1
        probabilities = np.ones_like(states, dtype=float)

        for beat in range(state_space.num_beats):
            to_states = state_space.first_states[beat]
            from_states = state_space.last_states[beat - 1]
            from_int = state_space.state_intervals[from_states]
            to_int = state_space.state_intervals[to_states]
            prob = exponential_transition(from_int, to_int, transition_lambda[beat])
            from_prob, to_prob = np.nonzero(prob)
            states = np.hstack((states, to_states[to_prob]))
            prev_states = np.hstack((prev_states, from_states[from_prob]))
            probabilities = np.hstack((probabilities, prob[prob != 0]))

        pointers, prev_states, probabilities = TransitionModel.make_sparse(
            state_space.num_states, states, prev_states, probabilities
        )
        super().__init__(state_space=state_space, pointers=pointers, prev_states=prev_states, probabilities=probabilities)


@dataclass
class ObservationModel:
    pointers: np.ndarray

    def log_densities(self, observations: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RNNDownBeatTrackingObservationModel(ObservationModel):
    def __init__(self, state_space: BarStateSpace, observation_lambda: int):
        self.observation_lambda = observation_lambda
        pointers = np.zeros(state_space.num_states, dtype=np.uint32)
        border = 1.0 / observation_lambda
        pointers[state_space.state_positions % 1 < border] = 1
        pointers[state_space.state_positions < border] = 2
        super().__init__(pointers)

    def log_densities(self, observations: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations, dtype=np.float32)
        eps = np.float32(np.spacing(1))
        log_densities = np.empty((len(obs), 3), dtype=np.float32)
        no_beat = np.float32(1.0) - np.sum(obs, axis=1)
        log_densities[:, 0] = np.log(np.maximum(no_beat / (self.observation_lambda - 1), eps))
        log_densities[:, 1] = np.log(np.maximum(obs[:, 0], eps))
        log_densities[:, 2] = np.log(np.maximum(obs[:, 1], eps))
        return log_densities


class HiddenMarkovModel:
    def __init__(self, transition_model: TransitionModel, observation_model: ObservationModel, initial: np.ndarray | None):
        self.transition_model = transition_model
        self.observation_model = observation_model
        if initial is None:
            num_states = transition_model.state_space.num_states
            self.log_initial = np.full(num_states, -np.log(num_states), dtype=np.float32)
        else:
            self.log_initial = np.log(np.asarray(initial, dtype=np.float32))
        self._pointers = transition_model.pointers.astype(np.int32, copy=False)
        self._prev_states = transition_model.prev_states.astype(np.int32, copy=False)
        self._obs_ptr = observation_model.pointers.astype(np.int32, copy=False)
        self._log_trans = np.log(
            np.maximum(transition_model.probabilities.astype(np.float32), np.spacing(1))
        ).astype(np.float32)

        # Precompute transition structure for vectorized Viterbi.
        # Split states into single-incoming (bulk vectorized) vs
        # multi-incoming (small loop over beat-boundary states only).
        edge_counts = np.diff(self._pointers)
        self._has_incoming = edge_counts > 0
        single = edge_counts == 1
        multi = edge_counts > 1
        self._single_states = np.where(single)[0].astype(np.int32)
        self._multi_states = np.where(multi)[0].astype(np.int32)
        # For single-transition states: the one incoming edge index
        self._single_edges = self._pointers[self._single_states].astype(np.int32)
        # For single-transition states: the one previous state
        self._single_prev = self._prev_states[self._single_edges]
        # For single-transition states: the log transition prob
        self._single_log_trans = self._log_trans[self._single_edges]
        # For multi-transition states: precompute (start, end) pairs
        self._multi_starts = self._pointers[self._multi_states]
        self._multi_ends = self._pointers[self._multi_states + 1]

    def viterbi(self, observations: np.ndarray) -> Tuple[np.ndarray, float]:
        obs = np.asarray(observations, dtype=np.float32)
        log_obs = self.observation_model.log_densities(obs).astype(np.float32)
        obs_ptr = self._obs_ptr
        num_states = int(self.transition_model.state_space.num_states)

        # Try Cython first (fastest)
        if _CYTHON_AVAILABLE:
            bt_pointers, final_viterbi = viterbi_loop_cython(
                log_obs,
                self._prev_states,
                self._pointers,
                self._log_trans,
                obs_ptr,
                self.log_initial,
            )
            # Backtrack to find best path
            last_state = int(np.argmax(final_viterbi))
            path = np.empty(len(obs), dtype=np.int32)
            path[-1] = last_state
            for t in range(len(obs) - 2, -1, -1):
                path[t] = bt_pointers[t + 1, path[t + 1]]
            log_prob = float(final_viterbi[last_state])
            return path.astype(int), log_prob

        # Fall back to Numba if available
        if _numba is not None:
            path, log_prob = _viterbi_decode(
                log_obs,
                obs_ptr,
                self._pointers,
                self._prev_states,
                self._log_trans,
                self.log_initial,
            )
            return path.astype(int), float(log_prob)

        return self._viterbi_vectorized(log_obs, obs_ptr, num_states)

    def _viterbi_vectorized(
        self,
        log_obs: np.ndarray,
        obs_ptr: np.ndarray,
        num_states: int,
    ) -> Tuple[np.ndarray, float]:
        """Vectorized Viterbi: only the time loop is in Python.

        The inner state loop is replaced by bulk numpy operations:
        - Single-incoming states (vast majority): fully vectorized
        - Multi-incoming states (beat boundaries): small loop (~60 per meter)
        """
        num_frames = len(log_obs)
        prev_states = self._prev_states
        log_trans = self._log_trans
        single_states = self._single_states
        single_prev = self._single_prev
        single_log_trans = self._single_log_trans
        multi_states = self._multi_states
        multi_starts = self._multi_starts
        multi_ends = self._multi_ends

        # Only store current and previous delta to save memory.
        # Backtrack pointers still need full (T, S) storage.
        delta_prev = self.log_initial + log_obs[0, obs_ptr]
        psi = np.empty((num_frames, num_states), dtype=np.int32)

        for t in range(1, num_frames):
            delta_cur = np.full(num_states, -np.inf, dtype=np.float32)

            # --- Bulk: single-incoming states (vectorized) ---
            delta_cur[single_states] = (
                delta_prev[single_prev] + single_log_trans + log_obs[t, obs_ptr[single_states]]
            )
            psi[t, single_states] = single_prev

            # --- Small loop: multi-incoming states (beat boundaries) ---
            for i in range(len(multi_states)):
                state = multi_states[i]
                s, e = multi_starts[i], multi_ends[i]
                edge_prev = prev_states[s:e]
                scores = delta_prev[edge_prev] + log_trans[s:e]
                best = int(np.argmax(scores))
                delta_cur[state] = scores[best] + log_obs[t, obs_ptr[state]]
                psi[t, state] = edge_prev[best]

            delta_prev = delta_cur

        # Backtrack
        last_state = int(np.argmax(delta_prev))
        path = np.empty(num_frames, dtype=np.int32)
        path[-1] = last_state
        for t in range(num_frames - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]
        log_prob = float(delta_prev[last_state])
        return path.astype(int), log_prob


if _numba is not None:

    @_numba.njit(cache=True)
    def _viterbi_decode(
        log_obs: np.ndarray,
        obs_ptr: np.ndarray,
        pointers: np.ndarray,
        prev_states: np.ndarray,
        log_trans: np.ndarray,
        log_initial: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        num_frames = log_obs.shape[0]
        num_states = obs_ptr.shape[0]
        delta = np.full((num_frames, num_states), -np.inf, dtype=np.float32)
        psi = np.full((num_frames, num_states), -1, dtype=np.int32)

        for state in range(num_states):
            delta[0, state] = log_initial[state] + log_obs[0, obs_ptr[state]]

        for t in range(1, num_frames):
            for state in range(num_states):
                start = pointers[state]
                end = pointers[state + 1]
                if start == end:
                    continue
                best_prev = prev_states[start]
                best_score = delta[t - 1, best_prev] + log_trans[start]
                for idx in range(start + 1, end):
                    prev = prev_states[idx]
                    score = delta[t - 1, prev] + log_trans[idx]
                    if score > best_score:
                        best_score = score
                        best_prev = prev
                delta[t, state] = best_score + log_obs[t, obs_ptr[state]]
                psi[t, state] = best_prev

        last_state = 0
        best_last = delta[num_frames - 1, 0]
        for state in range(1, num_states):
            score = delta[num_frames - 1, state]
            if score > best_last:
                best_last = score
                last_state = state

        path = np.empty(num_frames, dtype=np.int32)
        path[num_frames - 1] = last_state
        for t in range(num_frames - 2, -1, -1):
            path[t] = psi[t + 1, path[t + 1]]

        return path, float(best_last)


class DBNDownBeatTrackingProcessor:
    MIN_BPM = 55.0
    MAX_BPM = 215.0
    NUM_TEMPI = 60
    TRANSITION_LAMBDA = 100
    OBSERVATION_LAMBDA = 16
    THRESHOLD = 0.05
    CORRECT = True

    def __init__(
        self,
        beats_per_bar: Iterable[int],
        min_bpm: Iterable[float] | float = MIN_BPM,
        max_bpm: Iterable[float] | float = MAX_BPM,
        num_tempi: Iterable[int] | int = NUM_TEMPI,
        transition_lambda: Iterable[float] | float = TRANSITION_LAMBDA,
        observation_lambda: int = OBSERVATION_LAMBDA,
        threshold: float = THRESHOLD,
        correct: bool = CORRECT,
        fps: float | None = None,
    ):
        beats_per_bar = np.array(beats_per_bar, ndmin=1)
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        transition_lambda = np.array(transition_lambda, ndmin=1)
        if len(min_bpm) != len(beats_per_bar):
            min_bpm = np.repeat(min_bpm, len(beats_per_bar))
        if len(max_bpm) != len(beats_per_bar):
            max_bpm = np.repeat(max_bpm, len(beats_per_bar))
        if len(num_tempi) != len(beats_per_bar):
            num_tempi = np.repeat(num_tempi, len(beats_per_bar))
        if len(transition_lambda) != len(beats_per_bar):
            transition_lambda = np.repeat(transition_lambda, len(beats_per_bar))
        if not (len(min_bpm) == len(max_bpm) == len(num_tempi) == len(beats_per_bar) == len(transition_lambda)):
            raise ValueError("Parameter lengths must match beats_per_bar length.")

        if fps is None:
            raise ValueError("fps must be provided.")

        self.hmms = []
        for i, beats in enumerate(beats_per_bar):
            min_interval = 60.0 * fps / max_bpm[i]
            max_interval = 60.0 * fps / min_bpm[i]
            st = BarStateSpace(beats, min_interval, max_interval, int(num_tempi[i]))
            tm = BarTransitionModel(st, float(transition_lambda[i]))
            om = RNNDownBeatTrackingObservationModel(st, observation_lambda)
            self.hmms.append(HiddenMarkovModel(tm, om, None))

        self.beats_per_bar = beats_per_bar
        self.threshold = threshold
        self.correct = correct
        self.fps = fps

    def process(self, activations: np.ndarray) -> np.ndarray:
        first = 0
        if self.threshold:
            activations, first = threshold_activations(activations, self.threshold)
        if not activations.any():
            return np.empty((0, 2))

        results = [hmm.viterbi(activations) for hmm in self.hmms]
        best = int(np.argmax([r[1] for r in results]))
        path, _ = results[best]

        st = self.hmms[best].transition_model.state_space
        om = self.hmms[best].observation_model
        positions = st.state_positions[path]
        beat_numbers = positions.astype(int) + 1

        if self.correct:
            beats = np.empty(0, dtype=int)
            beat_range = om.pointers[path] >= 1
            if not beat_range.any():
                return np.empty((0, 2))
            idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
            if beat_range[0]:
                idx = np.r_[0, idx]
            if beat_range[-1]:
                idx = np.r_[idx, beat_range.size]
            if idx.any():
                for left, right in idx.reshape((-1, 2)):
                    peak = np.argmax(activations[left:right]) // 2 + left
                    beats = np.hstack((beats, peak))
        else:
            beats = np.nonzero(np.diff(beat_numbers))[0] + 1

        return np.vstack(((beats + first) / float(self.fps), beat_numbers[beats])).T

    def __call__(self, activations: np.ndarray) -> np.ndarray:
        """Make the processor callable to match madmom's interface."""
        return self.process(activations)
