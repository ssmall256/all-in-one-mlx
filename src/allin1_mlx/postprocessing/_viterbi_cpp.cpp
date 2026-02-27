#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <limits>
#include <vector>

namespace py = pybind11;

static py::tuple viterbi_loop_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> log_obs,
    py::array_t<int, py::array::c_style | py::array::forcecast> tm_states,
    py::array_t<int, py::array::c_style | py::array::forcecast> tm_pointers,
    py::array_t<float, py::array::c_style | py::array::forcecast> tm_log_probs,
    py::array_t<int, py::array::c_style | py::array::forcecast> om_pointers,
    py::array_t<float, py::array::c_style | py::array::forcecast> initial_dist_log
) {
  auto log_obs_buf = log_obs.request();
  auto tm_states_buf = tm_states.request();
  auto tm_pointers_buf = tm_pointers.request();
  auto tm_log_probs_buf = tm_log_probs.request();
  auto om_pointers_buf = om_pointers.request();
  auto initial_buf = initial_dist_log.request();

  if (log_obs_buf.ndim != 2) {
    throw std::runtime_error("log_obs must be 2D");
  }
  if (tm_pointers_buf.ndim != 1 || tm_states_buf.ndim != 1 || tm_log_probs_buf.ndim != 1 ||
      om_pointers_buf.ndim != 1 || initial_buf.ndim != 1) {
    throw std::runtime_error("tm_* and om_pointers/initial must be 1D");
  }

  const int64_t num_obs = log_obs_buf.shape[0];
  const int64_t num_states = om_pointers_buf.shape[0];
  const int64_t obs_stride0 = log_obs_buf.strides[0] / sizeof(float);
  const int64_t obs_stride1 = log_obs_buf.strides[1] / sizeof(float);

  const float *log_obs_ptr = static_cast<float *>(log_obs_buf.ptr);
  const int *tm_states_ptr = static_cast<int *>(tm_states_buf.ptr);
  const int *tm_pointers_ptr = static_cast<int *>(tm_pointers_buf.ptr);
  const float *tm_log_probs_ptr = static_cast<float *>(tm_log_probs_buf.ptr);
  const int *om_pointers_ptr = static_cast<int *>(om_pointers_buf.ptr);
  const float *initial_ptr = static_cast<float *>(initial_buf.ptr);

  auto bt_pointers = py::array_t<int>({
      static_cast<py::ssize_t>(num_obs),
      static_cast<py::ssize_t>(num_states),
  });
  auto final_viterbi = py::array_t<float>(
      py::array::ShapeContainer({static_cast<py::ssize_t>(num_states)})
  );
  auto bt_buf = bt_pointers.request();
  auto final_buf = final_viterbi.request();
  int *bt_ptr = static_cast<int *>(bt_buf.ptr);
  float *final_ptr = static_cast<float *>(final_buf.ptr);

  std::vector<float> prev_viterbi(num_states, -std::numeric_limits<float>::infinity());
  std::vector<float> curr_viterbi(num_states, -std::numeric_limits<float>::infinity());

  for (int64_t state = 0; state < num_states; ++state) {
    const int obs_idx = om_pointers_ptr[state];
    const float obs_val = log_obs_ptr[0 * obs_stride0 + obs_idx * obs_stride1];
    prev_viterbi[state] = initial_ptr[state] + obs_val;
    bt_ptr[state] = 0;
  }

  for (int64_t t = 1; t < num_obs; ++t) {
    const int bt_row_offset = static_cast<int>(t * num_states);
    for (int64_t state = 0; state < num_states; ++state) {
      const int start = tm_pointers_ptr[state];
      const int end = tm_pointers_ptr[state + 1];
      if (start == end) {
        curr_viterbi[state] = -std::numeric_limits<float>::infinity();
        bt_ptr[bt_row_offset + state] = 0;
        continue;
      }
      int best_prev = tm_states_ptr[start];
      float best_val = prev_viterbi[best_prev] + tm_log_probs_ptr[start];
      for (int idx = start + 1; idx < end; ++idx) {
        const int prev = tm_states_ptr[idx];
        const float val = prev_viterbi[prev] + tm_log_probs_ptr[idx];
        if (val > best_val) {
          best_val = val;
          best_prev = prev;
        }
      }
      const int obs_idx = om_pointers_ptr[state];
      const float obs_val = log_obs_ptr[t * obs_stride0 + obs_idx * obs_stride1];
      curr_viterbi[state] = best_val + obs_val;
      bt_ptr[bt_row_offset + state] = best_prev;
    }
    prev_viterbi.swap(curr_viterbi);
  }

  for (int64_t state = 0; state < num_states; ++state) {
    final_ptr[state] = prev_viterbi[state];
  }

  return py::make_tuple(bt_pointers, final_viterbi);
}

static py::tuple viterbi_decode_obs_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> observations,
    py::array_t<int, py::array::c_style | py::array::forcecast> tm_states,
    py::array_t<int, py::array::c_style | py::array::forcecast> tm_pointers,
    py::array_t<float, py::array::c_style | py::array::forcecast> tm_log_probs,
    py::array_t<int, py::array::c_style | py::array::forcecast> om_pointers,
    py::array_t<float, py::array::c_style | py::array::forcecast> initial_dist_log,
    int observation_lambda
) {
  auto obs_buf = observations.request();
  auto tm_states_buf = tm_states.request();
  auto tm_pointers_buf = tm_pointers.request();
  auto tm_log_probs_buf = tm_log_probs.request();
  auto om_pointers_buf = om_pointers.request();
  auto initial_buf = initial_dist_log.request();

  if (obs_buf.ndim != 2 || obs_buf.shape[1] != 2) {
    throw std::runtime_error("observations must be 2D with shape (T, 2)");
  }
  if (tm_pointers_buf.ndim != 1 || tm_states_buf.ndim != 1 || tm_log_probs_buf.ndim != 1 ||
      om_pointers_buf.ndim != 1 || initial_buf.ndim != 1) {
    throw std::runtime_error("tm_* and om_pointers/initial must be 1D");
  }
  if (observation_lambda <= 1) {
    throw std::runtime_error("observation_lambda must be > 1");
  }

  const int64_t num_obs = obs_buf.shape[0];
  const int64_t num_states = om_pointers_buf.shape[0];
  const int64_t obs_stride0 = obs_buf.strides[0] / sizeof(float);
  const int64_t obs_stride1 = obs_buf.strides[1] / sizeof(float);

  const float *obs_ptr = static_cast<float *>(obs_buf.ptr);
  const int *tm_states_ptr = static_cast<int *>(tm_states_buf.ptr);
  const int *tm_pointers_ptr = static_cast<int *>(tm_pointers_buf.ptr);
  const float *tm_log_probs_ptr = static_cast<float *>(tm_log_probs_buf.ptr);
  const int *om_pointers_ptr = static_cast<int *>(om_pointers_buf.ptr);
  const float *initial_ptr = static_cast<float *>(initial_buf.ptr);

  std::vector<int> bt_pointers(static_cast<size_t>(num_obs * num_states), 0);
  std::vector<float> prev_viterbi(num_states, -std::numeric_limits<float>::infinity());
  std::vector<float> curr_viterbi(num_states, -std::numeric_limits<float>::infinity());

  const float eps = std::numeric_limits<float>::epsilon();
  const float obs_div = 1.0f / static_cast<float>(observation_lambda - 1);
  std::vector<float> log0(static_cast<size_t>(num_obs));
  std::vector<float> log1(static_cast<size_t>(num_obs));
  std::vector<float> log2(static_cast<size_t>(num_obs));
  for (int64_t t = 0; t < num_obs; ++t) {
    const float o0 = obs_ptr[t * obs_stride0 + 0 * obs_stride1];
    const float o1 = obs_ptr[t * obs_stride0 + 1 * obs_stride1];
    const float no_beat = 1.0f - (o0 + o1);
    log0[t] = std::log(std::max(no_beat * obs_div, eps));
    log1[t] = std::log(std::max(o0, eps));
    log2[t] = std::log(std::max(o1, eps));
  }

  for (int64_t state = 0; state < num_states; ++state) {
    const int obs_idx = om_pointers_ptr[state];
    const float obs_val = obs_idx == 0 ? log0[0] : (obs_idx == 1 ? log1[0] : log2[0]);
    prev_viterbi[state] = initial_ptr[state] + obs_val;
    bt_pointers[state] = 0;
  }

  for (int64_t t = 1; t < num_obs; ++t) {
    const int bt_row_offset = static_cast<int>(t * num_states);
    for (int64_t state = 0; state < num_states; ++state) {
      const int start = tm_pointers_ptr[state];
      const int end = tm_pointers_ptr[state + 1];
      if (start == end) {
        curr_viterbi[state] = -std::numeric_limits<float>::infinity();
        bt_pointers[bt_row_offset + state] = 0;
        continue;
      }
      int best_prev = tm_states_ptr[start];
      float best_val = prev_viterbi[best_prev] + tm_log_probs_ptr[start];
      for (int idx = start + 1; idx < end; ++idx) {
        const int prev = tm_states_ptr[idx];
        const float val = prev_viterbi[prev] + tm_log_probs_ptr[idx];
        if (val > best_val) {
          best_val = val;
          best_prev = prev;
        }
      }
      const int obs_idx = om_pointers_ptr[state];
      const float obs_val = obs_idx == 0 ? log0[t] : (obs_idx == 1 ? log1[t] : log2[t]);
      curr_viterbi[state] = best_val + obs_val;
      bt_pointers[bt_row_offset + state] = best_prev;
    }
    prev_viterbi.swap(curr_viterbi);
  }

  int64_t last_state = 0;
  float best_last = prev_viterbi[0];
  for (int64_t state = 1; state < num_states; ++state) {
    const float val = prev_viterbi[state];
    if (val > best_last) {
      best_last = val;
      last_state = state;
    }
  }

  auto path = py::array_t<int>(
      py::array::ShapeContainer({static_cast<py::ssize_t>(num_obs)})
  );
  auto path_buf = path.request();
  int *path_ptr = static_cast<int *>(path_buf.ptr);
  path_ptr[num_obs - 1] = static_cast<int>(last_state);
  for (int64_t t = num_obs - 2; t >= 0; --t) {
    const int next_state = path_ptr[t + 1];
    path_ptr[t] = bt_pointers[(t + 1) * num_states + next_state];
  }

  return py::make_tuple(path, best_last);
}

PYBIND11_MODULE(_viterbi_cpp, m) {
  m.doc() = "C++ Viterbi loop for DBN postprocessing";
  m.def("viterbi_loop_cpp", &viterbi_loop_cpp, "Viterbi forward pass (C++)");
  m.def("viterbi_decode_obs_cpp", &viterbi_decode_obs_cpp, "Viterbi decode with observation model (C++)");
}
