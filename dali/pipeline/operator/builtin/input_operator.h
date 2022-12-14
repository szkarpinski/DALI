// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_INPUT_OPERATOR_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_INPUT_OPERATOR_H_

#include <list>
#include <memory>
#include <vector>
#include "dali/core/common.h"
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/operators/input/caching_list.h"
#include "dali/pipeline/operator/batch_size_provider.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/worker_thread.h"

namespace dali {

namespace detail {

struct CudaEventWrapper : CUDAEvent {
  CudaEventWrapper() : CUDAEvent(CUDAEvent::Create()) {}
};

}  // namespace detail


struct InputSourceState {
  /**
   * True if the data that was shared as no_copy required copy regardless.
   * Happens for non-contiguous TensorList with GPU memory.
   */
  bool copied_shared_data = false;

  /**
   * @brief Actual value of no_copy option used in this call. Always false for CopyUserData(...)
   * and always true for ShareUserData(...)
   */
  bool no_copy = false;
};


/**
 * @brief Option used to override the InputOperator ``no_copy`` parameter
 *
 * It allows to:
 *  * DEFAULT - leave the default (the ``no_copy`` parameter is used),
 *  * FORCE_COPY - always make a copy,
 *  * FORCE_NO_COPY - always share the data without copy.
 */
enum class InputOperatorNoCopyMode {
  DEFAULT,
  FORCE_COPY,
  FORCE_NO_COPY
};


/**
 * @brief Options that can be configured when setting data for the External Source
 */
struct InputOperatorSettingMode {
  /**
   * @brief If SetExternalInputHelper should be blocking - waits until provided data is copied
   *        to the internal buffer
   */
  bool sync = false;

  /**
   * @brief If true, a copy kernel will be used to make a contiguous buffer instead of
   *  cudaMemcpyAsync.
   */
  bool use_copy_kernel = false;

  /**
   * @brief Select whether to use the parameter defined in the External Source or
   *  override the mode of operation forcing the copy or no-copy
   */
  InputOperatorNoCopyMode no_copy_mode = InputOperatorNoCopyMode::DEFAULT;
};


template<typename Backend>
class InputOperator : public Operator<Backend>, virtual public BatchSizeProvider {
  using uptr_tl_type = std::unique_ptr<TensorList<Backend>>;
  using uptr_cuda_event_type = std::unique_ptr<detail::CudaEventWrapper>;

 public:
  explicit InputOperator(const OpSpec &spec) :
          Operator<Backend>(spec),
          device_id_(spec.GetArgument<int>("device_id")),
          blocking_(spec.GetArgument<bool>("blocking")),
          no_copy_(spec.GetArgument<bool>("no_copy")),
          sync_worker_(device_id_, false, "InputOperator sync_worker_") {
    if (std::is_same<Backend, GPUBackend>::value) {
      internal_copy_stream_ = CUDAStreamPool::instance().Get(device_id_);
      internal_copy_order_ = internal_copy_stream_;
    }
    sync_worker_.WaitForInit();
  }


  virtual ~InputOperator() {
    sync_worker_.ForceStop();
    sync_worker_.Shutdown();
  }

  DISABLE_COPY_MOVE_ASSIGN(InputOperator);


  /**
   * Template method is used, since the data availability (i.e. block or fail)
   * shall be handled in the same way in every InputOperator.
   */
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) final {
    HandleDataAvailability();
    return SetupImplDerived(output_desc, ws);
  }


  /**
   * @brief Sets the data that should be passed out of the op on the next iteration.
   */
  template<typename SrcBackend>
  void SetDataSource(const vector<Tensor<SrcBackend>> &vect_of_tensors, AccessOrder order = {},
                     InputOperatorSettingMode ext_src_setting_mode = {}) {
    DeviceGuard g(device_id_);
    DomainTimeRange tr("[DALI][InputOperator] SetDataSource", DomainTimeRange::kViolet);
    DALI_ENFORCE(vect_of_tensors.size() > 0, "Provided batch cannot be empty.");
    TensorList<SrcBackend> tl(vect_of_tensors.size());
    tl.SetupLike(vect_of_tensors[0]);
    for (int i = 0; i < tl.num_samples(); ++i) {
      tl.SetSample(i, const_cast<Tensor<SrcBackend> &>(vect_of_tensors[i]));
    }
    SetDataSourceHelper(tl, order, ext_src_setting_mode);
  }


  /**
   * @brief Sets the data that should be passed out of the op on the next iteration.
   */
  template<typename SrcBackend>
  void SetDataSource(const TensorList<SrcBackend> &tl, AccessOrder order = {},
                     InputOperatorSettingMode ext_src_setting_mode = {}) {
    DeviceGuard g(device_id_);
    DomainTimeRange tr("[DALI][InputOperator] SetDataSource", DomainTimeRange::kViolet);
    SetDataSourceHelper(tl, order, ext_src_setting_mode);
  }


 protected:
  virtual bool SetupImplDerived(std::vector<OutputDesc> &output_desc, const Workspace &ws) = 0;


  /**
   * Peeks the data that is next in line.
   */
  const TensorList<Backend> &PeekCurrentData() {
    return *tl_data_.PeekFront();
  }


  int NextBatchSize() override {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    return tl_data_.PeekProphet()->num_samples();
  }


  void Advance() override {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    tl_data_.AdvanceProphet();
  }

  ///@{
  /**
   * Injects current data portion into the provided TensorList and recycles
   * the inner container for the data.
   *
   * This function will take a best effort not to copy the data,
   * however it might not always be possible.
   *
   * @param target Where the data shall be injected.
   * @param tp TheadPool used to copy the data.
   */
  void ForwardCurrentData(TensorList<CPUBackend> &target, ThreadPool &tp);

  void ForwardCurrentData(TensorList<GPUBackend> &target, cudaStream_t stream = nullptr);
  ///@}


  int device_id_;
  bool blocking_ = true;
  bool no_copy_ = false;


 private:
  /**
   * Checks if the data is available.
   * If it's not, either blocks or throws an error, depending on ``blocking``
   * operator argument.
   */
  void HandleDataAvailability() {
    std::unique_lock<std::mutex> busy_lock(busy_m_);
    if (blocking_) {
      cv_.wait(busy_lock, [&data = state_] { return !data.empty(); });
    } else {
      if (state_.empty()) {
        DALI_FAIL("No data was provided to the ExternalSource. Make sure to feed it properly.");
      }
    }
  }


  // pass cuda_event by pointer to allow default, nullptr value, with the
  // reference it is not that easy
  template<typename DataType>
  void RecycleBuffer(DataType &data,
                     std::list<uptr_cuda_event_type> *cuda_event = nullptr,
                     std::list<uptr_cuda_event_type> *copy_to_gpu = nullptr) {
    // No need to synchronize on copy_to_gpu - it was already synchronized before
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    tl_data_.Recycle(data);
    if (copy_to_gpu) {
      copy_to_storage_events_.Recycle(*copy_to_gpu);
    }
  }


  template<typename SrcBackend>
  std::enable_if_t<!std::is_same<SrcBackend, Backend>::value>
  ShareUserData(const TensorList<SrcBackend> &t, AccessOrder /* order = {}*/,
                bool /* use_copy_kernel */) {
    DALI_FAIL(make_string("no_copy is supported only for the same data source device type "
                          "as operator. Received: ",
                          std::is_same<SrcBackend, CPUBackend>::value ? "CPU" : "GPU",
                          " input for ",
                          std::is_same<Backend, CPUBackend>::value ? "CPU" : "GPU",
                          " operator."));
  }


  template<typename SrcBackend>
  std::enable_if_t<
          std::is_same<SrcBackend, Backend>::value && std::is_same<SrcBackend, CPUBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &batch, AccessOrder /* order = {}*/,
                bool /*use_copy_kernel = false*/) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    state_.push_back({false, true});
    auto tl_elm = GetEmptyOutputBatch();
    // set pinned if needed
    if (batch.is_pinned() != tl_elm.front()->is_pinned()) {
      tl_elm.front()->Reset();
      tl_elm.front()->set_pinned(batch.is_pinned());
    }
    tl_elm.front()->ShareData(const_cast<TensorList<CPUBackend> &>(batch));
    tl_data_.PushBack(tl_elm);
  }


  /**
   * @brief Attempts to share data from tensor vector to tensor list without
   *        an additional copy if the batch is contiguous.
   *        In case of scattered samples, the data is copied to a contiguous
   *        buffer.
   * @remarks Mixing contiguous and non-contiguous inputs in subsequents calls
   *        is not supported and could lead to data corruption.
   * @param batch source data
   * @param order CUDA stream use to schedule the copy (or host order to make the copy
   *              host-syncrhonous)
   * @param use_copy_kernel If true, a copy kernel will be used to make a
   *        contiguous buffer instead of cudaMemcpyAsync.
   */
  template<typename SrcBackend>
  std::enable_if_t<
          std::is_same<SrcBackend, Backend>::value && std::is_same<SrcBackend, GPUBackend>::value>
  ShareUserData(const TensorList<SrcBackend> &batch, AccessOrder order = {},
                bool use_copy_kernel = false) {
    std::lock_guard<std::mutex> busy_lock(busy_m_);
    auto tl_elm = GetEmptyOutputBatch();
    bool copied_shared_data = false;

    if (batch.IsContiguous()) {
      auto batch_owner = unsafe_sample_owner(const_cast<TensorList<SrcBackend> &>(batch), 0);
      tl_elm.front()->ShareData(batch_owner, batch.nbytes(), batch.is_pinned(), batch.shape(),
                                batch.type(), batch.device_id(), batch.order());
      zero_copy_noncontiguous_gpu_input_ = true;
    } else {
      // it is not contiguous so we need to copy
      tl_elm.front()->Copy(batch, order, use_copy_kernel);
      std::list<uptr_cuda_event_type> copy_to_storage_event;
      copy_to_storage_event = copy_to_storage_events_.GetEmpty();
      CUDA_CALL(cudaEventRecord(*copy_to_storage_event.front(), order.stream()));
      copy_to_storage_events_.PushBack(copy_to_storage_event);

      if (zero_copy_noncontiguous_gpu_input_) {
        DALI_WARN("ExternalSource operator should not mix contiguous and noncontiguous inputs. "
                  "In such a case the internal memory used to gather data in a contiguous chunk "
                  "of memory would be trashed.");
      }
      copied_shared_data = true;
    }
    state_.push_back({copied_shared_data, true});
    tl_data_.PushBack(tl_elm);
  }


  template<typename SrcBackend, typename B = Backend>
  std::enable_if_t<std::is_same<B, CPUBackend>::value>
  CopyUserData(const TensorList<SrcBackend> &batch, AccessOrder order, bool /* sync */,
               bool /* use_copy_kernel */) {
    std::list<uptr_tl_type> tl_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm = GetEmptyOutputBatch();
    }
    // set pinned if needed
    tl_elm.front()->set_order(AccessOrder::host());
    if (batch.is_pinned() != tl_elm.front()->is_pinned()) {
      tl_elm.front()->Reset();
      tl_elm.front()->set_pinned(batch.is_pinned());
    }
    AccessOrder copy_order =
            std::is_same<SrcBackend, CPUBackend>::value
            ? AccessOrder::host()  // do not use a device order for a host to host copy
            : order;
    tl_elm.front()->Copy(batch, copy_order);
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_data_.PushBack(tl_elm);
      state_.push_back({false, false});
    }
  }


  template<typename SrcBackend, typename B = Backend>
  std::enable_if_t<std::is_same<B, GPUBackend>::value>
  CopyUserData(const TensorList<SrcBackend> &batch, AccessOrder order, bool sync,
               bool use_copy_kernel) {
    std::list<uptr_cuda_event_type> copy_to_storage_event;
    std::list<uptr_tl_type> tl_elm;
    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_elm = GetEmptyOutputBatch();
      copy_to_storage_event = copy_to_storage_events_.GetEmpty();
    }
    // If we got a host order we most probably got it via FeedPipeline and we are trying to pass the
    // data from CPU to GPU. As we keep the order in tl_data_ as internal_copy_stream_, we will use
    // an actual stream for running and synchronizing with the copy. Note that the Copy can be truly
    // asynchronous if it comes from pinned memory or happens on a device with integrated memory
    // (like Xavier) where CPU and GPU share the same memory.
    if (!order.is_device()) {
      order = tl_elm.front()->order();
    }
    tl_elm.front()->Copy(batch, order, use_copy_kernel);
    CUDA_CALL(cudaEventRecord(*copy_to_storage_event.front(), order.stream()));
    if (sync) {
      CUDA_CALL(cudaEventSynchronize(*copy_to_storage_event.front()));
    }

    {
      std::lock_guard<std::mutex> busy_lock(busy_m_);
      tl_data_.PushBack(tl_elm);
      copy_to_storage_events_.PushBack(copy_to_storage_event);
      state_.push_back({false, false});
    }
  }


  template<typename SrcBackend>
  void SetDataSourceHelper(const TensorList<SrcBackend> &batch, AccessOrder order = {},
                           InputOperatorSettingMode ext_src_setting_mode = {}) {
    // Note: If we create a GPU source, we will need to figure
    // out what stream we want to do this copy in. CPU we can
    // pass anything as it is ignored.

    bool actual_no_copy = no_copy_;
    switch (ext_src_setting_mode.no_copy_mode) {
      case InputOperatorNoCopyMode::FORCE_COPY:
        actual_no_copy = false;
        break;
      case InputOperatorNoCopyMode::FORCE_NO_COPY:
        actual_no_copy = true;
        break;
      default:
        actual_no_copy = no_copy_;
        break;
    }

    if (actual_no_copy) {
      ShareUserData(batch, order, ext_src_setting_mode.use_copy_kernel);
    } else {
      CopyUserData(batch, order, ext_src_setting_mode.sync, ext_src_setting_mode.use_copy_kernel);
    }
    cv_.notify_one();
  }


  /**
   * @brief Get the empty output batch from tl_data_, first assigning the correct order to it.
   * @warning User is responsible for holding busy_m_ mutex when calling this function.
   */
  std::list<uptr_tl_type> GetEmptyOutputBatch() {
    auto result = tl_data_.GetEmpty();
    result.front()->set_order(internal_copy_order_);
    return result;
  }


  CachingList<uptr_tl_type> tl_data_;
  CachingList<uptr_cuda_event_type> copy_to_storage_events_;

  std::mutex busy_m_;
  std::condition_variable cv_;

  std::list<InputSourceState> state_;

  /*
   * indicates that user provide noncontiguous GPU input with zero copy option so DALI needs
   * to create an internal copy, it is used to raise a warning when the user mixes contiguous and
   * noncontiguous GPU inputs with zero copy what trashed GPU allocated memory
   */
  bool zero_copy_noncontiguous_gpu_input_ = false;

  WorkerThread sync_worker_;
  CUDAStreamLease internal_copy_stream_ = {};
  AccessOrder internal_copy_order_ = AccessOrder::host();
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_INPUT_OPERATOR_H_
