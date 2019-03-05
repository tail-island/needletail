#pragma once

#include <array>
#include <cstring>
#include <fstream>
#include <iostream>
#include <tuple>

#include <tensorflow/c/c_api.h>

#include "game.hpp"

namespace needletail {
  template<int BATCH_SIZE>
  class tf_predict final {
    TF_Graph*                 _graph;
    TF_Session*               _session;

    TF_Tensor*                _x_tensor;

    std::array<TF_Output,  1> _i_operations;
    std::array<TF_Tensor*, 1> _i_tensors;

    std::array<TF_Output,  2> _o_operations;
    std::array<TF_Tensor*, 2> _o_tensors;

    TF_Status*                _status;

  private:
    static auto create_graph_def(const std::string& model_path) {
      auto [data, length] = [&]() {
        auto stream = std::ifstream(model_path, std::ios::binary | std::ifstream::ate);

        auto length = stream.tellg();
        auto data   = new char[length];

        stream.seekg(0);
        stream.read(data, length);

        return std::make_tuple(data, length);
      }();

      auto result = TF_NewBuffer();

      result->data             = data;
      result->length           = length;
      result->data_deallocator = [](auto data, auto length) { delete static_cast<char*>(data); };

      return result;
    }

    static auto create_graph(const std::string& model_path) {
      auto result = TF_NewGraph();

      auto graph_def = create_graph_def(model_path);
      auto options   = TF_NewImportGraphDefOptions();
      auto status    = TF_NewStatus();

      TF_GraphImportGraphDef(result, graph_def, options, status);

      if (TF_GetCode(status) != TF_OK) {
        throw std::runtime_error(TF_Message(status));
      }

      TF_DeleteStatus(status);
      TF_DeleteImportGraphDefOptions(options);
      TF_DeleteBuffer(graph_def);

      return result;
    }

    static auto create_session(TF_Graph* graph) {
      auto options = TF_NewSessionOptions();
      auto status  = TF_NewStatus();

      auto result  = TF_NewSession(graph, options, status);

      if (TF_GetCode(status) != TF_OK) {
        throw std::runtime_error(TF_Message(status));
      }

      TF_DeleteStatus(status);
      TF_DeleteSessionOptions(options);

      return result;
    }

    static auto create_x_tensor() {
      return TF_AllocateTensor(TF_FLOAT, std::array<std::int64_t, 4>{BATCH_SIZE, 128, 128, 34}.data(), 4, sizeof(float) * BATCH_SIZE * 128 * 128 * 34);
    }

    static auto create_i_operations(TF_Graph* graph) {
      return std::array<TF_Output, 1>{TF_Output{TF_GraphOperationByName(graph, "input_1"), 0}};
    }

    static auto create_i_tensors(TF_Tensor* x_tensor) {
      return std::array<TF_Tensor*, 1>{x_tensor};
    }

    static auto create_o_operations(TF_Graph* graph) {
      return std::array<TF_Output, 2>{TF_Output{TF_GraphOperationByName(graph, "activation_152/Softmax"), 0}, TF_Output{TF_GraphOperationByName(graph, "activation_153/Tanh"), 0}};
    }

    static auto create_o_tensors() {
      return std::array<TF_Tensor*, 2>();
    }

    static auto create_status() {
      return TF_NewStatus();
    }

  public:
    tf_predict(const std::string& model_path):
      _graph(create_graph(model_path)),
      _session(create_session(_graph)),
      _x_tensor(create_x_tensor()),
      _i_operations(create_i_operations(_graph)),
      _i_tensors(create_i_tensors(_x_tensor)),
      _o_operations(create_o_operations(_graph)),
      _o_tensors(create_o_tensors()),
      _status(create_status())
    {
      ;
    }

    ~tf_predict() {
      TF_DeleteSession(_session, _status);
      TF_DeleteGraph(_graph);
      TF_DeleteTensor(_x_tensor);
      TF_DeleteStatus(_status);
    }

  public:
    auto operator()(const std::array<float, BATCH_SIZE * 128 * 128 * 34>& x) {
      std::memcpy(TF_TensorData(_x_tensor), x.data(), TF_TensorByteSize(_x_tensor));

      TF_SessionRun(_session, 0, _i_operations.data(), _i_tensors.data(), _i_tensors.size(), _o_operations.data(), _o_tensors.data(), _o_tensors.size(), 0, 0, 0, _status);

      if (TF_GetCode(_status) != TF_OK) {
        throw std::runtime_error(TF_Message(_status));
      }

      auto y_1 = std::array<float, BATCH_SIZE * MAX_POINT_SIZE>();
      auto y_2 = std::array<float, BATCH_SIZE *              1>();

      std::memcpy(y_1.data(), TF_TensorData(_o_tensors[0]), sizeof(float) * BATCH_SIZE * MAX_POINT_SIZE);
      std::memcpy(y_2.data(), TF_TensorData(_o_tensors[1]), sizeof(float) * BATCH_SIZE *              1);

      TF_DeleteTensor(_o_tensors[0]);
      TF_DeleteTensor(_o_tensors[1]);

      return std::make_tuple(y_1, y_2);
    }
  };
}
