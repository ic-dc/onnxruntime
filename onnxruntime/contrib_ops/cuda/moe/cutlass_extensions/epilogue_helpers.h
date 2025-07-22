/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/**
 * @file epilogue_helpers.h
 *
 * This file includes types for the epilogues. The empty structs exist so we can signal to template
 * code the type of epilogue we want to run, and let the underlying code specify the details such as
 * element types, accumulator type and elements per vector access.
 *
 */

#pragma once

#include "contrib_ops/cuda/moe/cutlass_extensions/epilogue/thread/fused_activations.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"

namespace ort_fastertransformer {

struct EpilogueOpBiasSilu {};

struct EpilogueOpBiasReLU {};

struct EpilogueOpBiasFtGelu {};

struct EpilogueOpDefaultSwiGLU {};

struct EpilogueOpDefaultSilu {};

struct EpilogueOpDefaultReLU {};

struct EpilogueOpDefaultFtGelu {};

struct EpilogueOpBias {};

struct EpilogueOpDefault {};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator, typename Op>
struct Epilogue {};

constexpr auto BiasScaleMode = cutlass::epilogue::thread::ScaleType::NoBetaScaling;

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasSilu> {
  using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType, ElementsPerVectorAccess, ElementAccumulator,
                                                              ElementAccumulator, BiasScaleMode>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasReLU> {
  using Op = cutlass::epilogue::thread::LinearCombinationRelu<ElementType, ElementsPerVectorAccess, ElementAccumulator,
                                                              ElementAccumulator, BiasScaleMode>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBiasFtGelu> {
  using Op = cutlass::epilogue::thread::LinearCombinationGeneric<
      cutlass::epilogue::thread::GELU_taylor, ElementType, ElementsPerVectorAccess, ElementAccumulator,
      ElementAccumulator, BiasScaleMode, cutlass::FloatRoundStyle::round_to_nearest, true>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpBias> {
  using Op = cutlass::epilogue::thread::LinearCombination<ElementType, ElementsPerVectorAccess, ElementAccumulator,
                                                          ElementAccumulator, BiasScaleMode>;
};

constexpr auto DefaultScaleMode = cutlass::epilogue::thread::ScaleType::Default;

namespace epilogue {
namespace thread {

template <
    typename ElementOutput_,
    int ElementsPerAccess,
    typename ElementAccumulator_ = ElementOutput_,
    typename ElementCompute_ = ElementAccumulator_,
    cutlass::epilogue::thread::ScaleType::Kind Scale = cutlass::epilogue::thread::ScaleType::Default,
    cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest,
    typename ElementSource_ = ElementOutput_>
class LinearCombinationSwiGLU {
 public:
  using ElementOutput = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementSource = ElementSource_;
  using ElementCompute = ElementCompute_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static const cutlass::epilogue::thread::ScaleType::Kind kScale = Scale;
  static const cutlass::FloatRoundStyle kRound = Round;

  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentSource = cutlass::Array<ElementSource, kElementsPerAccess>;
  using FragmentOutput = cutlass::Array<ElementOutput, kElementsPerAccess>;
  using ComputeFragment = cutlass::Array<ElementCompute, kElementsPerAccess>;

  struct Params {
    ElementCompute alpha;
    ElementCompute beta;
    ElementCompute const* alpha_ptr;
    ElementCompute const* beta_ptr;

    CUTLASS_HOST_DEVICE
    Params() : alpha(1), beta(0), alpha_ptr(nullptr), beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha, ElementCompute beta = ElementCompute(0)) : alpha(alpha), beta(beta), alpha_ptr(nullptr), beta_ptr(nullptr) {}

    CUTLASS_HOST_DEVICE
    Params(ElementCompute const* alpha_ptr, ElementCompute const* beta_ptr = nullptr) : alpha(1), beta(0), alpha_ptr(alpha_ptr), beta_ptr(beta_ptr) {}
  };

 private:
  ElementCompute alpha_;
  ElementCompute beta_;

 public:
  CUTLASS_HOST_DEVICE
  LinearCombinationSwiGLU(Params const& params) {
    alpha_ = (params.alpha_ptr ? *params.alpha_ptr : params.alpha);
    beta_ = (params.beta_ptr ? *params.beta_ptr : params.beta);
  }

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return beta_ != ElementCompute(0);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator, FragmentSource const& source) const {
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess> accumulator_converter;
    cutlass::NumericArrayConverter<ElementCompute, ElementSource, kElementsPerAccess> source_converter;

    ComputeFragment converted_acc = accumulator_converter(accumulator);
    ComputeFragment converted_source = source_converter(source);

    ComputeFragment intermediate;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      intermediate[i] = alpha_ * converted_acc[i] + beta_ * converted_source[i];
    }

    // SwiGLU logic
    ComputeFragment swiglu_result;
    constexpr float swiglu_alpha = 1.702f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess / 2; ++i) {
      ElementCompute x_glu = intermediate[2 * i];
      ElementCompute x_linear = intermediate[2 * i + 1];

      ElementCompute sigmoid_arg = swiglu_alpha * x_glu;
      ElementCompute sigmoid_out = (ElementCompute)1 / ((ElementCompute)1 + cutlass::fast_exp(-sigmoid_arg));

      ElementCompute swish_out = x_glu * sigmoid_out;
      swiglu_result[i] = swish_out * (x_linear + (ElementCompute)1);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = kElementsPerAccess / 2; i < kElementsPerAccess; ++i) {
      swiglu_result[i] = ElementCompute(0);
    }

    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess> output_converter;
    return output_converter(swiglu_result);
  }

  CUTLASS_HOST_DEVICE
  FragmentOutput operator()(FragmentAccumulator const& accumulator) const {
    cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess> accumulator_converter;
    ComputeFragment converted_acc = accumulator_converter(accumulator);

    // SwiGLU logic
    ComputeFragment swiglu_result;
    constexpr float swiglu_alpha = 1.702f;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess / 2; ++i) {
      ElementCompute x_glu = converted_acc[2 * i];
      ElementCompute x_linear = converted_acc[2 * i + 1];

      ElementCompute sigmoid_arg = swiglu_alpha * x_glu;
      ElementCompute sigmoid_out = (ElementCompute)1 / ((ElementCompute)1 + cutlass::fast_exp(-sigmoid_arg));

      ElementCompute swish_out = x_glu * sigmoid_out;
      swiglu_result[i] = swish_out * (x_linear + (ElementCompute)1);
    }

    CUTLASS_PRAGMA_UNROLL
    for (int i = kElementsPerAccess / 2; i < kElementsPerAccess; ++i) {
      swiglu_result[i] = ElementCompute(0);
    }

    cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kElementsPerAccess> output_converter;
    return output_converter(swiglu_result);
  }
};

}  // namespace thread
}  // namespace epilogue

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefaultSwiGLU> {
  using Op = epilogue::thread::LinearCombinationSwiGLU<
      ElementType, ElementsPerVectorAccess, ElementAccumulator, ElementAccumulator,
      cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest, ElementType>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefaultSilu> {
  using Op = cutlass::epilogue::thread::LinearCombinationSilu<ElementType, ElementsPerVectorAccess, ElementAccumulator,
                                                              ElementAccumulator, DefaultScaleMode>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefaultReLU> {
  using Op = cutlass::epilogue::thread::LinearCombinationRelu<ElementType, ElementsPerVectorAccess, ElementAccumulator,
                                                              ElementAccumulator, DefaultScaleMode>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefaultFtGelu> {
  using Op = cutlass::epilogue::thread::LinearCombinationGeneric<
      cutlass::epilogue::thread::GELU_taylor, ElementType, ElementsPerVectorAccess, ElementAccumulator,
      ElementAccumulator, DefaultScaleMode, cutlass::FloatRoundStyle::round_to_nearest, true>;
};

template <typename ElementType, int ElementsPerVectorAccess, typename ElementAccumulator>
struct Epilogue<ElementType, ElementsPerVectorAccess, ElementAccumulator, EpilogueOpDefault> {
  using Op = cutlass::epilogue::thread::LinearCombination<ElementType, ElementsPerVectorAccess, ElementAccumulator,
                                                          ElementAccumulator, DefaultScaleMode>;
};

}  // namespace ort_fastertransformer
