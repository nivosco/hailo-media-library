/**
 * Copyright (c) 2021-2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the LGPL license (https://www.gnu.org/licenses/old-licenses/lgpl-2.1.txt)
 **/
#include "ocr_post.hpp"

#include <stdint.h>
#include <array>
#include <cmath>
#include <string>
#include <cstddef>
#include <memory>

#include "hailo_postprocess_tools/labels/ppocrv5_char_dict.hpp"
#include "hailo_postprocess_tools/labels/ppocrv6_char_dict.hpp"
#include "hailo_postprocess_tools/objects/hailo_objects.hpp"
#include "hailo_postprocess_tools/objects/hailo_tensors.hpp"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

//******************************************************************
// LICENSE PLATE OCR POST-PROCESSING
//******************************************************************

// PP-OCR CTC decoder — converts the quantized recognition tensor to text.
//
// Tensor shape: 1×40×C (height=1, width=40 timesteps, features=C classes).
// Classes 0,1 are blank/padding (CTC ignored tokens); classes 2..C-2 map to the
// character dictionary and class C-1 is the space token:
//   PP-OCRv5 mobile: C=18385, PPOCRV5_CHAR_DICT, softmax output (probabilities)
//   PP-OCRv6 small/medium: C=18710, PPOCRV6_CHAR_DICT, logits (softmax not in the HEF)
//
// Optimization: argmax is performed on quantized uint8 values directly (valid
// because qp_scale > 0). On AArch64, NEON SIMD processes 16 bytes at a time for
// ~8-16x speedup on the argmax scan. For logits, the winning class probability is
// computed only for emitted characters, using a 256-entry exp table indexed by the
// quantized distance to the maximum.

static constexpr uint32_t NUM_IGNORED_TOKENS = 2;
static constexpr uint32_t PPOCRV5_NUM_CLASSES = 18385;
static constexpr uint32_t PPOCRV6_NUM_CLASSES = 18710;
static constexpr std::size_t UINT8_LEVELS = 256;

struct CharDict
{
    const char *const *chars;
    std::size_t size;
};

static bool get_char_dict(uint32_t num_classes, CharDict &dict)
{
    if (num_classes == PPOCRV5_NUM_CLASSES)
    {
        dict = {common::PPOCRV5_CHAR_DICT, common::PPOCRV5_DICT_SIZE};
        return true;
    }
    if (num_classes == PPOCRV6_NUM_CLASSES)
    {
        dict = {common::PPOCRV6_CHAR_DICT, common::PPOCRV6_DICT_SIZE};
        return true;
    }
    return false;
}

// Softmax probability of the max class: 1 / sum_c exp((q_c - q_max) * scale).
static float max_class_probability_from_logits(const uint8_t *data, uint32_t len, uint8_t max_val,
                                               const std::array<float, UINT8_LEVELS> &exp_table)
{
    float sum = 0.0f;
    for (uint32_t c = 0; c < len; ++c)
        sum += exp_table[max_val - data[c]];
    return 1.0f / sum;
}

// Find argmax (index and value) over a uint8 array of length `len`.
static inline void argmax_uint8(const uint8_t *data, uint32_t len, uint32_t &out_idx, uint8_t &out_val)
{
#ifdef __aarch64__
    // Pass 1: find the maximum value using NEON (16 bytes per iteration)
    uint8x16_t vmax = vdupq_n_u8(0);
    uint32_t c = 0;
    for (; c + 16 <= len; c += 16)
        vmax = vmaxq_u8(vmax, vld1q_u8(data + c));
    uint8_t best_val = vmaxvq_u8(vmax);
    for (; c < len; ++c)
        best_val = (data[c] > best_val) ? data[c] : best_val;

    // Pass 2: find the first index that holds the max value
    uint8x16_t target = vdupq_n_u8(best_val);
    for (c = 0; c + 16 <= len; c += 16)
    {
        uint8x16_t cmp = vceqq_u8(vld1q_u8(data + c), target);
        if (vmaxvq_u8(cmp))
        {
            // At least one match in this 16-byte chunk — find the lane
            uint64_t lo = vgetq_lane_u64(vreinterpretq_u64_u8(cmp), 0);
            if (lo)
            {
                out_idx = c + static_cast<uint32_t>(__builtin_ctzll(lo) >> 3);
            }
            else
            {
                uint64_t hi = vgetq_lane_u64(vreinterpretq_u64_u8(cmp), 1);
                out_idx = c + 8 + static_cast<uint32_t>(__builtin_ctzll(hi) >> 3);
            }
            out_val = best_val;
            return;
        }
    }
    // Tail: match must be in the remaining elements
    for (; c < len; ++c)
    {
        if (data[c] == best_val)
        {
            out_idx = c;
            out_val = best_val;
            return;
        }
    }
#else
    // Scalar fallback for non-AArch64 builds (host tests, etc.)
    uint32_t best = 0;
    uint8_t best_val = data[0];
    for (uint32_t c = 1; c < len; ++c)
    {
        if (data[c] > best_val)
        {
            best_val = data[c];
            best = c;
        }
    }
    out_idx = best;
    out_val = best_val;
#endif
}

void ocr_postprocess(HailoROIPtr roi)
{
    if (!roi->has_tensors())
        return;

    auto tensors = roi->get_tensors();
    if (tensors.size() != 1)
        return;
    auto tensor = tensors[0];
    const uint32_t num_timesteps = tensor->width();
    const uint32_t num_classes = tensor->features();
    const float qp_scale = tensor->qp_scale();
    const float qp_zp = tensor->qp_zp();
    const uint8_t *data = tensor->data();

    CharDict dict{};
    if (!get_char_dict(num_classes, dict))
        return;

    const bool output_is_logits = tensor->name().find("softmax") == std::string::npos;
    std::array<float, UINT8_LEVELS> exp_table{};
    if (output_is_logits)
    {
        for (std::size_t d = 0; d < UINT8_LEVELS; ++d)
            exp_table[d] = std::exp(-static_cast<float>(d) * qp_scale);
    }

    std::string decoded_text;
    float confidence_sum = 0.0f;
    uint32_t num_chars = 0;
    uint32_t prev_idx = UINT32_MAX;

    for (uint32_t t = 0; t < num_timesteps; ++t)
    {
        const uint8_t *ts = data + static_cast<std::size_t>(t) * num_classes;

        // Prefetch next timestep into L1 cache
        if (t + 1 < num_timesteps)
            __builtin_prefetch(ts + num_classes, 0, 1);

        // SIMD-accelerated argmax on quantized values
        uint32_t best = 0;
        uint8_t best_val = 0;
        argmax_uint8(ts, num_classes, best, best_val);

        // Skip blank/padding tokens
        if (best < NUM_IGNORED_TOKENS)
        {
            prev_idx = best;
            continue;
        }

        // CTC duplicate removal
        if (best == prev_idx)
            continue;

        prev_idx = best;

        // Map to dictionary index
        uint32_t dict_idx = best - NUM_IGNORED_TOKENS;
        if (dict_idx >= dict.size)
            continue;

        decoded_text += dict.chars[dict_idx];
        confidence_sum += output_is_logits ? max_class_probability_from_logits(ts, num_classes, best_val, exp_table)
                                           : (static_cast<float>(best_val) - qp_zp) * qp_scale;
        ++num_chars;
    }

    if (decoded_text.empty())
        return;

    float mean_confidence = confidence_sum / static_cast<float>(num_chars);
    roi->add_object(std::make_shared<HailoClassification>("ocr", decoded_text, mean_confidence));
}

void filter(HailoROIPtr roi)
{
    ocr_postprocess(roi);
}
