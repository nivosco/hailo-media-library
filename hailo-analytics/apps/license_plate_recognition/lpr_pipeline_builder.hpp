#pragma once

#include <tl/expected.hpp>
#include <string>

#include "hailo_analytics/analytics/tiling.hpp"
#include "hailo_analytics/analytics/license_plate_recognition.hpp"
#include "hailo_analytics/pipeline/core/pipeline.hpp"
#include "hailo_analytics/pipeline/core/stage.hpp"

namespace lpr_app
{

enum class TrackingMode
{
    SLOW,
    FAST,
    BALANCED
};

TrackingMode tracking_mode_from_string(const std::string &mode_str);

inline constexpr int DEFAULT_VOTE_MIN_COUNT = 3;

/**
 * @brief OCR settings a user may tune; the defaults are the validated configuration.
 */
struct OcrOptions
{
    std::string hef_path;                        ///< OCR HEF; empty selects the bundled PP-OCRv6-medium HEF
    int vote_min_count = DEFAULT_VOTE_MIN_COUNT; ///< Matching reads needed before a plate text is published
};

hailo_analytics::analytics::tiling::tiling_detection_config_t default_tiling_config(
    TrackingMode mode = TrackingMode::BALANCED);

hailo_analytics::analytics::license_plate_recognition::bbox_crop_ocr_config_t default_lpr_config(
    const OcrOptions &options = {});

// use_hailort_service routes inference through hailort_server rather than opening an exclusive
// VDevice. It must be set whenever media_library_service owns the media library, so that the two
// processes share the NN core instead of contending for it.
tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_tiling_pipeline(
    const std::string &name, TrackingMode mode = TrackingMode::BALANCED, bool use_hailort_service = false);

tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_ocr_pipeline(
    const std::string &name, bool use_hailort_service = false, const OcrOptions &options = {});

} // namespace lpr_app
