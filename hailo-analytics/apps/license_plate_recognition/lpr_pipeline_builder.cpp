#include "lpr_pipeline_builder.hpp"

#include <ctype.h>
#include <stdint.h>
#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <string_view>
#include <map>
#include <optional>
#include <vector>

#include "hailo_analytics/analytics/ai_models_config.hpp"
#include "hailo_postprocess_tools/labels/hailo_yolov8n.hpp"
#include "hailo_analytics/analytics/common_configs.hpp"
#include "hailo_analytics/analytics/detection.hpp"
#include "hailo_analytics/analytics/license_plate_recognition.hpp"
#include "hailo_analytics/analytics/tiling.hpp"

namespace lpr_app
{

namespace ai_models = hailo_analytics::analytics::ai_models;

namespace tiling = hailo_analytics::analytics::tiling;
namespace lpr = hailo_analytics::analytics::license_plate_recognition;

// Half the analytics-API default of 100 ms — prioritise latency over batch fill.
static constexpr std::chrono::milliseconds LPR_SCHEDULER_TIMEOUT{50};

// License plates are too small to track at the detection rate; they take the ID of their vehicle.
static constexpr const char *LICENSE_PLATE_LABEL = "license_plate";
static constexpr const char *VEHICLE_LABEL = "vehicle";

// The OCR scheduler timeout bounds how long a partly filled batch waits.
static constexpr int OCR_BATCH_SIZE = 8;
// PaddleOCR keeps the text aspect ratio and pads with gray.
static constexpr dsp_color_t OCR_PAD_COLOR = {.y = 128, .u = 128, .v = 128};

static constexpr std::string_view TRACKING_MODE_FAST = "fast";
static constexpr std::string_view TRACKING_MODE_SLOW = "slow";
static constexpr std::string_view TRACKING_MODE_BALANCED = "balanced";
static constexpr std::string_view DETECTOR_YOLOV8N = "n";
static constexpr std::string_view DETECTOR_YOLOV8S = "s";

struct TrackerConfig
{
    uint8_t max_missed_frames;
    float smoothing_alpha;
    float association_threshold;
};

static constexpr TrackerConfig FAST_CONFIG{1, 0.9f, 0.3f};
static constexpr TrackerConfig BALANCED_CONFIG{3, 0.7f, 0.3f};
static constexpr TrackerConfig SLOW_CONFIG{6, 0.5f, 0.4f};

static constexpr const TrackerConfig &get_tracker_config(TrackingMode mode)
{
    switch (mode)
    {
    case TrackingMode::FAST:
        return FAST_CONFIG;
    case TrackingMode::SLOW:
        return SLOW_CONFIG;
    case TrackingMode::BALANCED:
        return BALANCED_CONFIG;
    }
    __builtin_unreachable();
}

TrackingMode tracking_mode_from_string(const std::string &mode_str)
{
    std::string lower = mode_str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == TRACKING_MODE_FAST)
    {
        return TrackingMode::FAST;
    }
    if (lower == TRACKING_MODE_SLOW)
    {
        return TrackingMode::SLOW;
    }
    if (lower == TRACKING_MODE_BALANCED)
    {
        return TrackingMode::BALANCED;
    }
    throw std::invalid_argument("Invalid tracking mode: '" + mode_str +
                                "'. Expected: " + std::string(TRACKING_MODE_SLOW) + ", " +
                                std::string(TRACKING_MODE_FAST) + ", or " + std::string(TRACKING_MODE_BALANCED));
}

DetectorModel detector_model_from_string(const std::string &model_str)
{
    if (model_str == DETECTOR_YOLOV8N)
    {
        return DetectorModel::YOLOV8N;
    }
    if (model_str == DETECTOR_YOLOV8S)
    {
        return DetectorModel::YOLOV8S;
    }
    throw std::invalid_argument("Invalid detector: '" + model_str + "'. Expected: " + std::string(DETECTOR_YOLOV8N) +
                                " or " + std::string(DETECTOR_YOLOV8S));
}

tiling::tiling_detection_config_t default_tiling_config(TrackingMode mode, DetectorModel detector)
{
    auto cfg = tiling::base_config();
    ai_models::apply_to(detector == DetectorModel::YOLOV8S ? ai_models::YOLOV8S : ai_models::YOLOV8N,
                        cfg.detection_config);
    cfg.tiling_config.queue_size = 1;
    cfg.aggregator_config.main_queue_size = 3;

    cfg.detection_config.ai_config.scheduler_timeout = LPR_SCHEDULER_TIMEOUT;

    // NMS class indices (0-based): 0=person, 1=vehicle, 2=face, 3=license_plate
    // Skip person and face — keep only vehicle and license_plate
    cfg.detection_config.ai_config.nms_classes_filter_mask = {false, true, false, true};

    // Enable detection tracker with mode-specific parameters
    const auto &tracker_params = get_tracker_config(mode);
    cfg.tracker_config.enabled = true;
    cfg.tracker_config.queue_size = 1;
    cfg.tracker_config.labels_map = common::hailo_yolov8n;
    cfg.tracker_config.max_missed_frames = tracker_params.max_missed_frames;
    cfg.tracker_config.smoothing_alpha = tracker_params.smoothing_alpha;
    cfg.tracker_config.association_threshold = tracker_params.association_threshold;
    cfg.tracker_config.anchored_labels = std::map<std::string, std::string>{{LICENSE_PLATE_LABEL, VEHICLE_LABEL}};

    return cfg;
}

lpr::bbox_crop_ocr_config_t default_lpr_config(const OcrOptions &options)
{
    auto cfg = lpr::base_config();
    ai_models::apply_to(ai_models::PADDLE_OCR, cfg.ocr_config);
    if (!options.hef_path.empty())
        cfg.ocr_config.ai_config.hef_path = options.hef_path;
    cfg.ocr_config.ai_config.batch_size = OCR_BATCH_SIZE;
    cfg.ocr_config.ai_config.scheduler_threshold = OCR_BATCH_SIZE;
    cfg.bbox_crop_config.input_width = 1920;
    cfg.bbox_crop_config.input_height = 1080;
    cfg.bbox_crop_config.queue_size = 1;
    // Plates exist only on detection frames, so every frame is offered to the crop stage.
    cfg.bbox_crop_config.crop_every_x_frames = 1;
    cfg.bbox_crop_config.letterbox_alignment = DSP_LETTERBOX_UP_LEFT;
    cfg.bbox_crop_config.letterbox_color = OCR_PAD_COLOR;
    cfg.quality_gate_config.vote_min_count = options.vote_min_count;
    cfg.aggregator_config.main_queue_size = 3;
    cfg.aggregator_config.sub_queue_size = 20;
    cfg.ocr_config.ai_config.queue_size = 20;
    cfg.ocr_config.post_config.queue_size = 20;

    return cfg;
}

tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_tiling_pipeline(
    const std::string &name, TrackingMode mode, bool use_hailort_service, DetectorModel detector)
{
    auto cfg = default_tiling_config(mode, detector);
    cfg.detection_config.ai_config.use_hailort_service = use_hailort_service;
    return tiling::generate_tiling_detection_pipeline(name, cfg);
}

tl::expected<hailo_analytics::pipeline::PipelinePtr, hailo_analytics::pipeline::AppStatus> build_ocr_pipeline(
    const std::string &name, bool use_hailort_service, const OcrOptions &options)
{
    auto cfg = default_lpr_config(options);
    cfg.ocr_config.ai_config.use_hailort_service = use_hailort_service;
    return lpr::generate_bbox_crop_ocr_pipeline(name, cfg);
}

} // namespace lpr_app
