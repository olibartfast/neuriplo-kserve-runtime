#include "PipelineSteps.hpp"

#include "TensorBytes.hpp"

#include "neuriplo/tasks/core/image_io.hpp"
#include "neuriplo/tasks/core/model_info.hpp"
#include "neuriplo/tasks/core/result_types.hpp"
#include "neuriplo/tasks/core/task_config.hpp"
#include "neuriplo/tasks/core/task_factory.hpp"
#include "neuriplo/tasks/core/task_interface.hpp"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

constexpr const char *kFrameSizeTensor = "FRAME_SIZE";

// Cap on the pixel count a preprocess step will decode. An attacker-supplied
// JPEG/PNG can declare dimensions far beyond anything the header bytes
// actually carry (a "decompression bomb"): a few hundred bytes can declare a
// 30000x30000 image, which the decoder would then allocate ~2.7 GB for. Baked
// in rather than left to the decoder, because the decoder's own allocation is
// the thing this cap exists to avoid ever reaching.
constexpr int64_t kPipelineMaxDecodedPixels = 64LL * 1024LL * 1024LL; // 64 megapixels

// Reads the declared width/height straight out of a PNG, JPEG, or BMP header,
// without decoding, so a declared-huge image can be rejected before the
// decoder allocates its pixel buffer. Returns false when the bytes are not a
// recognized header of one of those formats (truncated, a different format,
// or not an image at all) -- the caller then falls through to the normal
// decode path, which rejects those on its own terms. TGA has no signature to
// detect it by, so a declared-huge TGA is not caught here (left as a known
// gap: see the roadmap follow-up for this cap).
bool decodedImageDimensions(const uint8_t *data, size_t size, int64_t &width, int64_t &height) {
    static constexpr uint8_t kPngSignature[8] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
    if (size >= 8 && std::memcmp(data, kPngSignature, sizeof(kPngSignature)) == 0) {
        // Walk chunks rather than assuming IHDR is the first one: Apple's
        // CgBI PNGs (iOS app-bundle PNGs, which stb_image's PNG decoder
        // accepts) insert a private "CgBI" chunk before IHDR -- and nothing
        // bounds how many of those a file can have. No separate chunk-count
        // limit is needed: every chunk this loop skips over advances `offset`
        // by at least 12 bytes (length + type + crc, even with no chunk
        // data), so the walk is already bounded by `size`.
        size_t offset = 8;
        while (offset + 12 <= size) {
            const auto chunk_length = (static_cast<uint32_t>(data[offset]) << 24) |
                                      (static_cast<uint32_t>(data[offset + 1]) << 16) |
                                      (static_cast<uint32_t>(data[offset + 2]) << 8) |
                                      static_cast<uint32_t>(data[offset + 3]);
            if (std::memcmp(data + offset + 4, "IHDR", 4) == 0) {
                if (offset + 16 > size) {
                    return false; // truncated IHDR
                }
                // IHDR data: 4-byte width then 4-byte height, both
                // big-endian, starting right after the 4-byte length and
                // 4-byte "IHDR" type.
                width = (static_cast<int64_t>(data[offset + 8]) << 24) |
                        (static_cast<int64_t>(data[offset + 9]) << 16) |
                        (static_cast<int64_t>(data[offset + 10]) << 8) |
                        static_cast<int64_t>(data[offset + 11]);
                height = (static_cast<int64_t>(data[offset + 12]) << 24) |
                         (static_cast<int64_t>(data[offset + 13]) << 16) |
                         (static_cast<int64_t>(data[offset + 14]) << 8) |
                         static_cast<int64_t>(data[offset + 15]);
                return true;
            }
            // length(4) + type(4) + data(chunk_length) + crc(4).
            offset += 12 + static_cast<size_t>(chunk_length);
        }
        return false;
    }

    if (size >= 2 && data[0] == 0xFF) {
        // JPEG: walk marker segments until a Start-Of-Frame marker, which
        // carries the frame's height/width right after its own length field.
        // Matches stb_image's stbi__get_marker exactly:
        //  - SOI itself may be preceded by extra 0xFF fill bytes, and so may
        //    any later marker code (the JPEG spec allows padding before the
        //    real marker byte); both are skipped rather than treated as the
        //    marker.
        //  - The very first marker read after SOI must be a real marker
        //    (junk there is an error, same as stb); every later one may be
        //    preceded by non-0xFF junk bytes too, left over by some encoders
        //    between segments, which stb skips one byte at a time while
        //    scanning for the next marker.
        size_t soi_offset = 0;
        while (soi_offset < size && data[soi_offset] == 0xFF) {
            ++soi_offset;
        }
        if (soi_offset >= size || data[soi_offset] != 0xD8) {
            return false; // not a JPEG after all
        }

        size_t offset = soi_offset + 1;
        bool first_marker = true;
        while (offset < size) {
            if (data[offset] != 0xFF) {
                if (first_marker) {
                    return false; // junk right after SOI: stb rejects this too
                }
                ++offset; // junk between segments: skip it byte by byte
                continue;
            }
            size_t code_offset = offset + 1;
            while (code_offset < size && data[code_offset] == 0xFF) {
                ++code_offset; // skip fill bytes before the real marker code
            }
            if (code_offset >= size) {
                return false; // truncated
            }
            const uint8_t marker = data[code_offset];
            first_marker = false;
            if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD9)) {
                offset = code_offset + 1; // markers with no payload (RSTn, TEM, EOI)
                continue;
            }
            const bool is_sof = marker >= 0xC0 && marker <= 0xCF && marker != 0xC4 &&
                                marker != 0xC8 && marker != 0xCC;
            if (is_sof) {
                if (code_offset + 7 >= size) {
                    return false; // truncated before height/width
                }
                height = (static_cast<int64_t>(data[code_offset + 4]) << 8) |
                         static_cast<int64_t>(data[code_offset + 5]);
                width = (static_cast<int64_t>(data[code_offset + 6]) << 8) |
                        static_cast<int64_t>(data[code_offset + 7]);
                return true;
            }
            if (code_offset + 2 >= size) {
                return false;
            }
            const size_t segment_length = (static_cast<size_t>(data[code_offset + 1]) << 8) |
                                          static_cast<size_t>(data[code_offset + 2]);
            if (segment_length < 2) {
                return false;
            }
            offset = code_offset + 1 + segment_length;
        }
        return false;
    }

    if (size >= 18 && data[0] == 'B' && data[1] == 'M') {
        // BMP: the DIB header-size field at offset 14 says which header
        // follows, and that determines where width/height live. stb_image's
        // BMP decoder accepts exactly five header sizes: the old
        // BITMAPCOREHEADER (12), which stores unsigned 16-bit width/height
        // right after it, and BITMAPINFOHEADER and its later revisions
        // (40, 56, 108, 124), which all share the signed 32-bit width/height
        // layout at offsets 18/22. Anything else is a header stb itself does
        // not decode.
        const uint32_t header_size =
            static_cast<uint32_t>(data[14]) | (static_cast<uint32_t>(data[15]) << 8) |
            (static_cast<uint32_t>(data[16]) << 16) | (static_cast<uint32_t>(data[17]) << 24);
        if (header_size == 12) {
            if (size < 22) {
                return false; // truncated
            }
            width = static_cast<int64_t>(data[18]) | (static_cast<int64_t>(data[19]) << 8);
            height = static_cast<int64_t>(data[20]) | (static_cast<int64_t>(data[21]) << 8);
            return true;
        }
        if (header_size == 40 || header_size == 56 || header_size == 108 || header_size == 124) {
            if (size < 26) {
                return false; // truncated
            }
            // Width/height are signed 32-bit little-endian. A negative height
            // means the rows are stored top-down; only the magnitude matters
            // for a pixel count. Negating INT32_MIN overflows a 32-bit int,
            // so the sign flip happens after widening to int64_t, where that
            // value is always representable and the negation cannot overflow.
            const auto read_i32_le = [data](size_t at) -> int64_t {
                const uint32_t bits = static_cast<uint32_t>(data[at]) |
                                      (static_cast<uint32_t>(data[at + 1]) << 8) |
                                      (static_cast<uint32_t>(data[at + 2]) << 16) |
                                      (static_cast<uint32_t>(data[at + 3]) << 24);
                return static_cast<int64_t>(static_cast<int32_t>(bits));
            };
            const int64_t raw_width = read_i32_le(18);
            const int64_t raw_height = read_i32_le(22);
            width = raw_width < 0 ? -raw_width : raw_width;
            height = raw_height < 0 ? -raw_height : raw_height;
            return true;
        }
        return false; // a header size stb_image does not recognize
    }

    return false; // unrecognized header
}

// The source image's pixel dimensions travel through the graph as an explicit
// tensor rather than as hidden state on the step: postprocessing needs them to
// map boxes back onto the original frame, and making the dependency a graph
// edge is what lets graph validation catch a postprocess step wired without it.
TensorMetadata frameSizeTensor() {
    // INT64 (height, width): the same convention the GPU postprocessing
    // operators consume, so one producer serves both postprocess paths.
    return {kFrameSizeTensor, "INT64", {2}};
}

// Mirrors neuriplo-infer's setInputFormat: pick the layout from the shape,
// since KServe metadata carries no layout field.
void setInputFormat(neuriplo_tasks::ModelInfo &model_info) {
    if (model_info.input_formats.empty() || model_info.input_shapes.empty() ||
        model_info.input_shapes[0].empty()) {
        return;
    }

    const auto &shape = model_info.input_shapes[0];
    if (shape.size() == 4) {
        const bool is_nchw = (shape[1] == 1 || shape[1] == 3);
        const bool is_nhwc = (shape[3] == 1 || shape[3] == 3);
        if (is_nchw && !is_nhwc) {
            model_info.input_formats[0] = "FORMAT_NCHW";
        } else if (!is_nchw && is_nhwc) {
            model_info.input_formats[0] = "FORMAT_NHWC";
        } else if (shape[2] > 3 && shape[3] > 3) {
            model_info.input_formats[0] = "FORMAT_NCHW";
        } else if (shape[1] > 3 && shape[2] > 3) {
            model_info.input_formats[0] = "FORMAT_NHWC";
        } else {
            model_info.input_formats[0] = "FORMAT_NCHW";
        }
    } else if (shape.size() == 3) {
        model_info.input_formats[0] = "FORMAT_NCHW";
    }
}

neuriplo_tasks::ModelInfo buildModelInfo(const ModelMetadata &metadata) {
    neuriplo_tasks::ModelInfo model_info;
    for (const auto &input : metadata.inputs) {
        std::vector<int64_t> shape = input.shape;
        if (shape.size() == 3) {
            shape.insert(shape.begin(), 1);
        }
        model_info.addInput(input.name, shape, 1);
    }
    for (const auto &output : metadata.outputs) {
        model_info.addOutput(output.name, output.shape, 1);
    }
    setInputFormat(model_info);
    if (!model_info.input_types.empty()) {
        model_info.input_types[0] = neuriplo_tasks::vision::PixelType::Float32;
    }
    return model_info;
}

neuriplo_tasks::TaskConfig buildTaskConfig(const PipelineStepConfig &step) {
    neuriplo_tasks::TaskConfig config;
    config.confidence_threshold = step.confidence_threshold;
    config.nms_threshold = step.nms_threshold;
    config.mask_threshold = step.mask_threshold;
    config.segmentation_output = step.envelope == PipelineEnvelope::Polygon
                                     ? neuriplo_tasks::SegmentationOutput::Polygon
                                     : neuriplo_tasks::SegmentationOutput::Mask;
    return config;
}

// Raw KServe bytes -> the task layer's element-variant tensor.
bool toTaskTensor(const OutputTensor &source, neuriplo_tasks::Tensor &target, std::string &error) {
    target.shape = source.shape;
    const size_t element_size = tensorElementSize(source.datatype);
    if (element_size == 0) {
        error = "tensor '" + source.name + "' has no fixed-size datatype: " + source.datatype;
        return false;
    }
    const size_t count = source.bytes.size() / element_size;
    target.data.resize(count);

    for (size_t i = 0; i < count; ++i) {
        if (source.datatype == "FP32" || source.datatype == "FP16") {
            target.data[i] = tensorScalarAt<float>(source.bytes, i);
        } else if (source.datatype == "INT32") {
            target.data[i] = tensorScalarAt<int32_t>(source.bytes, i);
        } else if (source.datatype == "INT64") {
            target.data[i] = tensorScalarAt<int64_t>(source.bytes, i);
        } else if (source.datatype == "UINT8" || source.datatype == "BOOL") {
            target.data[i] = tensorScalarAt<uint8_t>(source.bytes, i);
        } else {
            error = "tensor '" + source.name +
                    "' has a datatype the task layer cannot carry: " + source.datatype;
            return false;
        }
    }
    return true;
}

OutputTensor makeTensor(const TensorMetadata &metadata) {
    OutputTensor tensor;
    tensor.name = metadata.name;
    tensor.datatype = metadata.datatype;
    tensor.shape = metadata.shape;
    return tensor;
}

class PreprocessStep : public PipelineStep {
  public:
    PreprocessStep(PipelineStepConfig step, ModelMetadata neighbour)
        : step_(std::move(step)), neighbour_(std::move(neighbour)) {
        // [1, -1]: one image per request (the contract fixes max_batch_size at
        // 1), of a byte length that varies per request.
        inputs_.push_back({pipelineImageInputName(), pipelineImageInputDatatype(), {1, -1}});
        for (const auto &input : neighbour_.inputs) {
            outputs_.push_back(input);
        }
        outputs_.push_back(frameSizeTensor());
        model_info_ = buildModelInfo(neighbour_);
    }

    const std::vector<TensorMetadata> &inputs() const override {
        return inputs_;
    }
    const std::vector<TensorMetadata> &outputs() const override {
        return outputs_;
    }

    bool run(const std::vector<OutputTensor> &step_inputs, std::vector<OutputTensor> &step_outputs,
             std::string &error) override {
        if (step_inputs.size() != 1 || step_inputs[0].bytes.empty()) {
            error = "preprocess step '" + step_.name + "' expects one non-empty encoded image";
            return false;
        }

        try {
            const auto *bytes =
                reinterpret_cast<const uint8_t *>(step_inputs[0].bytes.data()); // NOLINT
            const auto byte_count = step_inputs[0].bytes.size();

            // Reject a declared-huge image before the decoder allocates its
            // pixel buffer. A header we cannot parse (undecodable, or not a
            // PNG/JPEG at all) falls through to decodeImage unchanged, which
            // rejects it on its own terms.
            int64_t declared_width = 0;
            int64_t declared_height = 0;
            // Division rather than multiplication: declared_width * declared_height
            // can overflow a signed int64_t (PNG dimensions are full uint32, so
            // both operands can be near 2^32), which is undefined behaviour. The
            // error message is built only from the bounded inputs, never from
            // their product.
            if (decodedImageDimensions(bytes, byte_count, declared_width, declared_height) &&
                declared_width > 0 && declared_height > 0 &&
                declared_width > kPipelineMaxDecodedPixels / declared_height) {
                error = "preprocess step '" + step_.name +
                        "' rejected an encoded image declaring " + std::to_string(declared_width) +
                        "x" + std::to_string(declared_height) + " pixels, over the " +
                        std::to_string(kPipelineMaxDecodedPixels) + "-pixel cap";
                return false;
            }

            const auto image = neuriplo_tasks::decodeImage(bytes, byte_count, 3);

            auto task = neuriplo_tasks::TaskFactory::createTaskInstance(
                step_.task_type, model_info_, buildTaskConfig(step_));
            auto preprocessed = task->preprocess({image});

            const size_t model_input_count = neighbour_.inputs.size();
            if (preprocessed.size() < model_input_count) {
                error = "preprocess step '" + step_.name + "' produced " +
                        std::to_string(preprocessed.size()) + " tensors for a model with " +
                        std::to_string(model_input_count) + " inputs";
                return false;
            }

            step_outputs.clear();
            for (size_t i = 0; i < model_input_count; ++i) {
                auto tensor = makeTensor(neighbour_.inputs[i]);
                tensor.bytes.resize(preprocessed[i].size());
                std::memcpy(tensor.bytes.data(), preprocessed[i].data(), preprocessed[i].size());
                step_outputs.push_back(std::move(tensor));
            }

            auto frame_size = makeTensor(frameSizeTensor());
            appendTensorScalar<int64_t>(frame_size.bytes, image.height());
            appendTensorScalar<int64_t>(frame_size.bytes, image.width());
            step_outputs.push_back(std::move(frame_size));
            return true;
        } catch (const std::exception &exception) {
            error = "preprocess step '" + step_.name + "' failed: " + exception.what();
            return false;
        }
    }

  private:
    PipelineStepConfig step_;
    ModelMetadata neighbour_;
    neuriplo_tasks::ModelInfo model_info_;
    std::vector<TensorMetadata> inputs_;
    std::vector<TensorMetadata> outputs_;
};

class PostprocessStep : public PipelineStep {
  public:
    PostprocessStep(PipelineStepConfig step, ModelMetadata neighbour)
        : step_(std::move(step)), neighbour_(std::move(neighbour)) {
        for (const auto &output : neighbour_.outputs) {
            inputs_.push_back(output);
        }
        inputs_.push_back(frameSizeTensor());
        outputs_ = pipelineEnvelopeOutputs(step_.envelope);
        model_info_ = buildModelInfo(neighbour_);
    }

    const std::vector<TensorMetadata> &inputs() const override {
        return inputs_;
    }
    const std::vector<TensorMetadata> &outputs() const override {
        return outputs_;
    }

    bool run(const std::vector<OutputTensor> &step_inputs, std::vector<OutputTensor> &step_outputs,
             std::string &error) override {
        if (step_inputs.size() != inputs_.size()) {
            error = "postprocess step '" + step_.name + "' expects " +
                    std::to_string(inputs_.size()) + " inputs, got " +
                    std::to_string(step_inputs.size());
            return false;
        }

        // FRAME_SIZE is INT64 (height, width). Getting the order wrong
        // transposes every box on a non-square frame, so the layout is fixed
        // rather than inferred. The datatype is checked explicitly rather than
        // inferred from byte length: a wrongly-typed producer (e.g. INT32 or
        // FP32) can coincidentally supply exactly 16 bytes and pass a
        // byte-length-only check while every value inside is read as the
        // wrong type.
        const auto &frame_size_tensor = step_inputs.back();
        if (frame_size_tensor.datatype != "INT64" ||
            frame_size_tensor.bytes.size() < 2 * sizeof(int64_t)) {
            error = "postprocess step '" + step_.name +
                    "' requires FRAME_SIZE as INT64 with at least 2 elements, got datatype '" +
                    frame_size_tensor.datatype + "' with " +
                    std::to_string(frame_size_tensor.bytes.size()) + " bytes";
            return false;
        }
        const auto frame_height =
            static_cast<int>(tensorScalarAt<int64_t>(frame_size_tensor.bytes, 0));
        const auto frame_width =
            static_cast<int>(tensorScalarAt<int64_t>(frame_size_tensor.bytes, 1));
        if (frame_width <= 0 || frame_height <= 0) {
            error = "postprocess step '" + step_.name + "' received a non-positive FRAME_SIZE";
            return false;
        }

        std::vector<neuriplo_tasks::Tensor> tensors;
        tensors.reserve(step_inputs.size() - 1);
        for (size_t i = 0; i + 1 < step_inputs.size(); ++i) {
            neuriplo_tasks::Tensor tensor;
            if (!toTaskTensor(step_inputs[i], tensor, error)) {
                return false;
            }
            tensors.push_back(std::move(tensor));
        }

        try {
            auto task = neuriplo_tasks::TaskFactory::createTaskInstance(
                step_.task_type, model_info_, buildTaskConfig(step_));
            const auto results =
                task->postprocess(neuriplo_tasks::vision::Size{frame_width, frame_height}, tensors);
            step_outputs = encodeEnvelope(results);
            return true;
        } catch (const std::exception &exception) {
            error = "postprocess step '" + step_.name + "' failed: " + exception.what();
            return false;
        }
    }

  private:
    struct DecodedDetection {
        int32_t x = 0;
        int32_t y = 0;
        int32_t width = 0;
        int32_t height = 0;
        float score = 0.0F;
        int32_t class_id = 0;
        const neuriplo_tasks::InstanceSegmentation *segmentation = nullptr;
    };

    std::vector<DecodedDetection>
    collect(const std::vector<neuriplo_tasks::Result> &results) const {
        // Every result first, unsorted and uncapped.
        std::vector<DecodedDetection> all;
        all.reserve(results.size());
        for (const auto &result : results) {
            if (const auto *segmentation =
                    std::get_if<neuriplo_tasks::InstanceSegmentation>(&result)) {
                all.push_back({static_cast<int32_t>(segmentation->bbox.x),
                               static_cast<int32_t>(segmentation->bbox.y),
                               static_cast<int32_t>(segmentation->bbox.width),
                               static_cast<int32_t>(segmentation->bbox.height),
                               segmentation->class_confidence,
                               static_cast<int32_t>(segmentation->class_id), segmentation});
            } else if (const auto *detection = std::get_if<neuriplo_tasks::Detection>(&result)) {
                all.push_back({static_cast<int32_t>(detection->bbox.x),
                               static_cast<int32_t>(detection->bbox.y),
                               static_cast<int32_t>(detection->bbox.width),
                               static_cast<int32_t>(detection->bbox.height),
                               detection->class_confidence,
                               static_cast<int32_t>(detection->class_id), nullptr});
            }
        }

        // The task layer's NMS does not promise score order, so capping while
        // scanning in task order kept whichever `kPipelineMaxDetections`
        // happened to come first and silently dropped higher-scoring
        // detections that came later. pipelineTopScoreIndices ranks by score
        // before the cap is applied.
        std::vector<float> scores;
        scores.reserve(all.size());
        for (const auto &detection : all) {
            scores.push_back(detection.score);
        }
        const auto ranked =
            pipelineTopScoreIndices(scores, static_cast<size_t>(kPipelineMaxDetections));

        std::vector<DecodedDetection> detections;
        detections.reserve(ranked.size());
        for (const auto index : ranked) {
            detections.push_back(all[index]);
        }
        return detections;
    }

    std::vector<OutputTensor> encodeEnvelope(const std::vector<neuriplo_tasks::Result> &results) {
        const auto detections = collect(results);
        const auto count = static_cast<int32_t>(detections.size());

        std::vector<OutputTensor> envelope;
        for (const auto &metadata : outputs_) {
            envelope.push_back(makeTensor(metadata));
        }

        appendTensorScalar<int32_t>(envelope[0].bytes, count);
        // BOXES, SCORES, and CLASSES are always emitted at full length; the
        // rows past NUM_DETECTIONS are contractual padding.
        for (int64_t i = 0; i < kPipelineMaxDetections; ++i) {
            const bool valid = i < count;
            const auto &detection = valid ? detections[static_cast<size_t>(i)] : DecodedDetection{};
            appendTensorScalar<int32_t>(envelope[1].bytes, detection.x);
            appendTensorScalar<int32_t>(envelope[1].bytes, detection.y);
            appendTensorScalar<int32_t>(envelope[1].bytes, detection.width);
            appendTensorScalar<int32_t>(envelope[1].bytes, detection.height);
            appendTensorScalar<float>(envelope[2].bytes, detection.score);
            appendTensorScalar<int32_t>(envelope[3].bytes, detection.class_id);
        }

        if (step_.envelope == PipelineEnvelope::Mask) {
            encodeMasks(detections, envelope);
        } else if (step_.envelope == PipelineEnvelope::Polygon) {
            encodePolygons(detections, envelope);
        }
        return envelope;
    }

    // Extracts one detection's mask as box-sized UINT8 bytes.
    //
    // The envelope carries no per-mask dimensions -- the contract says a mask
    // covers its detection's box, so the box is what tells the client how to
    // read the bytes back. Task postprocessors are not consistent here: some
    // populate mask_data, some only the mask image, and the mask image is
    // sometimes box-sized and sometimes full-frame. All three are normalized to
    // box-sized bytes so the client never has to guess.

    // Offset arrays are always kPipelineMaxDetections + 1 entries, whatever the
    // detection count. Emitting a short array on an empty frame is the defect
    // this shape exists to prevent.
    static void encodeMasks(const std::vector<DecodedDetection> &detections,
                            std::vector<OutputTensor> &envelope) {
        auto &offsets = envelope[4];
        auto &data = envelope[5];

        int64_t offset = 0;
        appendTensorScalar<int64_t>(offsets.bytes, offset);
        for (int64_t i = 0; i < kPipelineMaxDetections; ++i) {
            if (i < static_cast<int64_t>(detections.size())) {
                const auto *segmentation = detections[static_cast<size_t>(i)].segmentation;
                if (segmentation != nullptr) {
                    const auto mask = pipelineBoxSizedMask(*segmentation);
                    if (!mask.empty()) {
                        const size_t previous = data.bytes.size();
                        data.bytes.resize(previous + mask.size());
                        std::memcpy(data.bytes.data() + previous, mask.data(), mask.size());
                        offset += static_cast<int64_t>(mask.size());
                    }
                }
            }
            appendTensorScalar<int64_t>(offsets.bytes, offset);
        }
        data.shape = {static_cast<int64_t>(data.bytes.size())};
    }

    static void encodePolygons(const std::vector<DecodedDetection> &detections,
                               std::vector<OutputTensor> &envelope) {
        auto &instance_offsets = envelope[4];
        auto &ring_offsets = envelope[5];
        auto &points = envelope[6];

        int64_t ring_count = 0;
        int64_t point_count = 0;
        appendTensorScalar<int64_t>(instance_offsets.bytes, ring_count);
        appendTensorScalar<int64_t>(ring_offsets.bytes, point_count);

        for (int64_t i = 0; i < kPipelineMaxDetections; ++i) {
            if (i < static_cast<int64_t>(detections.size())) {
                const auto *segmentation = detections[static_cast<size_t>(i)].segmentation;
                if (segmentation != nullptr) {
                    for (const auto &polygon : segmentation->polygons) {
                        appendRing(polygon.exterior, points, ring_offsets, point_count, ring_count);
                        for (const auto &hole : polygon.holes) {
                            appendRing(hole, points, ring_offsets, point_count, ring_count);
                        }
                    }
                }
            }
            appendTensorScalar<int64_t>(instance_offsets.bytes, ring_count);
        }

        ring_offsets.shape = {ring_count + 1};
        points.shape = {point_count, 2};
    }

    static void appendRing(const std::vector<neuriplo_tasks::vision::Point2f> &ring,
                           OutputTensor &points, OutputTensor &ring_offsets, int64_t &point_count,
                           int64_t &ring_count) {
        // Rings with fewer than three points cannot bound an area; the contract
        // requires consumers never see them.
        if (ring.size() < 3) {
            return;
        }
        for (const auto &point : ring) {
            appendTensorScalar<int32_t>(points.bytes, static_cast<int32_t>(point.x));
            appendTensorScalar<int32_t>(points.bytes, static_cast<int32_t>(point.y));
        }
        point_count += static_cast<int64_t>(ring.size());
        ++ring_count;
        appendTensorScalar<int64_t>(ring_offsets.bytes, point_count);
    }

    PipelineStepConfig step_;
    ModelMetadata neighbour_;
    neuriplo_tasks::ModelInfo model_info_;
    std::vector<TensorMetadata> inputs_;
    std::vector<TensorMetadata> outputs_;
};

} // namespace

std::vector<uint8_t> pipelineBoxSizedMask(const neuriplo_tasks::InstanceSegmentation &seg) {
    const auto box_width = static_cast<int>(seg.bbox.width);
    const auto box_height = static_cast<int>(seg.bbox.height);
    if (box_width <= 0 || box_height <= 0) {
        return {};
    }
    const auto expected = static_cast<size_t>(box_width) * static_cast<size_t>(box_height);

    if (seg.mask_data.size() == expected) {
        return seg.mask_data;
    }
    if (seg.mask.empty()) {
        return {};
    }

    const auto *pixels = seg.mask.data();
    const auto rows = seg.mask.rows();
    const auto cols = seg.mask.cols();
    if (pixels == nullptr || rows <= 0 || cols <= 0 ||
        seg.mask.pixelType() != neuriplo_tasks::vision::PixelType::UInt8) {
        return {};
    }

    std::vector<uint8_t> mask(expected, 0);
    if (rows == box_height && cols == box_width) {
        std::memcpy(mask.data(), pixels, expected);
        return mask;
    }

    // Full-frame mask: copy out the box region, clipped to the frame.
    const auto origin_x = static_cast<int>(seg.bbox.x);
    const auto origin_y = static_cast<int>(seg.bbox.y);
    for (int row = 0; row < box_height; ++row) {
        const int source_row = origin_y + row;
        if (source_row < 0 || source_row >= rows) {
            continue;
        }
        for (int column = 0; column < box_width; ++column) {
            const int source_column = origin_x + column;
            if (source_column < 0 || source_column >= cols) {
                continue;
            }
            mask[static_cast<size_t>(row) * static_cast<size_t>(box_width) +
                 static_cast<size_t>(column)] =
                pixels[static_cast<size_t>(source_row) * static_cast<size_t>(cols) +
                       static_cast<size_t>(source_column)];
        }
    }
    return mask;
}

std::unique_ptr<PipelineStep> makeBuiltinPipelineStep(const PipelineStepConfig &step,
                                                      const ModelMetadata &neighbour,
                                                      std::string &error) {
    switch (step.kind) {
    case PipelineStepKind::Preprocess:
        if (neighbour.inputs.empty()) {
            error = "preprocess step '" + step.name +
                    "' has no following model step whose input it can produce";
            return nullptr;
        }
        return std::make_unique<PreprocessStep>(step, neighbour);
    case PipelineStepKind::Postprocess:
        if (neighbour.outputs.empty()) {
            error = "postprocess step '" + step.name +
                    "' has no preceding model step whose outputs it can decode";
            return nullptr;
        }
        return std::make_unique<PostprocessStep>(step, neighbour);
    case PipelineStepKind::Model:
        break;
    }
    error = "pipeline step '" + step.name + "' is not a built-in step";
    return nullptr;
}
