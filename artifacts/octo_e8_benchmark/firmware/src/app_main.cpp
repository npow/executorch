#include "RTE_Components.h"
#include CMSIS_device_header
#include "board_config.h"
#include "Driver_IO.h"
#include "ospi_psram_xip.h"
#include "APS512XXN_PSRAM.h"
#include "appkit_psram.hpp"
#include "sys_utils.h"
#include "ethosu_driver.h"
#include "arm_embedded_module.hpp"
#include <executorch/runtime/platform/runtime.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::runtime::EValue;
using executorch::runtime::Error;
using executorch::runtime::MemoryAllocator;
using arm::embedded::EmbeddedModule;

namespace torch::executor::native {
Tensor& _to_dim_order_copy_out(KernelRuntimeContext&, const Tensor&, bool,
    executorch::aten::OptionalArrayRef<int64_t>, Tensor&);
Tensor& cat_out(KernelRuntimeContext&, executorch::aten::ArrayRef<Tensor>, int64_t, Tensor&);
}

extern "C" ARM_DRIVER_GPIO Driver_GPIO15;

struct Benchmark {
    uint32_t magic;
    uint32_t state;
    uint32_t command;
    uint32_t backbone_bytes;
    uint32_t denoise_bytes;
    uint32_t error;
    uint32_t run;
    uint32_t phase;
    uint32_t backbone_ms[6];
    uint32_t denoise_ms[6];
    uint32_t total_ms[6];
    float actions[28];
    uint32_t upload_destination;
    uint32_t upload_bytes;
    uint32_t upload_crc;
    uint32_t upload_sequence;
    uint32_t upload_ack;
    uint32_t upload_operation;
};

extern "C" {
volatile Benchmark octo_benchmark;
volatile uint32_t octo_ticks;
volatile uint32_t octo_npu_snapshot[163];
uint32_t octo_command_bytes;
uint32_t octo_staged_command_bytes;
uint32_t octo_dma_chunk_bytes; // Retained for diagnostic host-script compatibility; unused.
volatile uint32_t octo_probe_output;
volatile uint32_t octo_probe_strides[4];
int ethosu_semaphore_take(void* semaphore, uint64_t) {
    auto* count = reinterpret_cast<volatile uint8_t*>(semaphore);
    const uint32_t start = octo_ticks;
    const uint32_t limit = octo_benchmark.command >= 3 ? 5000 : 120000;
    while (*count == 0) {
        if (octo_ticks - start >= limit) return -1;
        __WFI();
    }
    --*count;
    return 0;
}
void ethosu_inference_begin(struct ethosu_driver* driver, void*) {
    octo_npu_snapshot[0] = octo_ticks;
    octo_command_bytes = driver->job.custom_data_size;
    *reinterpret_cast<volatile uint32_t*>(0x49042054) = 0;
    const bool normal = octo_benchmark.command <= 2 || octo_benchmark.command >= 7;
    auto* attributes = reinterpret_cast<volatile uint32_t*>(0x49042040);
    attributes[0] = normal ? 0x20 : 0;
    attributes[1] = normal ? 0x20 : 0;
    attributes[2] = normal ? 0x24 : 4;
    attributes[3] = normal ? 0x24 : 4;
}
void ethosu_inference_end(struct ethosu_driver* driver, void*) {
    octo_npu_snapshot[1] = octo_ticks;
    octo_npu_snapshot[2] = driver->job.result;
    auto* registers = reinterpret_cast<volatile uint32_t*>(0x49042000);
    for (unsigned i = 0; i < 160; ++i) octo_npu_snapshot[i + 3] = registers[i];
}
char octo_last_error[2048];
void SysTick_Handler(void) { ++octo_ticks; }
void et_pal_emit_log_message(et_timestamp_t, et_pal_log_level_t, const char* filename,
                            const char*, size_t line, const char* message, size_t) {
    size_t offset = strlen(octo_last_error);
    if (offset + 2 < sizeof(octo_last_error))
        snprintf(octo_last_error + offset, sizeof(octo_last_error) - offset, "%s:%u %s\n", filename, unsigned(line), message);
}
}

alignas(32) uint8_t temp_pool[1024 * 1024] __attribute__((section(".bss.ai_pool")));
alignas(32) uint8_t fast_pool[384 * 1024] __attribute__((section(".bss.ai_pool")));
extern "C" {
unsigned char* ethosu_fast_scratch = fast_pool;
size_t ethosu_fast_scratch_size = sizeof(fast_pool);
uint64_t ethosu_address_remap(uint64_t address, int index) {
    if (index == -1 && address && octo_command_bytes >= 32 && octo_command_bytes - 32 <= sizeof(temp_pool)) {
        octo_staged_command_bytes = octo_command_bytes - 32;
        memcpy(temp_pool, reinterpret_cast<const void*>(static_cast<uintptr_t>(address)),
               octo_staged_command_bytes);
        SCB_CleanDCache_by_Addr(reinterpret_cast<uint32_t*>(temp_pool), octo_staged_command_bytes);
        return LocalToGlobal(temp_pool);
    }
    return address ? LocalToGlobal(reinterpret_cast<void*>(static_cast<uintptr_t>(address))) : 0;
}
unsigned int ethosu_config_select(uint64_t address, int index) {
    if (index == -1 && address == LocalToGlobal(temp_pool))
        *reinterpret_cast<volatile uint32_t*>(0x49042020) = octo_staged_command_bytes;
    return (address & 0xa0000000ULL) ? 2 : 0;
}
}
float embedding[384];
float actions[28];
float timestep[1];

static uint32_t rng = 2026;
static uint32_t crc32(const void* data, size_t bytes) {
    static uint32_t table[256];
    static bool initialized = false;
    if (!initialized) {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) value = (value >> 1) ^ ((value & 1) ? 0xedb88320U : 0);
            table[i] = value;
        }
        initialized = true;
    }
    uint32_t value = 0xffffffffU;
    auto* input = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < bytes; ++i) value = (value >> 8) ^ table[(value ^ input[i]) & 255];
    return value ^ 0xffffffffU;
}

static float gaussian() {
    float sum = -6.0f;
    for (int i = 0; i < 12; ++i) {
        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
        sum += static_cast<float>(rng >> 8) / 16777216.0f;
    }
    return sum;
}

struct Input {
    TensorImpl::SizesType sizes[5];
    TensorImpl::DimOrderType order[5];
    TensorImpl::StridesType strides[5];
    std::unique_ptr<TensorImpl> impl;
    Input(std::initializer_list<int> dimensions, void* data, ScalarType type = ScalarType::Float, bool history = false) {
        int i = 0;
        for (int size : dimensions) { sizes[i] = size; order[i] = i; ++i; }
        if (history) { order[2] = 3; order[3] = 4; order[4] = 2; }
        int stride = 1;
        for (int j = i - 1; j >= 0; --j) { strides[order[j]] = stride; stride *= sizes[order[j]]; }
        impl = std::make_unique<TensorImpl>(type, i, sizes, data, order, strides);
    }
    Tensor tensor() { return Tensor(impl.get()); }
};

static bool check_history_kernels() {
    uint8_t source[24], physical[24] = {}, output[24] = {};
    for (unsigned i = 0; i < 24; ++i) source[i] = i + 1;
    Input input({1, 2, 3, 2, 2}, source, ScalarType::Byte);
    Input history({1, 2, 3, 2, 2}, physical, ScalarType::Byte, true);
    Input result({1, 2, 3, 2, 2}, output, ScalarType::Byte);
    auto in = input.tensor(), interleaved = history.tensor(), out = result.tensor();
    executorch::runtime::KernelRuntimeContext context;
    const int64_t order[] = {0, 1, 3, 4, 2};
    torch::executor::native::_to_dim_order_copy_out(context, in, false,
        executorch::aten::ArrayRef<int64_t>(order, 5), interleaved);
    for (unsigned frame = 0; frame < 2; ++frame)
        for (unsigned pixel = 0; pixel < 4; ++pixel)
            for (unsigned channel = 0; channel < 3; ++channel)
                if (physical[frame * 12 + pixel * 3 + channel] !=
                    source[frame * 12 + channel * 4 + pixel]) return false;
    torch::executor::native::cat_out(context,
        executorch::aten::ArrayRef<Tensor>(&interleaved, 1), -1, out);
    return memcmp(source, output, sizeof(source)) == 0;
}

static void run_memory_probe(unsigned mode) {
    const bool exact = mode >= 11;
    auto* input = reinterpret_cast<float*>(exact ? 0xa3b00000 : 0x02300000);
    for (unsigned i = 0; i < (exact ? 2 * 6 * 256 * 256 : 16 * 64 * 64); ++i)
        input[i] = (int(i % 127) - 63) / 64.0f;
    const unsigned placement = exact ? 3 : (mode - 3) % 4;
    auto* pte = reinterpret_cast<uint8_t*>((placement & 1) ? 0xa3e00000 : 0x02340000);
    auto* scratch = reinterpret_cast<uint8_t*>((placement & 2) ? 0xa2600000 : 0x02280000);
    if (pte == reinterpret_cast<uint8_t*>(0x02340000))
        memcpy(pte, reinterpret_cast<void*>(0xa3e00000), octo_benchmark.backbone_bytes);
    executorch::runtime::Span<uint8_t> planned(reinterpret_cast<uint8_t*>(0xa1f00000), 6 * 1024 * 1024);
    executorch::runtime::HierarchicalAllocator hierarchy({&planned, 1});
    EmbeddedModule module(pte, octo_benchmark.backbone_bytes,
        std::make_unique<executorch::extension::BufferDataLoader>(pte, octo_benchmark.backbone_bytes),
        std::make_unique<MemoryAllocator>(exact ? 512 * 1024 : 1024 * 1024,
            reinterpret_cast<uint8_t*>(exact ? 0x02380000 : 0x02180000)),
        std::make_unique<MemoryAllocator>(exact ? 26 * 1024 * 1024 : 512 * 1024, scratch));
    Input tensor(exact ? std::initializer_list<int>{2, 6, 256, 256} : std::initializer_list<int>{1, 16, 64, 64}, input);
    octo_benchmark.phase = mode;
    octo_benchmark.error = 0;
    octo_probe_output = 0;
    memset(octo_last_error, 0, sizeof(octo_last_error));
    octo_benchmark.state = 4;
    if (exact) {
        const auto error = module.load_method("forward", &hierarchy);
        if (error != Error::Ok) { octo_benchmark.error = static_cast<uint32_t>(error); octo_benchmark.state = 92; return; }
    }
    uint32_t start = octo_ticks;
    auto result = module.execute("forward", std::vector<EValue>{tensor.tensor()});
    octo_benchmark.total_ms[0] = octo_ticks - start;
    if (!result.ok()) {
        octo_benchmark.error = static_cast<uint32_t>(result.error());
        octo_benchmark.state = 93;
        return;
    }
    const auto& output = result.get()[0].toTensor();
    octo_probe_output = reinterpret_cast<uintptr_t>(output.const_data_ptr<float>());
    for (unsigned i = 0; i < 4; ++i) octo_probe_strides[i] = output.strides()[i];
    octo_benchmark.state = 5;
}

extern "C" int app_main() {
    octo_benchmark.magic = 0x4f43544f;
    octo_benchmark.state = 1;
    SysTick_Config(SystemCoreClock / 1000);
    printf("Octo benchmark, core clock %lu\n", (unsigned long)SystemCoreClock);
    int ram = appkit_psram_init();
    if (ram != 0) { octo_benchmark.error = -ram; octo_benchmark.state = 90; return -1; }
    volatile uint32_t* probe = reinterpret_cast<volatile uint32_t*>(0xa3fff000);
    probe[0] = 0x1234abcd; probe[16] = 0xaabbccdd;
    SCB_CleanInvalidateDCache();
    if (probe[0] != 0x1234abcd || probe[16] != 0xaabbccdd) {
        octo_benchmark.state = 91; return -1;
    }
    if (!check_history_kernels()) { octo_benchmark.state = 96; return -1; }
    printf("External RAM and history kernels ready; waiting for models\n");
    octo_benchmark.state = 2;
    while (octo_benchmark.command == 0) {
        if (octo_benchmark.upload_sequence != octo_benchmark.upload_ack) {
            uint32_t address = octo_benchmark.upload_destination;
            uint32_t bytes = octo_benchmark.upload_bytes;
            uint32_t operation = octo_benchmark.upload_operation;
            if (address < 0xa0000000 || bytes > 64 * 1024 * 1024 || address > 0xa4000000U - bytes ||
                (operation == 1 && bytes > sizeof(temp_pool))) {
                octo_benchmark.upload_crc = 0;
            } else {
                auto* destination = reinterpret_cast<uint32_t*>(address);
                if (operation == 1) {
                    SCB_InvalidateDCache_by_Addr(reinterpret_cast<uint32_t*>(temp_pool), bytes);
                    memcpy(destination, temp_pool, bytes);
                    SCB_CleanInvalidateDCache_by_Addr(destination, bytes);
                }
                octo_benchmark.upload_crc = crc32(destination, bytes);
            }
            __DSB();
            octo_benchmark.upload_ack = octo_benchmark.upload_sequence;
        }
    }
    if (octo_benchmark.command >= 3) {
        executorch::runtime::runtime_init();
        while (octo_benchmark.command >= 3) {
            unsigned mode = octo_benchmark.command;
            run_memory_probe(mode);
            while (octo_benchmark.command == mode) __WFI();
        }
        while (true) __WFI();
    }
    const bool head_only = octo_benchmark.command == 2;
    SCB_InvalidateDCache();
    octo_benchmark.state = 3;
    executorch::runtime::runtime_init();
    executorch::runtime::Span<uint8_t> backbone_planned(reinterpret_cast<uint8_t*>(0xa1f00000), 6 * 1024 * 1024);
    executorch::runtime::Span<uint8_t> denoise_planned(reinterpret_cast<uint8_t*>(0xa2500000), 1024 * 1024);
    executorch::runtime::HierarchicalAllocator backbone_hierarchy({&backbone_planned, 1});
    executorch::runtime::HierarchicalAllocator denoise_hierarchy({&denoise_planned, 1});
    EmbeddedModule backbone(reinterpret_cast<uint8_t*>(0xa0000000), octo_benchmark.backbone_bytes,
        std::make_unique<executorch::extension::BufferDataLoader>(reinterpret_cast<uint8_t*>(0xa0000000), octo_benchmark.backbone_bytes),
        std::make_unique<MemoryAllocator>(512 * 1024, reinterpret_cast<uint8_t*>(0x02180000)),
        std::make_unique<MemoryAllocator>(26 * 1024 * 1024, reinterpret_cast<uint8_t*>(0xa2600000)));
    EmbeddedModule denoise(reinterpret_cast<uint8_t*>(0xa1c00000), octo_benchmark.denoise_bytes,
        std::make_unique<executorch::extension::BufferDataLoader>(reinterpret_cast<uint8_t*>(0xa1c00000), octo_benchmark.denoise_bytes),
        std::make_unique<MemoryAllocator>(128 * 1024, reinterpret_cast<uint8_t*>(0x02200000)),
        std::make_unique<MemoryAllocator>(26 * 1024 * 1024, reinterpret_cast<uint8_t*>(0xa2600000)));
    for (auto* module : {&backbone, &denoise}) {
        if (head_only && module == &backbone) continue;
        auto error = module->load_method("forward", module == &backbone ? &backbone_hierarchy : &denoise_hierarchy);
        if (error != Error::Ok) { octo_benchmark.error = static_cast<uint32_t>(error); octo_benchmark.state = 92; return -1; }
    }
    octo_benchmark.state = 4;
    Input primary({1, 2, 3, 256, 256}, reinterpret_cast<void*>(0xa1e00000), ScalarType::Byte);
    Input wrist({1, 2, 3, 128, 128}, reinterpret_cast<void*>(0xa1e60000), ScalarType::Byte);
    Input language({1, 16, 768}, reinterpret_cast<void*>(0xa1e80000));
    Input condition({1, 384}, embedding);
    Input action({1, 28}, actions);
    Input time({1, 1}, timestep);
    std::vector<EValue> backbone_inputs = {primary.tensor(), wrist.tensor(), language.tensor()};
    std::vector<EValue> denoise_inputs = {condition.tensor(), action.tensor(), time.tensor()};
    const float* schedule = reinterpret_cast<const float*>(0xa1e8c000);
    for (unsigned run = 0; run < 1; ++run) {
        octo_benchmark.run = run;
        octo_benchmark.phase = 1;
        uint32_t start = octo_ticks;
        if (head_only) {
            memcpy(embedding, reinterpret_cast<void*>(0xa1e90000), sizeof(embedding));
        } else {
            auto result = backbone.execute("forward", backbone_inputs);
            if (!result.ok()) { octo_benchmark.error = static_cast<uint32_t>(result.error()); octo_benchmark.state = 93; return -1; }
            memcpy(embedding, result.get()[0].toTensor().const_data_ptr<float>(), sizeof(embedding));
        }
        uint32_t middle = octo_ticks;
        octo_benchmark.backbone_ms[run] = middle - start;
        for (float& a : actions) a = gaussian();
        for (int step = 19; step >= 0; --step) {
            octo_benchmark.phase = 20 - step;
            timestep[0] = step;
            auto prediction = denoise.execute("forward", denoise_inputs);
            if (!prediction.ok()) { octo_benchmark.error = static_cast<uint32_t>(prediction.error()); octo_benchmark.state = 94; return -1; }
            auto* epsilon = prediction.get()[0].toTensor().const_data_ptr<float>();
            float beta = schedule[step], alpha = schedule[20 + step], alpha_hat = schedule[40 + step];
            for (int j = 0; j < 28; ++j) {
                float value = (actions[j] - (1 - alpha) / sqrtf(1 - alpha_hat) * epsilon[j]) / sqrtf(alpha);
                if (step > 0) value += sqrtf(beta) * gaussian();
                actions[j] = fminf(5.0f, fmaxf(-5.0f, value));
                if (!std::isfinite(epsilon[j])) { octo_benchmark.state = 95; return -1; }
            }
        }
        octo_benchmark.denoise_ms[run] = octo_ticks - middle;
        octo_benchmark.total_ms[run] = octo_ticks - start;
        printf("run %u: %lu ms\n", run, (unsigned long)octo_benchmark.total_ms[run]);
    }
    for (int j = 0; j < 28; ++j) octo_benchmark.actions[j] = actions[j];
    octo_benchmark.state = 5;
    while (true) { __WFI(); }
}
