#include "RTE_Components.h"
#include CMSIS_device_header
#include "RTE_Device.h"
#include "board_config.h"
#include "ospi_psram_xip.h"
#include "APS512XXN_PSRAM.h"
#include "sys_ctrl_aes.h"
#include "appkit_psram.hpp"
#include <cstring>

// Fixed read latency keeps both bytes of x16 reads aligned during refresh.
extern "C" void __real_ospi_send_polling(OSPI_Type*, ospi_transfer_t*);
extern "C" void __wrap_ospi_send_polling(OSPI_Type* ospi, ospi_transfer_t* transfer) {
    const uint32_t* original = transfer->tx_buff;
    uint32_t fixed_latency[3];
    const bool mode_zero = transfer->tx_total_cnt == 3 && original[0] == APS512XXN_CMD_MODE_REGISTER_WRITE &&
        original[1] == APS512XXN_MODE_REG0_ADDR;
    if (mode_zero) {
        memcpy(fixed_latency, original, sizeof(fixed_latency));
        fixed_latency[2] |= 0x2020;
        transfer->tx_buff = fixed_latency;
    }
    __real_ospi_send_polling(ospi, transfer);
    if (mode_zero) transfer->tx_buff = original + 3;
    for (volatile unsigned wait = 0; wait < 10000; ++wait) __NOP();
}

static int32_t reset_ram(OSPI_Type* ospi, AES_Type* aes) {
    aes_disable_xip(aes);
    ospi_disable(ospi);
    ospi->OSPI_TXFTLR = 0;
    return aps512xxn_psram_init(ospi, aes);
}

int appkit_psram_init() {
    int error = board_clocks_config(CLKEN_SYSPLL_MASK | CLKEN_HFOSC_MASK |
        CLKEN_CLK_100M_MASK | (1U << CLKEN_CLK_400M) | (1U << CLKEN_SRAM1));
    if (error) return error;
    ospi_psram_xip_config config = {};
    config.instance = OSPI_INSTANCE_0;
    config.ram_init = reset_ram;
    config.ram_type = RAM_TYPE_PSRAM;
    config.config_mode = CONFIG_MODE_USER_PARAMETERS;
    config.bus_speed = 25000000;
    config.spi_frf = OSPI_SPI_FRF_DUAL_OCTAL;
    config.wait_cycles = 4;
    config.ddr_drive_edge = 1;
    config.rxds_delay = 6;
    config.signal_delay = 22;
    config.dfs = 32;
    error = ospi_psram_xip_init(&config);
    aes_set_rxds_delay(AES0, 11);
    ospi_disable(OSPI0);
    OSPI0->OSPI_XIP_CTRL = (OSPI0->OSPI_XIP_CTRL &
        ~((0x1fU << XIP_CTRL_WAIT_CYCLES_OFFSET) | (1U << XIP_CTRL_RXDS_VL_EN_OFFSET))) |
        (7U << XIP_CTRL_WAIT_CYCLES_OFFSET);
    ospi_enable(OSPI0);
    return error;
}
