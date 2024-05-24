#ifndef __PINMUX_H__
#define __PINMUX_H__


#include "kdrv_scu_ext.h"


#define PINMUX_SPI_WP_N_GET()           GET_BITS(SCU_EXTREG_SPI_WP_N_IOCTRL, 0, 2)
#define PINMUX_SPI_WP_N_SET(val)        SET_MASKED_BITS(SCU_EXTREG_SPI_WP_N_IOCTRL, val, 0, 2)
#define PINMUX_SPI_WP_N_GPIO0           3

#define PINMUX_SPI_HOLD_N_GET()         GET_BITS(SCU_EXTREG_SPI_HOLD_N_IOCTRL, 0, 2)
#define PINMUX_SPI_HOLD_N_SET(val)      SET_MASKED_BITS(SCU_EXTREG_SPI_HOLD_N_IOCTRL, val, 0, 2)
#define PINMUX_SPI_HOLD_N_GPIO1         3

#define PINMUX_SWJ_TRST_GET()           GET_BITS(SCU_EXTREG_SWJ_TRST_IOCTRL, 0, 2)
#define PINMUX_SWJ_TRST_SET(val)        SET_MASKED_BITS(SCU_EXTREG_SWJ_TRST_IOCTRL, val, 0, 2)
#define PINMUX_SWJ_TRST_GPIO2           3

#define PINMUX_SWJ_TDI_GET()            GET_BITS(SCU_EXTREG_SWJ_TDI_IOCTRL, 0, 2)
#define PINMUX_SWJ_TDI_SET(val)         SET_MASKED_BITS(SCU_EXTREG_SWJ_TDI_IOCTRL, val, 0, 2)
#define PINMUX_SWJ_TDI_GPIO3            3

#define PINMUX_SWJ_SWDITMS_GET()        GET_BITS(SCU_EXTREG_SWJ_SWDITMS_IOCTRL, 0, 2)
#define PINMUX_SWJ_SWDITMS_SET(val)     SET_MASKED_BITS(SCU_EXTREG_SWJ_SWDITMS_IOCTRL, val, 0, 2)
#define PINMUX_SWJ_SWDITMS_GPIO4        3

#define PINMUX_SWJ_SWCLKTCK_GET()       GET_BITS(SCU_EXTREG_SWJ_SWCLKTCK_IOCTRL, 0, 2)
#define PINMUX_SWJ_SWCLKTCK_SET(val)    SET_MASKED_BITS(SCU_EXTREG_SWJ_SWCLKTCK_IOCTRL, val, 0, 2)
#define PINMUX_SWJ_SWCLKTCK_GPIO5       3

#define PINMUX_SWJ_TDO_GET()            GET_BITS(SCU_EXTREG_SWJ_TDO_IOCTRL, 0, 2)
#define PINMUX_SWJ_TDO_SET(val)         SET_MASKED_BITS(SCU_EXTREG_SWJ_TDO_IOCTRL, val, 0, 2)
#define PINMUX_SWJ_TDO_GPIO6            3
#define PINMUX_SWJ_TDO_PWM4             5

#define PINMUX_LC_PCLK_GET()            GET_BITS(SCU_EXTREG_LC_PCLK_IOCTRL, 0, 2)
#define PINMUX_LC_PCLK_SET(val)         SET_MASKED_BITS(SCU_EXTREG_LC_PCLK_IOCTRL, val, 0, 2)
#define PINMUX_LC_PCLK_GPIO7            3
#define PINMUX_LC_PCLK                  0

#define PINMUX_LC_VS_GET()              GET_BITS(SCU_EXTREG_LC_VS_IOCTRL, 0, 2)
#define PINMUX_LC_VS_SET(val)           SET_MASKED_BITS(SCU_EXTREG_LC_VS_IOCTRL, val, 0, 2)
#define PINMUX_LC_VS_LCM_DB0            7
#define PINMUX_LC_VS_GPIO8              3
#define PINMUX_LC_VS                    0

#define PINMUX_LC_HS_GET()              GET_BITS(SCU_EXTREG_LC_HS_IOCTRL, 0, 2)
#define PINMUX_LC_HS_SET(val)           SET_MASKED_BITS(SCU_EXTREG_LC_HS_IOCTRL, val, 0, 2)
#define PINMUX_LC_HS_LCM_DB1            7
#define PINMUX_LC_HS_GPIO9              3
#define PINMUX_LC_HS                    0

#define PINMUX_LC_DE_GET()              GET_BITS(SCU_EXTREG_LC_DE_IOCTRL, 0, 2)
#define PINMUX_LC_DE_SET(val)           SET_MASKED_BITS(SCU_EXTREG_LC_DE_IOCTRL, val, 0, 2)
#define PINMUX_LC_DE_LCM_DB2            7
#define PINMUX_LC_DE_GPIO10             3
#define PINMUX_LC_DE                    0

#define PINMUX_LC_DATA0_GET()           GET_BITS(SCU_EXTREG_LC_DATA0_IOCTRL, 0, 2)
#define PINMUX_LC_DATA0_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA0_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA0_LCM_DB3         7
#define PINMUX_LC_DATA0_GPIO11          3
#define PINMUX_LC_DATA0                 0

#define PINMUX_LC_DATA1_GET()           GET_BITS(SCU_EXTREG_LC_DATA1_IOCTRL, 0, 2)
#define PINMUX_LC_DATA1_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA1_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA1_LCM_DB4         7
#define PINMUX_LC_DATA1_GPIO12          3
#define PINMUX_LC_DATA1                 0

#define PINMUX_LC_DATA2_GET()           GET_BITS(SCU_EXTREG_LC_DATA2_IOCTRL, 0, 2)
#define PINMUX_LC_DATA2_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA2_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA2_LCM_DB5         7
#define PINMUX_LC_DATA2_GPIO13          3
#define PINMUX_LC_DATA2                 0

#define PINMUX_LC_DATA3_GET()           GET_BITS(SCU_EXTREG_LC_DATA3_IOCTRL, 0, 2)
#define PINMUX_LC_DATA3_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA3_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA3_LCM_DB6         7
#define PINMUX_LC_DATA3_GPIO14          3
#define PINMUX_LC_DATA3                 0

#define PINMUX_LC_DATA4_GET()           GET_BITS(SCU_EXTREG_LC_DATA4_IOCTRL, 0, 2)
#define PINMUX_LC_DATA4_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA4_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA4_LCM_DB7         7
#define PINMUX_LC_DATA4_GPIO15          3
#define PINMUX_LC_DATA4                 0

#define PINMUX_LC_DATA5_GET()           GET_BITS(SCU_EXTREG_LC_DATA5_IOCTRL, 0, 2)
#define PINMUX_LC_DATA5_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA5_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA5_LCM_DB8         7
#define PINMUX_LC_DATA5_GPIO16          3
#define PINMUX_LC_DATA5                 0

#define PINMUX_LC_DATA6_GET()           GET_BITS(SCU_EXTREG_LC_DATA6_IOCTRL, 0, 2)
#define PINMUX_LC_DATA6_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA6_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA6_LCM_DB9         7
#define PINMUX_LC_DATA6_GPIO17          3
#define PINMUX_LC_DATA6_UART2_TX        1
#define PINMUX_LC_DATA6                 0

#define PINMUX_LC_DATA7_GET()           GET_BITS(SCU_EXTREG_LC_DATA7_IOCTRL, 0, 2)
#define PINMUX_LC_DATA7_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA7_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA7_LCM_DB10        7
#define PINMUX_LC_DATA7_GPIO18          3
#define PINMUX_LC_DATA7_UART2_RX        1
#define PINMUX_LC_DATA7                 0

#define PINMUX_LC_DATA8_GET()           GET_BITS(SCU_EXTREG_LC_DATA8_IOCTRL, 0, 2)
#define PINMUX_LC_DATA8_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA8_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA8_LCM_DB11        7
#define PINMUX_LC_DATA8_GPIO19          3
#define PINMUX_LC_DATA8_DPI          5
#define PINMUX_LC_DATA8                 0

#define PINMUX_LC_DATA9_GET()           GET_BITS(SCU_EXTREG_LC_DATA9_IOCTRL, 0, 2)
#define PINMUX_LC_DATA9_SET(val)        SET_MASKED_BITS(SCU_EXTREG_LC_DATA9_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA9_LCM_DB12        7
#define PINMUX_LC_DATA9_GPIO20          3
#define PINMUX_LC_DATA9_DPI          5
#define PINMUX_LC_DATA9                 0

#define PINMUX_LC_DATA10_GET()          GET_BITS(SCU_EXTREG_LC_DATA10_IOCTRL, 0, 2)
#define PINMUX_LC_DATA10_SET(val)       SET_MASKED_BITS(SCU_EXTREG_LC_DATA10_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA10_LCM_DB13       7
#define PINMUX_LC_DATA10_GPIO21         3
#define PINMUX_LC_DATA10_DPI          5
#define PINMUX_LC_DATA10                0

#define PINMUX_LC_DATA11_GET()          GET_BITS(SCU_EXTREG_LC_DATA11_IOCTRL, 0, 2)
#define PINMUX_LC_DATA11_SET(val)       SET_MASKED_BITS(SCU_EXTREG_LC_DATA11_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA11_LCM_DB14       7
#define PINMUX_LC_DATA11_GPIO0          3
#define PINMUX_LC_DATA11_DPI          5
#define PINMUX_LC_DATA11                0

#define PINMUX_LC_DATA12_GET()          GET_BITS(SCU_EXTREG_LC_DATA12_IOCTRL, 0, 2)
#define PINMUX_LC_DATA12_SET(val)       SET_MASKED_BITS(SCU_EXTREG_LC_DATA12_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA12_PWM5			2
#define PINMUX_LC_DATA12_LCM_DB15       7
#define PINMUX_LC_DATA12_GPIO1          3
#define PINMUX_LC_DATA12_DPI          5
#define PINMUX_LC_DATA12                0

#define PINMUX_LC_DATA13_GET()          GET_BITS(SCU_EXTREG_LC_DATA13_IOCTRL, 0, 2)
#define PINMUX_LC_DATA13_SET(val)       SET_MASKED_BITS(SCU_EXTREG_LC_DATA13_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA13_LCM_DB16       7
#define PINMUX_LC_DATA13_GPIO2          3
#define PINMUX_LC_DATA13_DPI          5
#define PINMUX_LC_DATA13                0

#define PINMUX_LC_DATA14_GET()          GET_BITS(SCU_EXTREG_LC_DATA14_IOCTRL, 0, 2)
#define PINMUX_LC_DATA14_SET(val)       SET_MASKED_BITS(SCU_EXTREG_LC_DATA14_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA14_LCM_DB17       7
#define PINMUX_LC_DATA14_GPIO3          3
#define PINMUX_LC_DATA14_DPI          5
#define PINMUX_LC_DATA14                0

#define PINMUX_LC_DATA15_GET()          GET_BITS(SCU_EXTREG_LC_DATA15_IOCTRL, 0, 2)
#define PINMUX_LC_DATA15_SET(val)       SET_MASKED_BITS(SCU_EXTREG_LC_DATA15_IOCTRL, val, 0, 2)
#define PINMUX_LC_DATA15_LCM_SCn        7
#define PINMUX_LC_DATA_GPIO4            3
#define PINMUX_LC_DATA15_DPI          5
#define PINMUX_LC_DATA15                0

#define PINMUX_SD_CLK_GET()             GET_BITS(SCU_EXTREG_SD_CLK_IOCTRL, 0, 2)
#define PINMUX_SD_CLK_SET(val)          SET_MASKED_BITS(SCU_EXTREG_SD_CLK_IOCTRL, val, 0, 2)
#define PINMUX_SD_CLK_LC_DATA16         1
#define PINMUX_SD_CLK_LCM_WR            7
#define PINMUX_SD_CLK_GPIO22            3
#define PINMUX_SD_CLK_DPI          5

#define PINMUX_SD_CMD_GET()             GET_BITS(SCU_EXTREG_SD_CMD_IOCTRL, 0, 2)
#define PINMUX_SD_CMD_SET(val)          SET_MASKED_BITS(SCU_EXTREG_SD_CMD_IOCTRL, val, 0, 2)
#define PINMUX_SD_CMD_LC_DATA17         1
#define PINMUX_SD_CMD_LCM_RS            7
#define PINMUX_SD_CMD_GPIO23            3
#define PINMUX_SD_CMD_DPI          5

#define PINMUX_SD_DATA0_GET()           GET_BITS(SCU_EXTREG_SD_DATA0_IOCTRL, 0, 2)
#define PINMUX_SD_DATA0_SET(val)        SET_MASKED_BITS(SCU_EXTREG_SD_DATA0_IOCTRL, val, 0, 2)
#define PINMUX_SD_DATA0_LC_DATA18       1
#define PINMUX_SD_DATA0_LCM_RD          7
#define PINMUX_SD_DATA0_GPIO24          3
#define PINMUX_SD_DATA0_DPI          5

#define PINMUX_SD_DATA1_GET()           GET_BITS(SCU_EXTREG_SD_DATA1_IOCTRL, 0, 2)
#define PINMUX_SD_DATA1_SET(val)        SET_MASKED_BITS(SCU_EXTREG_SD_DATA1_IOCTRL, val, 0, 2)
#define PINMUX_SD_DATA1_LC_DATA19       1
#define PINMUX_SD_DATA1_LCM_RESETn      7
#define PINMUX_SD_DATA1_GPIO25          3
#define PINMUX_SD_DATA1_DPI          5

#define PINMUX_SD_DATA2_GET()           GET_BITS(SCU_EXTREG_SD_DATA2_IOCTRL, 0, 2)
#define PINMUX_SD_DATA2_SET(val)        SET_MASKED_BITS(SCU_EXTREG_SD_DATA2_IOCTRL, val, 0, 2)
#define PINMUX_SD_DATA2_LC_DATA20       1
#define PINMUX_SD_DATA2_LCM_BLCTRL      7
#define PINMUX_SD_DATA2_GPIO26          3

#define PINMUX_SD_DATA3_GET()           GET_BITS(SCU_EXTREG_SD_DATA3_IOCTRL, 0, 2)
#define PINMUX_SD_DATA3_SET(val)        SET_MASKED_BITS(SCU_EXTREG_SD_DATA3_IOCTRL, val, 0, 2)
#define PINMUX_SD_DATA3_LC_DATA21       1
#define PINMUX_SD_DATA3_LCM_TP_INT1     7
#define PINMUX_SD_DATA3_GPIO27          3

#define PINMUX_UART0_RX_GET()           GET_BITS(SCU_EXTREG_UART0_RX_IOCTRL, 0, 2)
#define PINMUX_UART0_RX_SET(val)        SET_MASKED_BITS(SCU_EXTREG_UART0_RX_IOCTRL, val, 0, 2)
#define PINMUX_UART0_RX_GPIO7           3

#define PINMUX_UART0_TX_GET()           GET_BITS(SCU_EXTREG_UART0_TX_IOCTRL, 0, 2)
#define PINMUX_UART0_TX_SET(val)        SET_MASKED_BITS(SCU_EXTREG_UART0_TX_IOCTRL, val, 0, 2)
#define PINMUX_UART0_TX_GPIO28          3

#define PINMUX_I2C0_CLK_GET()           GET_BITS(SCU_EXTREG_I2C0_CLK_IOCTRL, 0, 2)
#define PINMUX_I2C0_CLK_SET(val)        SET_MASKED_BITS(SCU_EXTREG_I2C0_CLK_IOCTRL, val, 0, 2)
#define PINMUX_I2C0_CLK_GPIO29          3

#define PINMUX_I2C0_DATA_GET()          GET_BITS(SCU_EXTREG_I2C0_DATA_IOCTRL, 0, 2)
#define PINMUX_I2C0_DATA_SET(val)       SET_MASKED_BITS(SCU_EXTREG_I2C0_DATA_IOCTRL, val, 0, 2)
#define PINMUX_I2C0_DATA_GPIO30         3

#define PINMUX_PWM0_GET()               GET_BITS(SCU_EXTREG_PWM0_IOCTRL, 0, 2)
#define PINMUX_PWM0_SET(val)            SET_MASKED_BITS(SCU_EXTREG_PWM0_IOCTRL, val, 0, 2)
#define PINMUX_PWM0                     0
#define PINMUX_PWM0_GPIO31              3

#define PINMUX_OTG_DRV_VBUS_GET()       GET_BITS(SCU_EXTREG_OTG_DRV_VBUS_IOCTRL, 0, 2)
#define PINMUX_OTG_DRV_VBUS_SET(val)    SET_MASKED_BITS(SCU_EXTREG_OTG_DRV_VBUS_IOCTRL, val, 0, 2)

#define PINMUX_SPARE0_GET()             GET_BITS(SCU_EXTREG_SPARE0_IOCTRL, 0, 2)
#define PINMUX_SPARE0_SET(val)          SET_MASKED_BITS(SCU_EXTREG_SPARE0_IOCTRL, val, 0, 2)

#define PINMUX_SPARE1_GET()             GET_BITS(SCU_EXTREG_SPARE1_IOCTRL, 0, 2)
#define PINMUX_SPARE1_SET(val)          SET_MASKED_BITS(SCU_EXTREG_SPARE1_IOCTRL, val, 0, 2)

#define DEBUG_PINMUX
#ifdef DEBUG_PINMUX
void debug_pinmux(void);
#endif

void pinmux_init(void);

#endif