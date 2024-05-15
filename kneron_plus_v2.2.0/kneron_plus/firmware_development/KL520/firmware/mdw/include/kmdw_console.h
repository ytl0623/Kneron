/**
 * @file        kmdw_console.h
 * @brief       log message to console APIs
 *
 * @copyright   Copyright (c) 2020 Kneron Inc. All rights reserved.
 */

#ifndef __KMDW_CONSOLE_H__
#define __KMDW_CONSOLE_H__

#include <stdio.h>
#include "kmdw_status.h"
#include "kdrv_uart.h"
#include "kmdw_memory.h"

#define LOG_NONE        0       /**< log level for none */
#define LOG_CRITICAL    1       /**< log level for critical */
#define LOG_ERROR       2       /**< log level for error */
#define LOG_USER        3       /**< log level for user */

#define LOG_INFO        4       /**< log level for info */
#define LOG_TRACE       5       /**< log level for trace */
#define LOG_DBG         6       /**< log level for dbg */

#define LOG_PROFILE     9       /**< log level for profile */

typedef void (*print_callback)(const char *log);

extern void kmdw_console_set_log_level_scpu(uint32_t level);
extern void kmdw_console_set_log_level_ncpu(uint32_t level);
extern char kmdw_console_getc(void);
extern void kmdw_console_putc(char Ch);
extern void kmdw_console_puts(char *str);
extern int kmdw_console_echo_gets(char *buf, int len);
extern void kmdw_console_hook_callback(print_callback print_cb);
extern void kmdw_console_wait_rx_done(kdrv_uart_handle_t handle);
extern void kmdw_console_wait_tx_done(kdrv_uart_handle_t handle);
extern kmdw_status_t kmdw_uart_console_init(uint8_t uart_dev, uint32_t baudrate);
extern kmdw_status_t kmdw_uart_uninitialize(void);
extern uint32_t kmdw_console_get_log_level_scpu(void);

extern kmdw_status_t kmdw_printf(const char *fmt, ...);

extern kmdw_status_t kmdw_level_printf(int level, const char *fmt, ...);

extern bool ModelFromDDR; //check model is from flash or ddr

#ifdef LOG_ENABLE

#define DSG(__format__, ...)                         \
    {                                                \
        kmdw_printf(__format__ "\n", ##__VA_ARGS__); \
    }

#define DSG_NOLF(__format__, ...)               \
    {                                           \
        kmdw_printf(__format__, ##__VA_ARGS__); \
    }

#define dbg_msg(fmt, ...) kmdw_level_printf(LOG_DBG, fmt, ##__VA_ARGS__)
#define trace_msg(fmt, ...) kmdw_level_printf(LOG_TRACE, fmt, ##__VA_ARGS__)
#define info_msg(fmt, ...) kmdw_level_printf(LOG_INFO, fmt, ##__VA_ARGS__)
#define err_msg(fmt, ...) kmdw_level_printf(LOG_ERROR, fmt, ##__VA_ARGS__)
#define critical_msg(fmt, ...) kmdw_level_printf(LOG_CRITICAL, fmt, ##__VA_ARGS__)
#define profile_msg(fmt, ...) kmdw_level_printf(LOG_PROFILE, fmt, ##__VA_ARGS__)

#define dlog(fmt, ...) kmdw_level_printf(LOG_DBG, "[%s][%s] " fmt "\r\n", DEF_LOG_CATEG, __func__, ##__VA_ARGS__)

#else //////////////////////////////////////////////////////////////////////

#define DSG_NOLF(__format__, ...)
#define DSG(__format__, ...)
#define dbg_msg(fmt, ...)
#define trace_msg(fmt, ...)
#define info_msg(fmt, ...)
#define err_msg(fmt, ...)
#define critical_msg(fmt, ...)
#define dlog(fmt, ...)

#endif // LOG_ENABLE ///////////////////////////////////////////////////////

#define dbg_msg_api(fmt, ...)       dbg_msg(fmt, ##__VA_ARGS__)
#define dbg_msg_app(fmt, ...)       dbg_msg(fmt, ##__VA_ARGS__)
#define dbg_msg_algo(fmt, ...)      critical_msg(fmt, ##__VA_ARGS__)
#define dbg_msg_console(fmt, ...)   critical_msg(fmt "\n", ##__VA_ARGS__)
#define dbg_msg_user(fmt, ...)      kmdw_level_printf(LOG_USER, fmt, ##__VA_ARGS__)

#endif // __KMDW_CONSOLE_H__
