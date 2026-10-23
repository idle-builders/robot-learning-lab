#ifndef ROBOT_IMU_H
#define ROBOT_IMU_H
#include <stddef.h>
#include <stdint.h>

/* LSM6DSO mode 1: 7-bit I2C 0x6a, 208 Hz, +/-4 g, +/-500 deg/s.
 * Bus callbacks return 1 only for a complete transfer. No platform dependency.
 * now_us is the host poll-start clock, not the sensor conversion timestamp. */
typedef int (*ImuRead)(void*,uint8_t,uint8_t*,size_t);
typedef int (*ImuWrite)(void*,uint8_t,const uint8_t*,size_t);
typedef struct { void* context; ImuRead read; ImuWrite write; } ImuBus;
enum ImuStatus { IMU_BOOT_WAIT=0, IMU_RESET_WAIT=1, IMU_SETTLING=2, IMU_READY=3, IMU_FAULT=4 };
enum ImuError { IMU_OK=0, IMU_IO=1, IMU_ID=2, IMU_RESET_TIMEOUT=3, IMU_CONFIG=4, IMU_CLOCK=5 };
typedef struct {
    double gyro_rad_s[3],specific_force_m_s2[3];
    uint64_t poll_start_us,earliest_update_us;
    uint32_t sequence;
    uint8_t saturated;
} ImuSample;
typedef struct {
    uint32_t status,error;
    uint64_t last_poll_us,deadline_us,max_poll_gap_us,last_drain_us;
    uint8_t has_sample;
    ImuSample sample;
} Imu;
void imu_init(Imu*,uint64_t);
/* Returns 1 for a new sample, 0 for waiting/no data, -1 for a latched fault. */
int imu_poll(Imu*,const ImuBus*,uint64_t);
/* A successful older transfer must not be labeled fresh after failed IO. */
int imu_valid(const Imu*,uint64_t,uint64_t);
void imu_decode(const uint8_t[12],ImuSample*);
#endif
