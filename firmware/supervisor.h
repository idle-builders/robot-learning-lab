#ifndef ROBOT_SUPERVISOR_H
#define ROBOT_SUPERVISOR_H
#include <stdint.h>
#define ROBOT_MAX_JOINTS 32
enum RobotStatus { DISARMED=0, ACTIVE=1, TIMED_OUT=2, ESTOP=3, FAULT=4 };
typedef struct {
    uint32_t n, timeout_ms, status, has_command;
    uint64_t sequence, received_ms;
    double minimum_voltage, kp, kd;
    double lower[ROBOT_MAX_JOINTS], upper[ROBOT_MAX_JOINTS];
    double effort[ROBOT_MAX_JOINTS], speed[ROBOT_MAX_JOINTS];
    double target[ROBOT_MAX_JOINTS];
} Supervisor;
int supervisor_init(Supervisor*, uint32_t, uint32_t, double, double, double,
                    const double*, const double*, const double*, const double*);
int supervisor_command(Supervisor*, uint64_t, uint64_t, const double*);
uint32_t supervisor_check(Supervisor*, uint64_t, double, int);
uint32_t supervisor_tick(Supervisor*, uint64_t, double, int, const double*, const double*, double*);
/* Simulator smart-servo approximation: flagged targets are rad/s, others rad. */
uint32_t supervisor_tick_mixed(Supervisor*, uint64_t, double, int, const double*, const double*,
                               const uint32_t*, double, double*);
#endif
