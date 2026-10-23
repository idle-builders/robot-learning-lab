#ifndef ROBOT_ACTUATOR_H
#define ROBOT_ACTUATOR_H
#include "wire.h"
#include "supervisor.h"
typedef struct {
    uint8_t id,bus,mode; /* mode 0 position; mode 1 wheel velocity */
    int8_t direction;
    uint16_t zero;
    double lower,upper,speed;
} JointMap;
typedef struct {
    Supervisor guard;
    uint64_t session,last_sequence;
    uint8_t sequence_seen,armed,calibrated;
    double sent[ROBOT_MAX_JOINTS];
} ActuatorGuard;
int actuator_init(ActuatorGuard*,const JointMap*,unsigned,unsigned,double,uint64_t,int);
int actuator_command(ActuatorGuard*,const JointMap*,const WireCommand*,uint64_t,int,int);
uint32_t actuator_check(ActuatorGuard*,uint64_t,double,int,int);
int actuator_rows(ActuatorGuard*,const JointMap*,double,unsigned,uint8_t*,uint8_t*,size_t*);
#endif
