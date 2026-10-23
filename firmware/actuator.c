#include "actuator.h"
#include <math.h>
#include <string.h>
#define TICKS_PER_RAD (4096.0/6.2831853071795864769)
int actuator_init(ActuatorGuard* a,const JointMap* map,unsigned n,unsigned timeout,double vmin,uint64_t session,int calibrated) {
    memset(a,0,sizeof(*a));a->session=session;a->calibrated=!!calibrated;
    if(!n || n>ROBOT_MAX_JOINTS || !session) return 0;
    double lo[ROBOT_MAX_JOINTS],hi[ROBOT_MAX_JOINTS],effort[ROBOT_MAX_JOINTS],speed[ROBOT_MAX_JOINTS];
    for(unsigned i=0;i<n;i++) {
        if(map[i].bus>1 || map[i].id>253 || map[i].mode>1 || (map[i].direction!=1 && map[i].direction!=-1) || map[i].zero>4095) return 0;
        if(!isfinite(map[i].speed) || map[i].speed*TICKS_PER_RAD<1 || map[i].speed*TICKS_PER_RAD>32767) return 0;
        for(unsigned j=0;j<i;j++) if(map[i].bus==map[j].bus && map[i].id==map[j].id) return 0;
        lo[i]=map[i].mode ? -map[i].speed : map[i].lower;
        hi[i]=map[i].mode ? map[i].speed : map[i].upper;
        if(!map[i].mode && (map[i].zero+fmin(map[i].lower*map[i].direction,map[i].upper*map[i].direction)*TICKS_PER_RAD<0 || map[i].zero+fmax(map[i].lower*map[i].direction,map[i].upper*map[i].direction)*TICKS_PER_RAD>4095)) return 0;
        effort[i]=1;speed[i]=map[i].speed;
    }
    return supervisor_init(&a->guard,n,timeout,vmin,1,0,lo,hi,effort,speed);
}
int actuator_command(ActuatorGuard* a,const JointMap* map,const WireCommand* c,uint64_t now,int arm_switch,int stop_ok) {
    (void)map;
    if(c->session!=a->session || (a->sequence_seen && c->sequence<=a->last_sequence)) return 0;
    /* Expiry must also be checked before accepting a command between control ticks. */
    if(a->armed && (now<a->guard.received_ms || now-a->guard.received_ms>=a->guard.timeout_ms)) {
        a->armed=0;
        if(a->guard.status!=FAULT && a->guard.status!=ESTOP) a->guard.status=TIMED_OUT;
    }
    if(c->type==WIRE_STOP) { a->armed=0;a->guard.status=ESTOP; }
    else if(c->type==WIRE_ARM) {
        if(!a->calibrated || !arm_switch || !stop_ok || a->guard.status==ESTOP || a->guard.status==FAULT) return 0;
        a->armed=1;a->guard.has_command=0;a->guard.status=DISARMED;
        a->guard.received_ms=now;
    } else if(c->type==WIRE_COMMAND) {
        if(!a->armed || !arm_switch || !stop_ok || c->count!=a->guard.n) return 0;
        double values[ROBOT_MAX_JOINTS];
        for(unsigned i=0;i<c->count;i++) {
            double value=c->value[i];
            /* RLB1 endpoints are binary32. Restore the exact configured bound
             * only when its nearest wire representation rounded outward.
             * The next float beyond that representation still faults. */
            if(value<a->guard.lower[i] && c->value[i]==(float)a->guard.lower[i]) value=a->guard.lower[i];
            if(value>a->guard.upper[i] && c->value[i]==(float)a->guard.upper[i]) value=a->guard.upper[i];
            values[i]=value;
        }
        if(!supervisor_command(&a->guard,c->sequence,now,values)) { a->armed=0;return 0; }
    } else return 0;
    a->last_sequence=c->sequence;a->sequence_seen=1;return 1;
}
uint32_t actuator_check(ActuatorGuard* a,uint64_t now,double voltage,int arm_switch,int stop_ok) {
    uint32_t status=supervisor_check(&a->guard,now,voltage,!stop_ok);
    if(a->armed && (!arm_switch || now<a->guard.received_ms || now-a->guard.received_ms>=a->guard.timeout_ms)) {
        a->armed=0;if(status!=FAULT && status!=ESTOP) a->guard.status=TIMED_OUT;
    }
    if(a->guard.status==FAULT || a->guard.status==ESTOP || a->guard.status==TIMED_OUT) a->armed=0;
    return a->guard.status;
}
static void put16(uint8_t* p,uint16_t v) { p[0]=(uint8_t)v;p[1]=(uint8_t)(v>>8); }
int actuator_rows(ActuatorGuard* a,const JointMap* map,double dt,unsigned bus,uint8_t* ids,uint8_t* rows,size_t* count) {
    *count=0;
    if(!a->armed || a->guard.status!=ACTIVE || !isfinite(dt) || dt<=0 || dt>.1 || bus>1) return 0;
    for(unsigned i=0;i<a->guard.n;i++) if(map[i].bus==bus) {
        double value=a->guard.target[i];
        if(!map[i].mode) {
            double delta=map[i].speed*dt;
            value=fmax(a->sent[i]-delta,fmin(a->sent[i]+delta,value));
            a->sent[i]=value;
        }
        int ticks=map[i].mode ? 0 : (int)lround(map[i].zero+map[i].direction*value*TICKS_PER_RAD);
        int speed=(int)lround((map[i].mode ? map[i].direction*value : map[i].speed)*TICKS_PER_RAD);
        if(ticks<0 || ticks>4095 || speed<-32767 || speed>32767) { a->armed=0;a->guard.status=FAULT;return 0; }
        uint8_t* row=rows+7*(*count);ids[*count]=map[i].id;
        row[0]=20;put16(row+1,(uint16_t)ticks);put16(row+3,0);put16(row+5,servo_signmag((int16_t)speed));
        (*count)++;
    }
    return *count>0;
}
