#include "supervisor.h"
#include <math.h>
#include <string.h>
static double clamp(double x, double lo, double hi) { return fmax(lo,fmin(x,hi)); }
int supervisor_init(Supervisor* s, uint32_t n, uint32_t timeout, double vmin,
                    double kp, double kd, const double* lo, const double* hi,
                    const double* effort, const double* speed) {
    memset(s,0,sizeof(*s));
    if (!n || n>ROBOT_MAX_JOINTS || !timeout || !isfinite(vmin) || vmin<=0 ||
        !isfinite(kp) || kp<=0 || !isfinite(kd) || kd<0) { s->status=FAULT; return 0; }
    s->n=n; s->timeout_ms=timeout; s->minimum_voltage=vmin; s->kp=kp; s->kd=kd;
    for (uint32_t i=0;i<n;i++) {
        if (!isfinite(lo[i]) || !isfinite(hi[i]) || lo[i]>=hi[i] ||
            !isfinite(effort[i]) || effort[i]<=0 || !isfinite(speed[i]) || speed[i]<=0) {
            s->status=FAULT; return 0;
        }
        s->lower[i]=lo[i]; s->upper[i]=hi[i]; s->effort[i]=effort[i]; s->speed[i]=speed[i];
    }
    return 1;
}
int supervisor_command(Supervisor* s, uint64_t seq, uint64_t now, const double* q) {
    if (s->status==ESTOP || s->status==FAULT) return 0;
    if (s->has_command && (seq<=s->sequence || now<s->received_ms)) return 0;
    for (uint32_t i=0;i<s->n;i++) {
        if (!isfinite(q[i]) || q[i]<s->lower[i] || q[i]>s->upper[i]) {
            s->status=FAULT; return 0;
        }
    }
    memcpy(s->target,q,s->n*sizeof(double));
    s->sequence=seq; s->received_ms=now; s->has_command=1; s->status=ACTIVE;
    return 1;
}
uint32_t supervisor_check(Supervisor* s, uint64_t now, double voltage, int estop) {
    if (estop) s->status=ESTOP;
    if (s->status==ESTOP || s->status==FAULT) return s->status;
    if (!isfinite(voltage) || voltage<s->minimum_voltage) { s->status=FAULT; return s->status; }
    if (!s->has_command) return s->status;
    if (now<s->received_ms || now-s->received_ms>=s->timeout_ms) {
        s->status=TIMED_OUT; return s->status;
    }
    return s->status;
}
uint32_t supervisor_tick(Supervisor* s, uint64_t now, double voltage, int estop,
                         const double* q, const double* dq, double* output) {
    return supervisor_tick_mixed(s,now,voltage,estop,q,dq,NULL,1,output);
}
uint32_t supervisor_tick_mixed(Supervisor* s, uint64_t now, double voltage, int estop,
                         const double* q, const double* dq, const uint32_t* velocity_mode,
                         double velocity_gain, double* output) {
    for (uint32_t i=0;i<s->n;i++) output[i]=0;
    if (supervisor_check(s,now,voltage,estop)!=ACTIVE) return s->status;
    if (!isfinite(velocity_gain) || velocity_gain<=0) { s->status=FAULT; return s->status; }
    for (uint32_t i=0;i<s->n;i++) {
        if (!isfinite(q[i]) || !isfinite(dq[i]) || (velocity_mode && velocity_mode[i]>1)) {
            s->status=FAULT; return s->status;
        }
    }
    for (uint32_t i=0;i<s->n;i++) {
        double demand=(velocity_mode && velocity_mode[i]) ? velocity_gain*(s->target[i]-dq[i]) :
            s->kp*(s->target[i]-q[i])-s->kd*dq[i];
        double u=clamp(demand,-s->effort[i],s->effort[i]);
        if (fabs(dq[i])>=s->speed[i] && u*dq[i]>0) u=0;
        output[i]=u;
    }
    return s->status;
}
