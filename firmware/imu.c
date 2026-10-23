/* Independent minimal driver, register facts from ST DS12140 Rev3.
 * No FIFO or synchronization claim: BDU protects bytes, not simultaneous axes.
 * Device/sample callbacks are bounded by the target HAL, never retried forever. */
#include "imu.h"
#include <string.h>
#define WHO_AM_I 0x0f
#define CTRL3_C 0x12
#define STATUS_REG 0x1e
#define OUTX_L_G 0x22
static const uint8_t settings[][2]={
    {0x12,0x44}, /* BDU + IF_INC */
    {0x18,0xe2}, /* preserve default DEN axis bits; disable unused I3C */
    {0x13,0x08}, /* mask interrupt data-ready during filter settling */
    {0x0d,0x03}, /* accelerometer and gyro data-ready on INT1 */
    {0x10,0x58}, /* accelerometer 208 Hz, +/-4g, high-performance reset default */
    {0x11,0x54}, /* gyroscope 208 Hz, +/-500dps */
};
static int fail(Imu* s,unsigned error) { s->status=IMU_FAULT;s->error=error;return -1; }
void imu_init(Imu* s,uint64_t now) {
    memset(s,0,sizeof(*s));s->last_poll_us=now;s->deadline_us=now+100000;
}
static int16_t signed16(const uint8_t* p) {
    unsigned v=(unsigned)p[0]|((unsigned)p[1]<<8);
    return (int16_t)(v>=32768 ? (int)v-65536 : (int)v);
}
void imu_decode(const uint8_t bytes[12],ImuSample* out) {
    out->saturated=0;
    for(unsigned i=0;i<3;i++) {
        int16_t g=signed16(bytes+2*i),a=signed16(bytes+6+2*i);
        out->gyro_rad_s[i]=g*0.0175*(3.14159265358979323846/180.0);
        out->specific_force_m_s2[i]=a*0.000122*9.80665;
        if(g==-32768 || g==32767 || a==-32768 || a==32767) out->saturated=1;
    }
}
int imu_poll(Imu* s,const ImuBus* bus,uint64_t now) {
    if(s->status==IMU_FAULT) return -1;
    if(now<s->last_poll_us) return fail(s,IMU_CLOCK);
    uint64_t gap=now-s->last_poll_us;
    if(gap>s->max_poll_gap_us) s->max_poll_gap_us=gap;
    s->last_poll_us=now;
    if(!bus || !bus->read || !bus->write) return fail(s,IMU_IO);
    uint8_t value;
    if(s->status==IMU_BOOT_WAIT) {
        if(now<s->deadline_us) return 0;
        if(!bus->read(bus->context,WHO_AM_I,&value,1)) return fail(s,IMU_IO);
        if(value!=0x6c) return fail(s,IMU_ID);
        value=1;
        if(!bus->write(bus->context,CTRL3_C,&value,1)) return fail(s,IMU_IO);
        s->status=IMU_RESET_WAIT;s->deadline_us=now+100000;return 0;
    }
    if(s->status==IMU_RESET_WAIT) {
        if(now>=s->deadline_us) return fail(s,IMU_RESET_TIMEOUT);
        if(!bus->read(bus->context,CTRL3_C,&value,1)) return fail(s,IMU_IO);
        if(value&1) return 0;
        for(unsigned i=0;i<sizeof(settings)/sizeof(settings[0]);i++) {
            if(!bus->write(bus->context,settings[i][0],&settings[i][1],1)
                || !bus->read(bus->context,settings[i][0],&value,1)) return fail(s,IMU_IO);
            if(value!=settings[i][1]) return fail(s,IMU_CONFIG);
        }
        s->status=IMU_SETTLING;s->deadline_us=now+100000;return 0;
    }
    if(s->status==IMU_SETTLING) {
        if(now<s->deadline_us) return 0;
        /* Discard any values captured during the project settling allocation. */
        uint8_t discard[12];
        if(!bus->read(bus->context,OUTX_L_G,discard,sizeof(discard))) return fail(s,IMU_IO);
        s->last_drain_us=now;
        s->status=IMU_READY;return 0;
    }
    if(!bus->read(bus->context,STATUS_REG,&value,1)) return fail(s,IMU_IO);
    if((value&3)!=3) return 0;
    uint8_t bytes[12];
    if(!bus->read(bus->context,OUTX_L_G,bytes,sizeof(bytes))) return fail(s,IMU_IO);
    ImuSample candidate={0};
    imu_decode(bytes,&candidate);
    candidate.poll_start_us=now;candidate.sequence=s->sample.sequence+1;
    /* BDU may hold old words across a polling gap. Their earliest possible
     * register update is after the preceding complete drain started. This is
     * conservative; it is neither a conversion timestamp nor a filter-delay bound. */
    candidate.earliest_update_us=s->last_drain_us;s->last_drain_us=now;
    s->sample=candidate;s->has_sample=1;
    return 1;
}
int imu_valid(const Imu* s,uint64_t now,uint64_t max_age) {
    return s->status==IMU_READY && s->has_sample && !s->sample.saturated
        && now>=s->sample.poll_start_us && now>=s->sample.earliest_update_us
        && now-s->sample.earliest_update_us<max_age;
}
