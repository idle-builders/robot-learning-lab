/* Supported-rig controller candidate. No Wi-Fi, balance or physical qualification. */
#include "pico/stdlib.h"
#include "pico/rand.h"
#include "hardware/uart.h"
#include "hardware/adc.h"
#include "hardware/watchdog.h"
#include "hardware/i2c.h"
#include "actuator.h"
#include "imu.h"
#include "robot_map.h"
#include <stdio.h>
#include <math.h>

static uart_inst_t* buses[2]={uart0,uart1};
static const unsigned oe[2]={PIN_BUS0_OE_N,PIN_BUS1_OE_N};
static ActuatorGuard controller;
static uint8_t feedback[ROBOT_JOINT_COUNT][15];
static uint64_t feedback_ms[ROBOT_JOINT_COUNT];
static WireTimedRx receiver;
static bool was_active;
static bool power_requested;
static uint64_t power_requested_ms;
static double voltage;
static uint64_t rejected;
static Imu imu;
static uint64_t imu_read_end_us;
static int imu_read(void* context,uint8_t reg,uint8_t* data,size_t n) {
    (void)context;
    absolute_time_t deadline=make_timeout_time_us(IMU_TRANSFER_TIMEOUT_US);
    return i2c_write_blocking_until(i2c0,IMU_ADDRESS,&reg,1,true,deadline)==1
        && i2c_read_blocking_until(i2c0,IMU_ADDRESS,data,n,false,deadline)==(int)n;
}
static int imu_write(void* context,uint8_t reg,const uint8_t* data,size_t n) {
    (void)context;
    if(n>16) return 0;
    uint8_t packet[17];packet[0]=reg;
    for(size_t i=0;i<n;i++) packet[i+1]=data[i];
    return i2c_write_timeout_us(i2c0,IMU_ADDRESS,packet,n+1,false,IMU_TRANSFER_TIMEOUT_US)==(int)(n+1);
}
static const ImuBus imu_bus={NULL,imu_read,imu_write};
static void send(unsigned bus,const uint8_t* p,size_t n) {
    while(uart_is_readable(buses[bus])) (void)uart_getc(buses[bus]);
    gpio_put(oe[bus],0);
    uart_write_blocking(buses[bus],p,n);uart_tx_wait_blocking(buses[bus]);
    while(uart_is_readable(buses[bus])) (void)uart_getc(buses[bus]); /* discard local echo */
    gpio_put(oe[bus],1);
}
static bool exchange(unsigned bus,const uint8_t* p,size_t n,uint8_t id,uint8_t* data,size_t count) {
    uint8_t reply[32],error=0;size_t used=0;
    if(count+6>sizeof(reply)) return false;
    send(bus,p,n);
    uint64_t deadline=time_us_64()+SERVO_TIMEOUT_US;
    while(used<count+6 && time_us_64()<deadline) {
        if(uart_is_readable(buses[bus])) reply[used++]=(uint8_t)uart_getc(buses[bus]);
    }
    return servo_status(reply,used,id,data,count,&error) && !error;
}
static void broadcast_torque_off(void) {
    uint8_t p[8],zero=0;
    size_t n=servo_write(254,40,&zero,1,p,sizeof(p));
    for(unsigned bus=0;bus<2;bus++) send(bus,p,n);
}
static void torque_off(void) {
    /* Do not put UART operations ahead of removal of the power request. */
    gpio_put(PIN_MOTION_OK_OUT,0);
    power_requested=false;was_active=false;
    broadcast_torque_off(); /* Best effort while residual rail energy remains. */
}
static double read_voltage(void) { return adc_read()*ADC_REFERENCE_V*DIVIDER_RATIO/4095.0; }
static bool drive_allowed(void) {
    return actuator_check(&controller,to_ms_64_since_boot(get_absolute_time()),read_voltage(),
        gpio_get(PIN_ARM_SWITCH),gpio_get(PIN_STOP_CHAIN_OK))==ACTIVE && controller.armed;
}
static void fail_drive(void) {
    /* Preserve an already detected stop/timeout instead of hiding it as a bus error. */
    if(controller.guard.status==ACTIVE) controller.guard.status=FAULT;
    controller.armed=0;torque_off();
}
static bool read_all(bool check_modes) {
    for(unsigned i=0;i<ROBOT_JOINT_COUNT;i++) {
        const JointMap* m=&robot_map[i];uint8_t p[8],mode;
        if(!drive_allowed()) return false;
        if(check_modes) {
            size_t n=servo_read(m->id,33,1,p,sizeof(p));
            if(!exchange(m->bus,p,n,m->id,&mode,1) || mode!=m->mode) return false;
            if(!drive_allowed()) return false;
            n=servo_read(m->id,40,1,p,sizeof(p));
            if(!exchange(m->bus,p,n,m->id,&mode,1) || mode!=0) return false;
            if(!drive_allowed()) return false;
        }
        size_t n=servo_read(m->id,56,15,p,sizeof(p));
        if(!exchange(m->bus,p,n,m->id,feedback[i],15)) return false;
        feedback_ms[i]=to_ms_64_since_boot(get_absolute_time());
        if(feedback[i][7]>=60 || feedback[i][9]) return false; /* temperature and servo status */
        if(check_modes) {
            uint16_t raw=feedback[i][0]|((uint16_t)feedback[i][1]<<8);
            double q=(servo_signed(raw)-m->zero)*(6.2831853071795864769/4096.0)*m->direction;
            if(!m->mode && (q<m->lower-.05 || q>m->upper+.05)) return false;
            controller.sent[i]=q;
        }
        if(!gpio_get(PIN_STOP_CHAIN_OK) || !gpio_get(PIN_ARM_SWITCH)) return false;
    }
    return true;
}
static bool enable_known_servos(void) {
    for(unsigned i=0;i<ROBOT_JOINT_COUNT;i++) {
        if(!drive_allowed()) return false;
        uint8_t p[8],dummy=0,one=1;const JointMap* m=&robot_map[i];
        size_t n=servo_write(m->id,40,&one,1,p,sizeof(p));
        if(!exchange(m->bus,p,n,m->id,&dummy,0)) return false;
    }
    return drive_allowed();
}
static void control_cycle(uint64_t now) {
    voltage=read_voltage();
    uint32_t status=actuator_check(&controller,now,voltage,gpio_get(PIN_ARM_SWITCH),gpio_get(PIN_STOP_CHAIN_OK));
    if(status!=ACTIVE || !controller.armed) { torque_off();return; }
    if(!power_requested) {
        /* Power must precede replies. This output requests power; it is not proof
         * of rail readiness or actuator motion. The independent interlock may refuse it. */
        power_requested=true;power_requested_ms=now;
        gpio_put(PIN_MOTION_OK_OUT,1);return;
    }
    if(now<power_requested_ms) { fail_drive();return; }
    if(now-power_requested_ms<MOTOR_POWER_SETTLE_MS) return;
    if(!was_active) broadcast_torque_off();
    if(!read_all(!was_active)) { fail_drive();return; }
    now=to_ms_64_since_boot(get_absolute_time());
    if(actuator_check(&controller,now,read_voltage(),gpio_get(PIN_ARM_SWITCH),gpio_get(PIN_STOP_CHAIN_OK))!=ACTIVE) { torque_off();return; }
    for(unsigned bus=0;bus<2;bus++) {
        if(!drive_allowed()) { torque_off();return; }
        uint8_t ids[32],rows[224],packet[264];size_t count=0;
        if(!actuator_rows(&controller,robot_map,1.0/COMMAND_HZ,bus,ids,rows,&count)) { torque_off();return; }
        size_t n=servo_sync7(ids,rows,count,packet,sizeof(packet));
        if(!n) { controller.guard.status=FAULT;controller.armed=0;torque_off();return; }
        send(bus,packet,n);
    }
    if(!was_active && !enable_known_servos()) { fail_drive();return; }
    if(!drive_allowed()) { torque_off();return; }
    was_active=true;
}
static void telemetry(uint64_t now) {
    printf("{\"schema\":1,\"robot\":\"%s\",\"map\":\"%s\",\"session\":\"%016llx\",\"time_ms\":%llu,\"status\":%u,\"armed\":%u,\"calibrated\":%u,\"voltage_v\":%.3f,\"rejected\":%llu,\"raw_feedback\":[",ROBOT_NAME,ROBOT_MAP_HASH,(unsigned long long)controller.session,(unsigned long long)now,(unsigned)controller.guard.status,(unsigned)controller.armed,(unsigned)controller.calibrated,voltage,(unsigned long long)rejected);
    for(unsigned i=0;i<ROBOT_JOINT_COUNT;i++) {
        printf("%s{\"id\":%u,\"sample_ms\":%llu,\"bytes\":[",i?",":"",robot_map[i].id,(unsigned long long)feedback_ms[i]);
        for(unsigned j=0;j<15;j++) printf("%s%u",j?",":"",feedback[i][j]);
        printf("]}");
    }
    printf("],\"rx_expired\":%llu",(unsigned long long)receiver.expired);
    printf(",\"imu\":{\"part\":\"LSM6DSOTR\",\"config\":\"%s\",\"frame\":\"sensor\",\"calibrated\":false,\"status\":%u,\"error\":%u,\"valid\":%s,\"sequence\":%u,\"poll_start_us\":%llu,\"earliest_update_us\":%llu,\"read_end_us\":%llu,\"max_poll_gap_us\":%llu,\"saturated\":%u,\"gyro_rad_s\":[%.7f,%.7f,%.7f],\"specific_force_m_s2\":[%.7f,%.7f,%.7f]}}\n",
        IMU_CONFIG_HASH,(unsigned)imu.status,(unsigned)imu.error,imu_valid(&imu,time_us_64(),IMU_FRESHNESS_US)?"true":"false",
        (unsigned)imu.sample.sequence,(unsigned long long)imu.sample.poll_start_us,(unsigned long long)imu.sample.earliest_update_us,(unsigned long long)imu_read_end_us,
        (unsigned long long)imu.max_poll_gap_us,(unsigned)imu.sample.saturated,
        imu.sample.gyro_rad_s[0],imu.sample.gyro_rad_s[1],imu.sample.gyro_rad_s[2],
        imu.sample.specific_force_m_s2[0],imu.sample.specific_force_m_s2[1],imu.sample.specific_force_m_s2[2]);
}
int main(void) {
    /* External pull-ups on OE and pull-down on permit remain required during reset. */
    for(unsigned b=0;b<2;b++) { gpio_init(oe[b]);gpio_put(oe[b],1);gpio_set_dir(oe[b],GPIO_OUT); }
    gpio_init(PIN_MOTION_OK_OUT);gpio_put(PIN_MOTION_OK_OUT,0);gpio_set_dir(PIN_MOTION_OK_OUT,GPIO_OUT);
    gpio_init(PIN_STOP_CHAIN_OK);gpio_pull_down(PIN_STOP_CHAIN_OK);
    gpio_init(PIN_ARM_SWITCH);gpio_pull_down(PIN_ARM_SWITCH);
    const unsigned tx[2]={PIN_BUS0_TX,PIN_BUS1_TX},rx[2]={PIN_BUS0_RX,PIN_BUS1_RX};
    for(unsigned b=0;b<2;b++) {
        uart_init(buses[b],SERVO_BAUD);gpio_set_function(tx[b],GPIO_FUNC_UART);gpio_set_function(rx[b],GPIO_FUNC_UART);
        uart_set_format(buses[b],8,1,UART_PARITY_NONE);uart_set_hw_flow(buses[b],false,false);
    }
    adc_init();adc_gpio_init(PIN_BATTERY_ADC);adc_select_input(PIN_BATTERY_ADC-26);
    i2c_init(i2c0,IMU_BUS_HZ);
    gpio_set_function(PIN_IMU_SDA_RESERVED,GPIO_FUNC_I2C);
    gpio_set_function(PIN_IMU_SCL_RESERVED,GPIO_FUNC_I2C);
    /* Only carrier 4k7 pull-ups; keep INT1 low/high-Z at sensor power-up. */
    gpio_disable_pulls(PIN_IMU_SDA_RESERVED);gpio_disable_pulls(PIN_IMU_SCL_RESERVED);
    gpio_init(PIN_IMU_IRQ_RESERVED);gpio_disable_pulls(PIN_IMU_IRQ_RESERVED);
    imu_init(&imu,time_us_64());
    stdio_init_all();
    uint64_t session=get_rand_64();if(!session) session=1;
    torque_off();
    if(!actuator_init(&controller,robot_map,ROBOT_JOINT_COUNT,COMMAND_TIMEOUT_MS,MINIMUM_BUS_V,session,ROBOT_CALIBRATED)) while(true) tight_loop_contents();
    torque_off();watchdog_enable(100,true);
    uint64_t next=0,next_report=0,next_imu=0;
    while(true) {
        uint64_t now=to_ms_64_since_boot(get_absolute_time());
        WireCommand command;
        (void)wire_timed_expire(&receiver,now,HOST_FRAME_TIMEOUT_MS);
        for(unsigned i=0;i<128;i++) {
            int byte=getchar_timeout_us(0);if(byte==PICO_ERROR_TIMEOUT) break;
            now=to_ms_64_since_boot(get_absolute_time());
            if(wire_timed_push(&receiver,(uint8_t)byte,now,HOST_FRAME_TIMEOUT_MS,&command) && !actuator_command(&controller,robot_map,&command,now,gpio_get(PIN_ARM_SWITCH),gpio_get(PIN_STOP_CHAIN_OK))) rejected++;
        }
        now=to_ms_64_since_boot(get_absolute_time());
        if(!gpio_get(PIN_STOP_CHAIN_OK) || !gpio_get(PIN_ARM_SWITCH)) {
            (void)actuator_check(&controller,now,read_voltage(),gpio_get(PIN_ARM_SWITCH),gpio_get(PIN_STOP_CHAIN_OK));
            if(power_requested) torque_off();
        }
        if(now>=next) { control_cycle(now);next=now+1000/COMMAND_HZ; }
        uint64_t now_us=time_us_64();
        if(now_us>=next_imu) {
            if(imu_poll(&imu,&imu_bus,now_us)==1) imu_read_end_us=time_us_64();
            next_imu=now_us+IMU_POLL_US;
        }
        if(now>=next_report) { telemetry(now);next_report=now+100; }
        watchdog_update();sleep_us(500);
    }
}
