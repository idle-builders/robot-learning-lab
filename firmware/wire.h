#ifndef ROBOT_WIRE_H
#define ROBOT_WIRE_H
#include <stddef.h>
#include <stdint.h>
#define WIRE_MAX_JOINTS 32
#define WIRE_MAX_FRAME (28+4*WIRE_MAX_JOINTS)
enum WireType { WIRE_COMMAND=1, WIRE_STOP=2, WIRE_ARM=3 };
typedef struct {
    uint8_t type, count;
    uint64_t sequence, session;
    float value[WIRE_MAX_JOINTS];
} WireCommand;
typedef struct { uint8_t data[WIRE_MAX_FRAME]; size_t used, expected; } WireRx;
typedef struct {
    WireRx frame;
    uint64_t started_ms, last_ms, expired;
} WireTimedRx;
uint32_t wire_crc32(const uint8_t*, size_t);
int wire_decode(const uint8_t*,size_t,WireCommand*);
int wire_push(WireRx*,uint8_t,WireCommand*);
int wire_timed_expire(WireTimedRx*,uint64_t,uint32_t);
int wire_timed_push(WireTimedRx*,uint8_t,uint64_t,uint32_t,WireCommand*);
size_t servo_write(uint8_t,uint8_t,const uint8_t*,size_t,uint8_t*,size_t);
size_t servo_read(uint8_t,uint8_t,uint8_t,uint8_t*,size_t);
size_t servo_sync7(const uint8_t*,const uint8_t*,size_t,uint8_t*,size_t);
int servo_status(const uint8_t*,size_t,uint8_t,uint8_t*,size_t,uint8_t*);
int16_t servo_signed(uint16_t);
uint16_t servo_signmag(int16_t);
#endif
