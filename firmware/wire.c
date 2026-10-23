/* RLB1 is our transport. Feetech packet layout follows the manufacturer's SDK;
 * protocol implementation is independent, with no Arduino runtime dependency. */
#include "wire.h"
#include <string.h>
#include <math.h>
_Static_assert(sizeof(float)==4,"wire requires binary32 floats");
static uint16_t u16(const uint8_t* p) { return (uint16_t)p[0]|((uint16_t)p[1]<<8); }
static uint32_t u32(const uint8_t* p) { return (uint32_t)u16(p)|((uint32_t)u16(p+2)<<16); }
static uint64_t u64(const uint8_t* p) { return (uint64_t)u32(p)|((uint64_t)u32(p+4)<<32); }
uint32_t wire_crc32(const uint8_t* p,size_t n) {
    uint32_t crc=0xffffffffu;
    for (size_t i=0;i<n;i++) {
        crc^=p[i];
        for (unsigned b=0;b<8;b++) crc=(crc>>1)^(0xedb88320u & (0u-(crc&1u)));
    }
    return ~crc;
}
int wire_decode(const uint8_t* p,size_t n,WireCommand* out) {
    if (!p || !out || n<28 || n>WIRE_MAX_FRAME || memcmp(p,"RLB1",4)) return 0;
    uint8_t type=p[4],count=p[5];
    size_t payload=u16(p+6);
    if (count>WIRE_MAX_JOINTS || n!=28+payload || wire_crc32(p,n-4)!=u32(p+n-4)) return 0;
    if (type==WIRE_COMMAND) { if (!count || payload!=4u*count) return 0; }
    else if (type==WIRE_STOP || type==WIRE_ARM) { if (count || payload) return 0; }
    else return 0;
    WireCommand candidate={0};
    candidate.type=type; candidate.count=count;
    candidate.sequence=u64(p+8); candidate.session=u64(p+16);
    for (unsigned i=0;i<count;i++) {
        uint32_t bits=u32(p+24+4*i);
        memcpy(&candidate.value[i],&bits,4);
        if (!isfinite(candidate.value[i])) return 0;
    }
    *out=candidate; return 1;
}
int wire_push(WireRx* rx,uint8_t byte,WireCommand* out) {
    if (rx->used<4) {
        const uint8_t magic[4]={'R','L','B','1'};
        if (byte==magic[rx->used]) rx->data[rx->used++]=byte;
        else { rx->used=(byte=='R'); if (rx->used) rx->data[0]=byte; }
        return 0;
    }
    if (rx->used>=WIRE_MAX_FRAME) { rx->used=0;rx->expected=0;return 0; }
    rx->data[rx->used++]=byte;
    if (rx->used==8) {
        rx->expected=28u+u16(rx->data+6);
        if (rx->expected>WIRE_MAX_FRAME) { rx->used=0;rx->expected=0;return 0; }
    }
    if (rx->expected && rx->used==rx->expected) {
        int ok=wire_decode(rx->data,rx->used,out);
        rx->used=0;rx->expected=0;return ok;
    }
    return 0;
}
int wire_timed_expire(WireTimedRx* rx,uint64_t now,uint32_t timeout_ms) {
    if(!rx) return 0;
    int expired=rx->frame.used && (!timeout_ms || now<rx->last_ms ||
        now<rx->started_ms || now-rx->started_ms>=timeout_ms);
    if(expired) {
        rx->frame.used=0;rx->frame.expected=0;
        if(rx->expired<UINT64_MAX) rx->expired++;
    }
    rx->last_ms=now;
    return expired;
}
int wire_timed_push(WireTimedRx* rx,uint8_t byte,uint64_t now,uint32_t timeout_ms,WireCommand* out) {
    if(!rx || !out) return 0;
    (void)wire_timed_expire(rx,now,timeout_ms);
    if(!timeout_ms) return 0;
    int decoded=wire_push(&rx->frame,byte,out);
    /* A new/restarted magic prefix begins the total assembly deadline.
     * Later bytes cannot refresh it, even if they arrive continuously. */
    if(rx->frame.used==1) rx->started_ms=now;
    return decoded;
}
static void finish(uint8_t* p,size_t n) {
    unsigned sum=0; for(size_t i=2;i<n-1;i++) sum+=p[i];p[n-1]=(uint8_t)~sum;
}
size_t servo_write(uint8_t id,uint8_t address,const uint8_t* data,size_t n,uint8_t* out,size_t cap) {
    if(id>254 || !data || !out || !n || n>252 || cap<n+7) return 0;
    out[0]=out[1]=255;out[2]=id;out[3]=(uint8_t)(n+3);out[4]=3;out[5]=address;
    memcpy(out+6,data,n);finish(out,n+7);return n+7;
}
size_t servo_read(uint8_t id,uint8_t address,uint8_t n,uint8_t* out,size_t cap) {
    if(id>253 || !n || n>253 || !out || cap<8) return 0;
    out[0]=out[1]=255;out[2]=id;out[3]=4;out[4]=2;out[5]=address;out[6]=n;
    finish(out,8);return 8;
}
size_t servo_sync7(const uint8_t* ids,const uint8_t* rows,size_t n,uint8_t* out,size_t cap) {
    size_t size=8+8*n;
    if(!ids || !rows || !out || !n || n>31 || cap<size) return 0;
    for(size_t i=0;i<n;i++) {
        if(ids[i]>253) return 0;
        for(size_t j=0;j<i;j++) if(ids[j]==ids[i]) return 0;
    }
    out[0]=out[1]=255;out[2]=254;out[3]=(uint8_t)(size-4);out[4]=0x83;out[5]=41;out[6]=7;
    for(size_t i=0;i<n;i++) { out[7+8*i]=ids[i];memcpy(out+8+8*i,rows+7*i,7); }
    finish(out,size);return size;
}
int servo_status(const uint8_t* p,size_t n,uint8_t id,uint8_t* data,size_t count,uint8_t* error) {
    if(!p || !data || !error || n!=count+6 || n<6 || n>259 || p[0]!=255 || p[1]!=255 || p[2]!=id || p[3]!=count+2) return 0;
    unsigned sum=0;for(size_t i=2;i<n;i++) sum+=p[i];
    if((sum&255)!=255) return 0;
    memcpy(data,p+5,count);*error=p[4];return 1;
}
int16_t servo_signed(uint16_t v) { return v&0x8000 ? -(int16_t)(v&0x7fff) : (int16_t)v; }
uint16_t servo_signmag(int16_t v) { return v<0 ? ((uint16_t)(-(int32_t)v)&0x7fff)|0x8000 : (uint16_t)v; }
