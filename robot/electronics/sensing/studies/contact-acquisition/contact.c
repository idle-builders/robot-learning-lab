#include "contact.h"
#include <string.h>
void contact_init(Contact *state) { memset(state, 0, sizeof(*state)); }
size_t contact_record_size(void) { return sizeof(ContactRecord); }
int contact_read(Contact *state, const ContactBus *bus, uint8_t channel,
                 ContactRecord *record) {
    if (!state || !bus || !bus->now_us || !bus->transfer3 || !record || channel > 7)
        return 0;
    memset(record, 0, sizeof(*record));
    record->channel = channel;
    record->sequence = ++state->sequence[channel];
    const uint8_t tx[3] = {(uint8_t)(6 | (channel >> 2)),
                           (uint8_t)((channel & 3) << 6), 0};
    uint8_t rx[3] = {0};
    record->transfer_begin_us = bus->now_us(bus->context);
    int transferred = bus->transfer3(bus->context, tx, rx);
    record->transfer_end_us = bus->now_us(bus->context);
    record->transfer_complete = transferred == 3;
    if (record->transfer_complete)
        record->raw_count = (uint16_t)(((rx[1] & 15) << 8) | rx[2]);
    /* Completion does not establish ADC presence, continuity, or correct data. */
    return 1;
}
