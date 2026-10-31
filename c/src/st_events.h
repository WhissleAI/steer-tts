/* Recorded sound events for inline [laugh] [cough] [dog] ... tokens: the package's events.safetensors
 * (steertts/events.py EventBank, same picks and the same mel up to float rounding). Clips are Vocos log-mels that go
 * through the engine's own vocoder. */
#ifndef ST_EVENTS_H
#define ST_EVENTS_H
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct st_events st_events;

/* NULL + st_seterr when the file is not a SteerTTS event bank */
st_events* st_events_open(const char* path);
void st_events_free(st_events* b);
int st_events_has(const st_events* b, const char* type);   /* lower-case type name */

/* EventBank.loudness of a time-major log-mel [T, 100] */
float st_events_loudness(const float* mel, int T);

/* EventBank.mel(type, gender, f0, ref_loud, seed): gender 'f' / 'm' / 0, f0 <= 0 = unknown, has_ref 0 = keep the
 * clip's level. *mel [T, 100] time-major (st_free it), 80 ms of silence each side. */
int st_events_mel(const st_events* b, const char* type, char gender, double f0, int has_ref, float ref_loud,
                  int64_t seed, float** mel, int* T);

#ifdef __cplusplus
}
#endif
#endif
