/* The director's internals, for tests: its tokenizer and sentence features (st_director.c). */
#ifndef ST_DIRECTOR_H
#define ST_DIRECTOR_H

#define ST_DIRECTOR_FEATURES 18
#define ST_DIRECTOR_GENRES 10

/* The hashed tokens of a text, as training: CRC32 of "w:word" and "b:word_word" into 16384 buckets. */
int st_director_tokens(const char* text, int* out, int max);
void st_director_features(const char* text, int dialogue, int has_speaker, int para_len, int para_index,
                          float out[ST_DIRECTOR_FEATURES]);

#endif
