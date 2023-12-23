#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <inttypes.h>
#include <xmmintrin.h>
#include <immintrin.h>

// gcc simdhmaxbmark.c -o simdhmaxbmark.bin -O3 -lm -msse -Wall


uint64_t  get_cycles () {
  uint32_t lo,hi;
  asm  volatile("rdtsc":"=a"(lo),"=d"(hi));
  return  (( uint64_t)hi<<32 | lo);
}

int warmup() {
  uint32_t i;
  for(i=0; i<100000000; i++) {
    if (!get_cycles()) {
      printf("Warm up message\n");
    }
  };
  return 0;
}

inline float _mm_hmax_ps(__m128 arg) {
  // Returns the maximum 32 bit float value in arg.
  // Requires SSE1.
  __m128 temp, temp2;
  //temp = _mm_shuffle_ps(arg, arg, 78);  // Use this with second shuffle
  temp = _mm_shuffle_ps(arg, arg, 0xb1);  // Use this with movehl
  temp = _mm_max_ps(arg, temp);
  //temp2 = _mm_shuffle_ps(temp, temp, 165); // Comment out this line
  temp2 = _mm_movehl_ps(arg, temp);          // ...or this line
  temp2 = _mm_max_ps(temp2, temp);
  return _mm_cvtss_f32(temp2);
}

int main(int argc, char **argv)
{
  uint64_t j, cyclesstart, cyclesend;
  if (argc < 5) exit(1);
  __m128 vec = _mm_setr_ps(atof(argv[1]), atof(argv[2]), atof(argv[3]), atof(argv[4]));
  if (warmup() != 0) exit(1);
  uint64_t junk = 0;
  cyclesstart = get_cycles();
  for (j=0; j<1000000; j++) {
    junk |= (uint64_t)(_mm_hmax_ps(vec)); // shenanigans to fool the optimiser
    vec = _mm_shuffle_ps(vec, vec, 78);  
  }
  cyclesend = get_cycles();
  printf("%lu\n", junk);
  printf("_mm_hmax_ps() Cycles = %li (%lf per iteration)\n", cyclesend - cyclesstart, (double)(cyclesend - cyclesstart)/(j)); 
  exit(0);
}
// Nehalem Core i5 laptop
// ======================
// ./simdhmaxbmark.bin 1 2 3 4
// _mm_hmax_ps() Cycles = 7407898 (7.407898 per iteration) - movehl
// _mm_hmax_ps() Cycles = 5510951 (5.510951 per iteration) - 2nd shuffle
//
// Xeon E3-1225v2
// ==============
// ./simdhmaxbmark.bin 1 2 3 4
// __mm_hmax_ps() Cycles = 4450959 (4.450959 per iteration) - movehl
// __mm_hmax_ps() Cycles = 4001996 (4.001996 per iteration) - 2nd shuffle
// 

