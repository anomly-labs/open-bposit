/* bp16_quire_selftest.c — host-native bit-exact equivalence gate for bp16_quire.h.
 *
 * Copyright (c) 2026 Anomly, Inc.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Host-only (it has main(); not part of any device build). It is the guard that keeps the two fused MADD entry points
 * (bp16_madd_q256 / _packed) byte-for-byte identical to the scalar reference
 * path (bp16_prod_to_q256 + q256_add) after any edit to the shared windowed
 * arithmetic (q256_addsub_window). Build & run:
 *
 *   gcc -O2 -I. bp16_quire_selftest.c -o /tmp/bp16_quire_selftest && /tmp/bp16_quire_selftest
 *   gcc -O2 -DBP16Q_CHECK -I. bp16_quire_selftest.c -o /tmp/... # also arms the
 *                                                               # overflow guard
 *
 * Sweeps a coprime-strided lattice over the full 16-bit code space (~143k pairs)
 * plus a running-accumulation stream; prints "ALL BIT-EXACT" and exits 0 on pass.
 */
#include <stdio.h>
#include <string.h>
#include "bp16_quire.h"

static int eq(const unsigned a[8], const unsigned b[8]) { return memcmp(a, b, 32) == 0; }

int main(void) {
    long madd_fail = 0, packed_fail = 0, checked = 0;
    static const int step = 173; /* coprime-ish stride over 16-bit space */
    for (int a = 0; a < 65536; a += step) {
        for (int b = 0; b < 65536; b += step) {
            unsigned ref[8]; memset(ref, 0, sizeof ref);
            unsigned prod[8]; bp16_prod_to_q256(a, b, prod); q256_add(ref, prod);
            unsigned f[8]; memset(f, 0, sizeof f);
            bp16_madd_q256(f, a, b);
            if (!eq(ref, f)) { if (madd_fail < 5) printf("MADD mismatch a=%04x b=%04x\n", a, b); madd_fail++; }
            unsigned p[8]; memset(p, 0, sizeof p);
            bp16_madd_q256_packed(p, bp16_pack(a), bp16_pack(b));
            if (!eq(ref, p)) { if (packed_fail < 5) printf("PACKED mismatch a=%04x b=%04x\n", a, b); packed_fail++; }
            checked++;
        }
    }
    /* Running-sum ordering stress: fused-in-place vs scalar-reference stream. */
    unsigned accF[8]; memset(accF, 0, sizeof accF);
    unsigned accR[8]; memset(accR, 0, sizeof accR);
    for (int a = 0; a < 65536; a += step) {
        int b = (a * 7 + 12345) & 0xFFFF;
        bp16_madd_q256(accF, a, b);
        unsigned prod[8]; bp16_prod_to_q256(a, b, prod); q256_add(accR, prod);
    }
    int acc_ok = eq(accF, accR);
    printf("checked=%ld madd_fail=%ld packed_fail=%ld acc_stream=%s\n",
           checked, madd_fail, packed_fail, acc_ok ? "OK" : "FAIL");
    if (madd_fail == 0 && packed_fail == 0 && acc_ok) { printf("ALL BIT-EXACT\n"); return 0; }
    return 1;
}
