# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Thin Cython binding for gbcore (C port of PyBoy's DMG core)."""
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int64_t, uint64_t
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AS_STRING

import numpy as np

cdef extern from "gbcore.h":
    ctypedef struct GB:
        int64_t breakpoint_waiting
        int64_t frame_count
        int external_rom_count
        int external_ram_count
        uint8_t *rombanks
        uint8_t rambanks[16 * 8192]
        uint8_t internal_ram0[8192]
        uint8_t VRAM0[8192]
        uint32_t screenbuffer[144 * 160]
        uint8_t bootrom[0x100]
        int cgb
        uint64_t prof[16]
        uint64_t ioprof[512]
    int gb_init(GB *g, const uint8_t *rom, size_t rom_len, const uint8_t *bootrom, size_t bootrom_len)
    void gb_free(GB *g)
    int gb_tick(GB *g, int count, int render, int sound) nogil
    int gb_tick_resume(GB *g) nogil
    uint8_t gb_getitem(GB *g, uint16_t i)
    void gb_setitem(GB *g, uint16_t i, uint8_t v)
    int gb_breakpoint_add(GB *g, int bank, int addr)
    int gb_breakpoint_remove(GB *g, int bank, int addr)
    int gb_breakpoint_find(GB *g, int bank, int addr)
    int gb_send_input(GB *g, int event, int delay)
    size_t gb_state_size(GB *g)
    size_t gb_save_state(GB *g, uint8_t *out)
    int gb_load_state(GB *g, const uint8_t *data, size_t length)


cdef class Core:
    cdef GB *g
    cdef public object screen_ndarray
    cdef public object wram

    def __cinit__(self, bytes rom, bytes bootrom):
        self.g = <GB *>malloc(sizeof(GB))
        if self.g == NULL:
            raise MemoryError()
        rc = gb_init(self.g, <const uint8_t *>PyBytes_AS_STRING(rom), len(rom),
                     <const uint8_t *>PyBytes_AS_STRING(bootrom), len(bootrom))
        if rc != 0:
            gb_free(self.g)
            free(self.g)
            self.g = NULL
            raise ValueError(f"gbcore init failed ({rc}): unsupported cartridge or bad ROM")
        cdef uint8_t[:] sb = <uint8_t[:144 * 160 * 4]>(<uint8_t *>self.g.screenbuffer)
        self.screen_ndarray = np.asarray(sb).reshape(144, 160, 4)
        cdef uint8_t[:] wr = <uint8_t[:8192]>(self.g.internal_ram0)
        self.wram = np.asarray(wr)

    def __dealloc__(self):
        if self.g != NULL:
            gb_free(self.g)
            free(self.g)
            self.g = NULL

    cpdef int tick(self, int count, bint render, bint sound):
        cdef int r
        with nogil:
            r = gb_tick(self.g, count, render, sound)
        return r

    cpdef int tick_resume(self):
        cdef int r
        with nogil:
            r = gb_tick_resume(self.g)
        return r

    @property
    def breakpoint_waiting(self):
        return self.g.breakpoint_waiting

    @property
    def ioprof(self):
        return [self.g.ioprof[k] for k in range(512)]

    @property
    def prof(self):
        return [self.g.prof[k] for k in range(16)]

    @property
    def frame_count(self):
        return self.g.frame_count

    @property
    def external_rom_count(self):
        return self.g.external_rom_count

    @property
    def external_ram_count(self):
        return self.g.external_ram_count

    cpdef int getitem(self, int i):
        return gb_getitem(self.g, <uint16_t>i)

    cpdef setitem(self, int i, uint8_t v):
        gb_setitem(self.g, <uint16_t>i, v)

    cpdef list getslice(self, int start, int stop, int step):
        cdef int x
        cdef int n = (stop - start) // step
        cdef list out = [0] * n
        for x in range(start, stop, step):
            out[(x - start) // step] = gb_getitem(self.g, <uint16_t>x)
        return out

    cpdef bytes read_bytes(self, int start, int n):
        """bytes(memory[start:start + n]) with memory-bus semantics (start + n <= 0x10000)."""
        if n <= 0:
            return b""
        if start < 0 or start + n > 0x10000:
            raise IndexError("address out of range")
        cdef bytes out = PyBytes_FromStringAndSize(NULL, n)
        cdef uint8_t *p = <uint8_t *>PyBytes_AS_STRING(out)
        cdef int k
        for k in range(n):
            p[k] = gb_getitem(self.g, <uint16_t>(start + k))
        return out

    cpdef int rom_byte(self, int bank, int addr):
        return self.g.rombanks[<int64_t>bank * 0x4000 + addr]

    cpdef set_rom_byte(self, int bank, int addr, uint8_t v):
        self.g.rombanks[<int64_t>bank * 0x4000 + addr] = v

    cpdef int ram_byte(self, int bank, int addr):
        return self.g.rambanks[<int64_t>bank * 8192 + addr]

    cpdef set_ram_byte(self, int bank, int addr, uint8_t v):
        self.g.rambanks[<int64_t>bank * 8192 + addr] = v

    cpdef int vram_byte(self, int addr):
        return self.g.VRAM0[addr]

    cpdef set_vram_byte(self, int addr, uint8_t v):
        self.g.VRAM0[addr] = v

    cpdef int wram_byte(self, int addr):
        return self.g.internal_ram0[addr]

    cpdef set_wram_byte(self, int addr, uint8_t v):
        self.g.internal_ram0[addr] = v

    cpdef int boot_byte(self, int addr):
        return self.g.bootrom[addr]

    cpdef int breakpoint_add(self, int bank, int addr):
        return gb_breakpoint_add(self.g, bank, addr)

    cpdef int breakpoint_remove(self, int bank, int addr):
        return gb_breakpoint_remove(self.g, bank, addr)

    cpdef int breakpoint_find(self, int bank, int addr):
        return gb_breakpoint_find(self.g, bank, addr)

    cpdef send_input(self, int event, int delay):
        if gb_send_input(self.g, event, delay) != 0:
            raise OverflowError("input queue full")

    cpdef bytes save_state(self):
        cdef size_t n = gb_state_size(self.g)
        cdef bytes out = PyBytes_FromStringAndSize(NULL, n)
        gb_save_state(self.g, <uint8_t *>PyBytes_AS_STRING(out))
        return out

    cpdef load_state(self, bytes data):
        rc = gb_load_state(self.g, <const uint8_t *>PyBytes_AS_STRING(data), len(data))
        if rc != 0:
            raise ValueError(f"gbcore load_state failed ({rc})")

    cpdef load_cart_ram(self, bytes data):
        cdef int n = min(len(data), self.g.external_ram_count * 8192)
        memcpy(self.g.rambanks, PyBytes_AS_STRING(data), n)
