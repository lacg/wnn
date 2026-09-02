/* FLASH 4M / RAM 4M match QEMU's mps2-an386 SSRAM banks; do not raise past the
 * board or the ELF will not load. The n=256 controller model is ~3.42 MiB of
 * .rodata, which fits, but only just. */
MEMORY {
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 4M
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}
SECTIONS {
  .text : { KEEP(*(.isr_vector)) *(.text*) *(.rodata*) } > FLASH
  . = ALIGN(4);
  /* .data lives in RAM but is LOADED from FLASH, so startup must copy it.
     These symbols are what makes that copy possible — without them (and the
     copy in bench.h) every initialised static reads as zero. */
  _sidata = LOADADDR(.data);
  .data : { . = ALIGN(4); _sdata = .; *(.data*) . = ALIGN(4); _edata = .; } > RAM AT > FLASH
  .bss  : { . = ALIGN(4); _sbss = .; *(.bss*) *(COMMON) . = ALIGN(4); _ebss = .; } > RAM
  . = ALIGN(8);
  _estack = ORIGIN(RAM) + LENGTH(RAM);
}
