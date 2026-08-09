MEMORY {
  FLASH (rx) : ORIGIN = 0x00000000, LENGTH = 4M
  RAM  (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}
SECTIONS {
  .text : { KEEP(*(.isr_vector)) *(.text*) *(.rodata*) } > FLASH
  .data : { *(.data*) } > RAM AT > FLASH
  .bss  : { *(.bss*) *(COMMON) } > RAM
  . = ALIGN(8);
  _estack = ORIGIN(RAM) + LENGTH(RAM);
}
