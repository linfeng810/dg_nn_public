#!/usr/bin/env bash
f2py -c ../z00-fortran-src/compact_fcode.f90 -m map_to_sfc_matrix --f2cmap .f2py_f2cmap 
f2py -c --f90flags='-ffree-line-length-none -ffixed-line-length-none   -fdefault-real-8 -fbounds-check -ffpe-trap=invalid,zero,overflow' ../z00-fortran-src/combined_fortran_source.f90 -m sfc
f2py -c ../z00-fortran-src/get_neighbouring.f90 -m get_nb
f2py -c ../z00-fortran-src/classicIP.F90 -m classicIP
