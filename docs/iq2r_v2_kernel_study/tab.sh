#!/usr/bin/env bash
# usage: tab.sh LOG... -> gate us per M (rows) for each log (columns), I=512 then I=256 logs via suffix
for I in 512 256; do printf "I=%s M   " $I; for F in "$@"; do printf "%12s" "${F:0:12}"; done; echo
for M in 2 4 8 16 32 64 128 256 512; do row=$(printf "%6s" $M); any=0
 for F in "$@"; do v=$(grep -E "M= *$M " ${F}_$I.log 2>/dev/null | head -1 | sed -E 's/.*gate +([0-9.]+).*/\1/'); [ -n "$v" ] && any=1; row="$row$(printf "%12s" ${v:--})"; done
 [ $any = 1 ] && echo "$row"; done; done
