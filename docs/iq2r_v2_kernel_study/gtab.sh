#!/usr/bin/env bash
# usage: gtab.sh STAGE LOG... -> STAGE us per M for each LOG prefix (LOG_512.log / LOG_256.log)
ST=$1; shift
for I in 512 256; do printf "I=%s M   " $I; for F in "$@"; do printf "%11s" "${F:0:11}"; done; echo
for M in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096; do row=$(printf "%6s" $M); any=0
 for F in "$@"; do v=$(grep -E "M= *$M " ${F}_$I.log 2>/dev/null | head -1 | sed -nE "s/.* $ST +([0-9.]+).*/\1/p"); [ -n "$v" ] && any=1; row="$row$(printf "%11s" ${v:--})"; done
 [ $any = 1 ] && echo "$row"; done; done
