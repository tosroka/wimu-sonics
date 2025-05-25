#!/bin/bash

allowed=("yue" "music_gen" "sin")

if (( $# % 2 != 0 )); then
    echo "Podaj parzystą liczbę argumentów (słowo liczba ...)."
    exit 1
fi

for ((i=0; i<$#; i+=2)); do
    word_idx=$((i + 1))
    num_idx=$((i + 2))

    word="${!word_idx}"
    num="${!num_idx}"

    valid=false
    for w in "${allowed[@]}"; do
        if [[ "$word" == "$w" ]]; then
            valid=true
            break
        fi
    done

    if ! $valid; then
        echo "'$word' nie jest dozwolonym słowem. Możliwe wybory: $allowed"
        exit 1
    fi

    if ! [[ "$num" =~ ^[0-9]+$ ]]; then
        echo "'$num' nie jest liczbą całkowitą."
        exit 1
    fi

    if [[ $word == "sin" ]]; then
        python3 make_sin.py $num
    else
        python3 chat.py $word $num
    fi

done
