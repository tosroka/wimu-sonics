#!/bin/bash

allowed=("yue" "musicgen" "sin")

if (( $# % 2 != 0 )); then
    echo "Podaj parzystą liczbę argumentów (słowo liczba ...)."
    exit 1
fi

path=$(pwd)

for ((i=0; i<$#; i+=2)); do
    word_idx=$((i + 1))
    num_idx=$((i + 2))

    model="${!word_idx}"
    num="${!num_idx}"

    valid=false
    for w in "${allowed[@]}"; do
        if [[ "$model" == "$w" ]]; then
            valid=true
            break
        fi
    done

    if ! $valid; then
        echo "'$model' nie jest dozwolonym słowem. Możliwe wybory: ${allowed[*]}"
        exit 1
    fi

    if ! [[ "$num" =~ ^[0-9]+$ ]]; then
        echo "'$num' nie jest liczbą całkowitą."
        exit 1
    fi

    if [[ $model == "sin" ]]; then
        python3 make_sin.py "$num"
        continue
    fi

    last_num=$(python3 chat.py "$num" "$model")

    if ! [[ "$last_num" =~ ^[0-9]+$ ]]; then
        echo "Błąd: chat.py nie zwrócił liczby całkowitej: '$last_num'"
        exit 1
    fi

    if [[ $model == "yue" ]]; then
        cd ../../YuE/inference
        conda activate py38

        for ((j=0; j<num; j++)); do
            idx=$((j + last_num))
            python infer.py \
                --cuda_idx 1 \  # może 1 bo 0 często chyba zajęte i też domyślnie brane
                --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
                --stage2_model m-a-p/YuE-s2-1B-general \
                --genre_txt ../../wimu-sonics/data/prompt_egs/genre/${idx}.txt \
                --lyrics_txt ../../wimu-sonics/data/prompt_egs/lyrics/${idx}.txt \
                --run_n_segments 4 \
                --stage2_batch_size 4 \
                --output_dir ../../wimu-sonics/data/examples/YuE \
                --max_new_tokens 3000 \
                --repetition_penalty 1.1 \
                --prompt_start_time 0 \
                --prompt_end_time 120
        done

    elif [[ $model == "musicgen" ]]; then
        cd ../adiocraft
        # TODO
    fi

    cd "$path"

done
