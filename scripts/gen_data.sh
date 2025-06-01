#!/bin/bash

allowed=("yue" "musicgen" "sin")

if (( $# % 2 != 0 )); then
    echo "Give an even number of arguments (model number ...)."
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
        echo "'$model' is incorrect word. Valid choices: ${allowed[*]}"
        exit 1
    fi

    if ! [[ "$num" =~ ^[0-9]+$ ]]; then
        echo "'$num' is not int."
        exit 1
    fi

    if [[ $model == "sin" ]]; then
        python3 make_sin.py "$num"
        echo "Succesfully generated sin and saved in ../data/examples/$model"
        continue
    fi

    echo "Generating prompts for $model"
    last_num=$(python3 chat.py "$num" "$model")
    echo "Next number, from which there will be generated songs: '$last_num'"

    if ! [[ "$last_num" =~ ^[0-9]+$ ]]; then
        echo "Error: chat.py didn't return int: '$last_num'"
        exit 1
    fi

    if [[ $model == "yue" ]]; then
        if [ ! -d "../../YuE/inference" ]; then
            echo "There is no ../../YuE/inference"
            exit 1
        fi
        cd ../../YuE/inference
        echo "Entering in $model: $(pwd)"
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate py38

        for ((j=0; j<num; j++)); do
            idx=$((j + last_num))
            python3 infer.py \
                --cuda_idx 1 \
                --stage1_model m-a-p/YuE-s1-7B-anneal-en-icl \
                --stage2_model m-a-p/YuE-s2-1B-general \
                --genre_txt ../../wimu-sonics/data/prompt_egs/genre/${idx}.txt \
                --lyrics_txt ../../wimu-sonics/data/prompt_egs/lyrics/${idx}.txt \
                --run_n_segments 4 \
                --stage2_batch_size 4 \
                --output_dir ../output \
                --max_new_tokens 3000 \
                --repetition_penalty 1.1 \
                --prompt_start_time 0 \
                --prompt_end_time 120

            echo "Generated: '$idx' songs and saved in ../output"
        done

        if [ ! -d "../../wimu-sonics/data/examples/YuE" ]; then
            mkdir ../../wimu-sonics/data/examples/YuE
        fi
        cp ../output/*.mp3 ../../wimu-sonics/data/examples/YuE
        echo "Copy mp3 files from ../output to wimu-sonics/data/examples/YuE"
        
        echo "Returning to '$path'"
        cd "$path"

    elif [[ $model == "musicgen" ]]; then
        source ~/miniconda3/etc/profile.d/conda.sh
        conda activate py39
        
        echo "Entering in $model: $(pwd)"

        python3 gen_musicgen.py $last_num | exit 1
        
        echo "Generated new songs with musicgen and saved in ../data/examples/$model"
    fi

done
