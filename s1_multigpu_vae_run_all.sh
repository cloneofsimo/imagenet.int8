NUM_GPUS=8

START_INDEX=64
END_INDEX=69

for ((i=START_INDEX; i<=END_INDEX; i++)); do

    GPU_INDEX=$((i % NUM_GPUS))

    CUDA_VISIBLE_DEVICES=$GPU_INDEX python3.10 preprocess_to_streaming.py --device cuda --file_index $i &

    if [ $((i % NUM_GPUS)) -eq $((NUM_GPUS - 1)) ]; then
        wait
    fi
done