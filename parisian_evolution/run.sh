#! /bin/bash

echo "Starting experiments..."
OUTPUT_ROOT_DIR='./output'
ALPHA=(0.5 1)
PSIZE=(3 5 20 50 100)
SEED=(42 119 91120)

for size in "${PSIZE[@]}"; do
    for alpha in "${ALPHA[@]}"; do
        for seed in "${SEED[@]}"; do
            OUTPUT_DIR="${OUTPUT_ROOT_DIR}/pe_${size}_${alpha}_${seed}"
            echo "Running parisian evolution with problem size ${size}, alpha ${alpha}, seed ${seed}..."
            mkdir -p "${OUTPUT_DIR}"
            python parisian.py --output_dir "${OUTPUT_DIR}" --alpha "${alpha}" --problem_size "${size}" --seed "${seed}" >> "${OUTPUT_DIR}/output.txt"
        done
    done
done

for size in "${PSIZE[@]}"; do
    for alpha in "${ALPHA[@]}"; do
        for seed in "${SEED[@]}"; do
            OUTPUT_DIR="${OUTPUT_ROOT_DIR}/ec_${size}_${alpha}_${seed}"
            echo "Running evolutionary computing with problem size ${size}, alpha ${alpha}, seed ${seed}..."
            mkdir -p "${OUTPUT_DIR}"
            python evolution_computing.py --output_dir "${OUTPUT_DIR}" --alpha "${alpha}" --problem_size "${size}" --seed "${seed}" >> "${OUTPUT_DIR}/output.txt"
        done
    done
done
