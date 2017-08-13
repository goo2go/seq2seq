export VOCAB_SOURCE=/home/robin/data/lang8.vocab
export VOCAB_TARGET=/home/robin/data/lang8.vocab
export TRAIN_SOURCE=/home/robin/data/lang8.train.s
export TRAIN_TARGET=/home/robin/data/lang8.train.t
export DEV_SOURCE=/home/robin/data/lang8.dev.s
export DEV_TARGET=/home/robin/data/lang8.dev.t

export TRAIN_STEPS=180000

export MODEL_DIR=/workspace/robin/model/grammar/lang8_2w_enc_rnn_dec_rnn_ptb
mkdir -p $MODEL_DIR

#CURDIR="`pwd`"/"`dirname $0`"
#echo $CURDIR

cd ..

CUDA_VISIBLE_DEVICES=0 python -m bin.train \
    --config_paths="
        ./script_robin/configs/test.yml" \
    --model_params "
        vocab_source: $VOCAB_SOURCE
        vocab_target: $VOCAB_TARGET" \
    --input_pipeline_train "
        class: ParallelTextInputPipeline
        params:
            source_files:
                - $TRAIN_SOURCE
            target_files:
                - $TRAIN_TARGET" \
    --input_pipeline_dev "
        class: ParallelTextInputPipeline
        params:
            source_files:
                - $DEV_SOURCE
            target_files:
                - $DEV_TARGET" \
    --batch_size 32 \
    --train_steps $TRAIN_STEPS \
    --output_dir $MODEL_DIR
