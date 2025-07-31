TRAIN_TEST_SPLIT=styletrain

CUDA_VISIBLE_DEVICES=0,1 python $NAVSIM_DEVKIT_ROOT/planning/script/run_training.py \
        agent=diffusiondrive_style_agent \
        experiment_name=training_diffusiondrive_style_agent  \
        train_test_split=$TRAIN_TEST_SPLIT  \
        split=trainval   \
        trainer.params.max_epochs=100 \
        cache_path="${NAVSIM_EXP_ROOT}/training_cache/" \
        use_cache_without_dataset=True  \
        force_cache_computation=False