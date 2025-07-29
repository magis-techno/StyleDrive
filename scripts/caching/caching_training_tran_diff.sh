python $NAVSIM_DEVKIT_ROOT/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_style_agent \
    experiment_name=training_diffusiondrive_style_agent \
    train_test_split=styletrain \
    cache_path=$NAVSIM_EXP_ROOT/training_cache \
    worker=ray_low_resource