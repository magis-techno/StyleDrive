TRAIN_TEST_SPLIT=styletrain

CUDA_VISIBLE_DEVICES=6,7 python $NAVSIM_DEVKIT_ROOT/planning/script/run_training.py \
        experiment_name=training_ego_mlp_style_agent \
        trainer.params.max_epochs=50 \
        train_test_split=$TRAIN_TEST_SPLIT \
        agent=ego_status_mlp_style_agent