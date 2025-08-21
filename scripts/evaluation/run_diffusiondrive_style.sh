TRAIN_TEST_SPLIT=styletest
CKPT=$NAVSIM_DEVKIT_ROOT/../checkpoints/

python $NAVSIM_DEVKIT_ROOT/planning/script/run_pdm_score.py \
        train_test_split=$TRAIN_TEST_SPLIT \
        agent=diffusiondrive_style_agent \
        worker=ray_distributed \
        agent.checkpoint_path=$CKPT \
        experiment_name=eval_diff_style_agent_ablation
