TRAIN_TEST_SPLIT=styletest
CKPT=YOUR_CKPT_PATH

python $NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_pdm_score.py \
        train_test_split=$TRAIN_TEST_SPLIT \
        agent=ego_status_mlp_style_agent \
        worker=ray_distributed \
        agent.checkpoint_path=$CKPT \
        experiment_name=eval_ego_mlp_style_agent
