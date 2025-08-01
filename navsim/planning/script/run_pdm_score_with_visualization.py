"""
Run PDM Score evaluation with visualization support.
Extends the standard PDM scoring with BEV visualization of results.
"""

import os
import pickle
import lzma
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from nuplan.planning.utils.multithreading.worker_ray import RayDistributed

from navsim.common.dataclasses import PDMResults, Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.common.dataloader import MetricCacheLoader, SceneLoader, SceneFilter

from hydra.utils import instantiate
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer

from navsim.common.dataclasses import SensorConfig
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.script.utils import run_runners
from navsim.visualization.evaluation_viz import create_evaluation_visualization, save_evaluation_results

import logging
import json
import os

logger = logging.getLogger(__name__)

# Configuration paths
CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

# Load style test labels
dataset_root = os.environ.get('OPENSCENE_DATA_ROOT')
styletest_json_path = os.path.join(dataset_root, "extra_data/styletest.json")
with open(styletest_json_path, 'r', encoding='utf-8') as f:
    style_dict = json.load(f)
STYLETEST_LABEL = {key: value["ANC_result"] for key, value in style_dict.items()}


def run_pdm_score_with_visualization(
    args: List[Dict[str, Union[List[str], DictConfig]]],
    output_viz_dir: Optional[str] = None,
    max_visualizations: int = 50
) -> List[Dict[str, Any]]:
    """
    Run PDM scoring with visualization for selected scenarios.
    
    :param args: List of arguments for PDM scoring
    :param output_viz_dir: Directory to save visualizations
    :param max_visualizations: Maximum number of visualizations to generate
    :return: List of PDM results with visualization metadata
    """
    
    # Unpack arguments
    thread_id, node_id, tokens_to_evaluate, agent, metric_cache_loader, simulator, scorer, scene_loader = args
    
    if output_viz_dir:
        os.makedirs(output_viz_dir, exist_ok=True)
        viz_counter = 0
    
    pdm_results: List[Dict[str, Any]] = []
    
    for idx, token in enumerate(tokens_to_evaluate):
        logger.info(
            f"Processing scenario {idx + 1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}"
        )
        
        score_row: Dict[str, Any] = {"token": token, "valid": True, "has_visualization": False}
        
        try:
            # Load metric cache
            metric_cache_path = metric_cache_loader.metric_cache_paths[token]
            with lzma.open(metric_cache_path, "rb") as f:
                metric_cache: MetricCache = pickle.load(f)
            
            # Get scene and ground truth trajectory
            scene = scene_loader.get_scene_from_token(token)
            gt_trajectory = scene.get_future_trajectory()
            
            # Get agent prediction
            agent_input = scene_loader.get_agent_input_from_token(token)
            if agent.requires_scene:
                trajectory = agent.compute_trajectory(agent_input, scene)
            else:
                trajectory = agent.compute_trajectory(agent_input, token)
            
            # Get style for this scenario
            style = STYLETEST_LABEL.get(token, "N")
            
            # Run PDM evaluation
            pdm_result = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                style=style,
                gt_trajectory=gt_trajectory
            )
            
            # Add PDM results to score row
            score_row.update(pdm_result.__dict__)
            score_row["style"] = style
            
            # Generate visualization if requested and within limit
            if output_viz_dir and viz_counter < max_visualizations:
                try:
                    # Prepare trajectories for visualization
                    trajectories = {
                        "predicted": trajectory,
                        "ground_truth": gt_trajectory,
                        "pdm_reference": metric_cache.trajectory
                    }
                    
                    # Prepare PDM results for visualization
                    pdm_results_dict = {
                        "predicted": pdm_result,
                        # Could add GT and PDM reference scores if available
                    }
                    
                    # Get current frame for BEV background
                    current_frame = scene.frames[0]  # Use first frame as reference
                    
                    # Create visualization
                    evaluation_results = {
                        "trajectories": trajectories,
                        "pdm_results": pdm_results_dict,
                        "style": style
                    }
                    
                    # Try to get map API (might not be available in all configurations)
                    map_api = None
                    try:
                        map_api = scene.map_api
                    except:
                        logger.warning(f"Map API not available for scene {token}")
                    
                    # Create the visualization figure
                    fig = create_evaluation_visualization(
                        frame=current_frame,
                        trajectories=trajectories,
                        pdm_results=pdm_results_dict,
                        map_api=map_api,
                        scene_token=token,
                        style=style
                    )
                    
                    evaluation_results["figure"] = fig
                    
                    # Save visualization and data
                    viz_path = save_evaluation_results(
                        evaluation_results, 
                        output_viz_dir, 
                        token
                    )
                    
                    score_row["visualization_path"] = viz_path
                    score_row["has_visualization"] = True
                    viz_counter += 1
                    
                    # Close figure to free memory
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    
                    logger.info(f"Generated visualization for scene {token[:8]}")
                    
                except Exception as viz_error:
                    logger.warning(f"Failed to generate visualization for {token}: {viz_error}")
                    # Continue processing even if visualization fails
            
        except Exception as e:
            logger.warning(f"----------- Agent failed for token {token}:")
            traceback.print_exc()
            score_row["valid"] = False
        
        pdm_results.append(score_row)
    
    return pdm_results


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main function to run PDM scoring with visualization.
    """
    
    # Build components
    logger.info("Building PDM Scorer...")
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"
    
    logger.info("Building SceneLoader")
    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    
    logger.info("Building MetricCacheLoader")
    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))
    
    logger.info("Building Agent")
    planner = instantiate(cfg.agent)
    planner.initialize()
    
    # Setup visualization output directory
    output_viz_dir = None
    if cfg.get("enable_visualization", False):
        output_viz_dir = cfg.get("visualization_output_dir", "pdm_evaluation_visualizations")
        output_viz_dir = os.path.join(cfg.output_dir, output_viz_dir)
        logger.info(f"Visualization output directory: {output_viz_dir}")
    
    # Get tokens to evaluate
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    logger.info(f"Evaluating {len(tokens_to_evaluate)} scenarios")
    
    # Limit visualizations if specified
    max_visualizations = cfg.get("max_visualizations", 50)
    if output_viz_dir:
        logger.info(f"Will generate up to {max_visualizations} visualizations")
    
    # Prepare arguments for worker function
    args = [
        [
            0,  # thread_id
            0,  # node_id
            tokens_to_evaluate,
            planner,
            metric_cache_loader,
            simulator,
            scorer,
            scene_loader,
        ]
    ]
    
    # Build worker and run
    logger.info("Starting evaluation...")
    worker = build_worker(cfg.worker)
    
    if isinstance(worker, RayDistributed):
        # For distributed execution, we might need to modify this
        # For now, run single-threaded to ensure visualization works
        logger.warning("Visualization mode: Running single-threaded for compatibility")
        results = [run_pdm_score_with_visualization(args[0], output_viz_dir, max_visualizations)]
    else:
        # Use the worker but with our custom function
        results = run_runners(
            lambda x: run_pdm_score_with_visualization(x, output_viz_dir, max_visualizations),
            worker,
            args
        )
    
    # Flatten results
    pdm_score_rows = []
    for result in results:
        pdm_score_rows.extend(result)
    
    # Create DataFrame and save results
    pdm_score_df = pd.DataFrame(pdm_score_rows)
    
    if not pdm_score_df["valid"].all():
        logger.warning("Evaluation for some tokens failed. Check log for details")
    
    # Save evaluation results
    output_file = Path(cfg.output_dir) / f"{cfg.experiment_name}_pdm_score.csv"
    pdm_score_df.to_csv(output_file, index=False)
    logger.info(f"Saved PDM scores to {output_file}")
    
    # Generate summary statistics
    if output_viz_dir and pdm_score_df["has_visualization"].any():
        num_visualizations = pdm_score_df["has_visualization"].sum()
        logger.info(f"Generated {num_visualizations} visualizations in {output_viz_dir}")
        
        # Save summary of visualized scenes
        viz_scenes = pdm_score_df[pdm_score_df["has_visualization"]]
        summary_file = Path(output_viz_dir) / "visualization_summary.csv"
        viz_scenes[["token", "score", "style", "visualization_path"]].to_csv(summary_file, index=False)
        logger.info(f"Saved visualization summary to {summary_file}")


if __name__ == "__main__":
    main()