import mlflow
import yaml
from mlflow.tracking import MlflowClient

def main():
    with open("configs/train_config.yaml") as f:
        config = yaml.safe_load(f)
        
    client = MlflowClient()
    experiment = client.get_experiment_by_name(config['mlflow']['experiment_name'])
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.promote_to = '{config['model']['stage']}'",
        order_by=["metrics.auc DESC"],
        max_results=1
    )
    
    if not runs:
        print("No models passed the required criteria for promotion.")
        return
        
    best_run = runs[0]
    print(f"Registering run {best_run.info.run_id} (AUC: {best_run.data.metrics['auc']:.4f})")
    
    result = mlflow.register_model(f"runs:/{best_run.info.run_id}/model", config['model']['name'])
    client.transition_model_version_stage(
        name=config['model']['name'],
        version=result.version,
        stage=config['model']['stage'],
        archive_existing_versions=True
    )

if __name__ == "__main__":
    main()