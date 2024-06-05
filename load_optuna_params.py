import optuna

def load_best_hyperparameters(study_name: str, storage_name: str):
    # Studyの読み込み
    study = optuna.load_study(study_name=study_name, storage=storage_name)

    # 最適な試行の取得
    best_trial = study.best_trial

    # 最適なハイパーパラメータの表示
    print("Best trial parameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    return best_trial.params

if __name__ == "__main__":
    study_name = "hyper_gru"  # 確認したStudy名に変更
    storage_name = "sqlite:///optuna_studies/hyper_gru/optuna_study.db"  # 使用しているストレージのパス
    best_params = load_best_hyperparameters(study_name, storage_name)
