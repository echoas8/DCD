import os
import pandas as pd
import numpy as np
import ast
from scipy.stats import kendalltau
from sklearn.metrics import ndcg_score

NUM_LIST = [1, 5, 10, 15, 20, 25, 50, 75, 100]
RESULTS_DIR = './results'
OUTPUT_EXCEL_COS = './experiment_results_cosine.xlsx'
OUTPUT_EXCEL_SOFTKL = './experiment_results_softkl.xlsx'
GROUND_TRUTH_FILE = 'model_rankings.csv'

TARGET_KEYS = [
    'car', 'cifar', 'clevr', 'country', 'dr', 'dmlab', 'dtd', 'eurosat',
    'fer2013', 'gtsrb', 'kitti', 'mnist', 'oxford_flowers', 'oxford_pet',
    'patch_camelyon', 'renderedsst2', 'resisc45', 'stl10', 'sun397',
    'svhn', 'voc_pose'
]


def cosine_similarity(v1, v2, eps=1e-8):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if v1.shape != v2.shape:
        raise ValueError(f"维度不一致: v1 {v1.shape} vs v2 {v2.shape}")
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 < eps or norm2 < eps:
        return 0.0
    
    return np.dot(v1, v2) / (norm1 * norm2)


def symmetric_soft_kl(v1, v2, eps=1e-8):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if v1.shape != v2.shape:
        raise ValueError(f"维度不一致: v1 {v1.shape} vs v2 {v2.shape}")
    
    def softmax(v):
        v_max = np.max(v)
        exp_v = np.exp(v - v_max)
        return exp_v / np.sum(exp_v)
    
    p = softmax(v1)
    q = softmax(v2)
    
    m = 0.5 * (p + q)
    
    def kl_divergence(p_dist, q_dist):
        return np.sum(p_dist * np.log((p_dist + eps) / (q_dist + eps)))
    
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    
    similarity = 1.0 - jsd / np.log(2)
    return max(0.0, similarity)


def calculate_task_similarities(task_key, file_pairs, output_csv, similarity_func):
    similarity_matrix = {}
    valid_models = []

    for target_path, source_path in file_pairs:
        try:
            target_data = np.load(target_path, allow_pickle=True).item()
            source_data = np.load(source_path, allow_pickle=True).item()
        except Exception as e:
            continue

        if task_key not in target_data:
            continue

        model_name = os.path.basename(target_path).split('.')[0]
        valid_models.append(model_name)

        task_vec = target_data[task_key]
        other_keys = [k for k in source_data.keys() if k != task_key]

        similarities = []
        for k in other_keys:
            if k not in source_data:
                continue
            sim = similarity_func(task_vec, source_data[k])
            similarities.append(sim)
        similarities = np.array(similarities)

        if len(similarities) == 0:
            continue

        gamma = 4.0
        exp_probs = np.exp(gamma * (similarities - np.max(similarities)))
        norm_probs = exp_probs / np.sum(exp_probs)

        for key, sim in zip(other_keys, norm_probs):
            if key not in similarity_matrix:
                similarity_matrix[key] = []
            similarity_matrix[key].append(sim)

    if not similarity_matrix:
        return None

    similarity_df = pd.DataFrame(similarity_matrix, index=valid_models)
    similarity_df.to_csv(output_csv)
    return similarity_df


def preprocess_model_rankings(ranking_file):
    df = pd.read_csv(ranking_file, index_col=0)

    task_name_map = {
        'Car': 'car', 'Cifar100': 'cifar', 'clevr_count': 'clevr',
        'country211': 'country', 'diabetic_retinopathy': 'dr',
        'dmlab': 'dmlab', 'dtd': 'dtd', 'eurosat': 'eurosat',
        'fer2013': 'fer2013', 'gtsrb': 'gtsrb', 'kitti': 'kitti',
        'MNIST': 'mnist', 'oxford-flower': 'oxford_flowers',
        'oxford-pet': 'oxford_pet', 'patch_camelyon': 'patch_camelyon',
        'renderedsst2': 'renderedsst2', 'resisc45': 'resisc45',
        'stl10': 'stl10', 'sun397': 'sun397', 'svhn': 'svhn',
        'Voc2007': 'voc_pose'
    }
    df = df.rename(columns=task_name_map)

    new_indices = []
    for idx in df.index:
        try:
            parts = ast.literal_eval(idx)
            if isinstance(parts, list) and len(parts) >= 2:
                model_arch = parts[0].replace('-', '_').lower()
                pretrained = parts[1].replace('-', '_').lower()
                new_name = f"layer_conductance_{model_arch}_{pretrained}"
                new_indices.append(new_name)
            else:
                new_indices.append(idx)
        except:
            new_indices.append(idx)

    df.index = new_indices
    return df


def calculate_estimated_rankings(task_key, similarity_df, model_rankings_df, output_csv):
    estimated_scores = {}

    common_models = similarity_df.index.intersection(model_rankings_df.index)
    if len(common_models) == 0:
        return None

    available_tasks = similarity_df.columns.intersection(model_rankings_df.columns)
    source_tasks = [t for t in available_tasks if t != task_key]

    if not source_tasks:
        return None

    for model in common_models:
        sim_vec = similarity_df.loc[model, source_tasks].values
        rank_vec = model_rankings_df.loc[model, source_tasks].values

        valid_mask = ~np.isnan(rank_vec) & ~np.isnan(sim_vec)

        if np.sum(valid_mask) > 0:
            valid_sim = sim_vec[valid_mask]
            valid_rank = rank_vec[valid_mask]

            if np.sum(valid_sim) > 0:
                valid_sim = valid_sim / np.sum(valid_sim)

            score = np.sum(valid_sim * valid_rank)
            estimated_scores[model] = score
        else:
            estimated_scores[model] = np.inf

    result_df = pd.DataFrame(list(estimated_scores.items()), columns=['model', 'estimated_score'])
    result_df = result_df.sort_values(by='estimated_score', ascending=True)
    result_df['estimated_rank'] = range(1, len(result_df) + 1)
    result_df.to_csv(output_csv, index=False)
    return result_df


def calculate_ndcg_at_k(est_df, true_series, common_models, k=5):
    est_sub = est_df[est_df['model'].isin(common_models)].sort_values('estimated_rank', ascending=True)
    true_sub = true_series.loc[true_series.index.isin(common_models)].sort_values(ascending=True)

    pred_topk = est_sub.head(k).set_index('model')['estimated_rank']
    true_topk = true_sub.head(k)

    common_idx = pred_topk.index.intersection(true_topk.index)
    if len(common_idx) < 2:
        return 0.0

    common_idx = [m for m in pred_topk.index if m in common_idx]

    true_relevance = true_topk.loc[common_idx].rank(ascending=False).values.reshape(1, -1)
    pred_scores = (-pred_topk.loc[common_idx]).values.reshape(1, -1)

    return ndcg_score(true_relevance, pred_scores)


def calculate_kendall_tau_at_k(est_df, true_series, common_models, k=5):
    est_sub = est_df[est_df['model'].isin(common_models)].copy()
    true_sub = true_series.loc[common_models].copy()

    pred_topk = est_sub.sort_values(by='estimated_rank', ascending=True).head(k)
    pred_series = pred_topk.set_index('model')['estimated_rank']

    true_topk = true_sub.sort_values(ascending=True).head(k)

    common_idx = pred_series.index.intersection(true_topk.index)

    if len(common_idx) < 2:
        return 0.0

    vec_est = pred_series.loc[common_idx].values
    vec_true = true_topk.loc[common_idx].values

    tau, _ = kendalltau(vec_est, vec_true)
    return tau if not np.isnan(tau) else 0.0


def calculate_kendall_tau_full(est_df, true_series, common_models):
    est_sub = est_df[est_df['model'].isin(common_models)].set_index('model')['estimated_rank']
    true_sub = true_series.loc[common_models]

    common_idx = est_sub.index.intersection(true_sub.index)
    if len(common_idx) < 2:
        return 0.0

    vec_est = est_sub.loc[common_idx].values
    vec_true = true_sub.loc[common_idx].values

    tau, _ = kendalltau(vec_est, vec_true)
    return tau if not np.isnan(tau) else 0.0


def evaluate_task(task_name, est_df, gt_df):
    if task_name not in gt_df.columns:
        return None

    true_series = gt_df[task_name].dropna()
    common_models = list(set(est_df['model']) & set(true_series.index))

    if len(common_models) < 3:
        return None

    return {
        'ndcg@3': calculate_ndcg_at_k(est_df, true_series, common_models, k=3),
        'ndcg@5': calculate_ndcg_at_k(est_df, true_series, common_models, k=5),
        'ndcg@7': calculate_ndcg_at_k(est_df, true_series, common_models, k=7),
        'tau@5': calculate_kendall_tau_at_k(est_df, true_series, common_models, k=5),
        'tau': calculate_kendall_tau_full(est_df, true_series, common_models),
    }


def run_single_combination(target_num, source_num, gt_df, similarity_func, sim_suffix):
    target_results_dir = f'{RESULTS_DIR}/result_{target_num}'
    source_results_dir = f'{RESULTS_DIR}/result_{source_num}'
    sim_dir = f'./sim_{sim_suffix}_{target_num}_{source_num}'
    rank_dir = f'./rank_{sim_suffix}_{target_num}_{source_num}'

    if not os.path.exists(target_results_dir) or not os.path.exists(source_results_dir):
        print(f"[SKIP] ({target_num}, {source_num}): data dir not found")
        return None

    os.makedirs(sim_dir, exist_ok=True)
    os.makedirs(rank_dir, exist_ok=True)

    target_files = {os.path.basename(f): os.path.join(target_results_dir, f)
                    for f in os.listdir(target_results_dir) if f.endswith('.npy')}
    source_files = {os.path.basename(f): os.path.join(source_results_dir, f)
                    for f in os.listdir(source_results_dir) if f.endswith('.npy')}
    common_files = set(target_files.keys()) & set(source_files.keys())
    file_pairs = [(target_files[f], source_files[f]) for f in common_files]

    if not file_pairs:
        print(f"[SKIP] ({target_num}, {source_num}): no matching model files")
        return None

    task_results = []
    for task in TARGET_KEYS:
        sim_csv = f"{sim_dir}/sim_{task}.csv"
        rank_csv = f"{rank_dir}/rank_{task}.csv"

        if os.path.exists(rank_csv):
            est_df = pd.read_csv(rank_csv)
        else:
            sim_df = calculate_task_similarities(task, file_pairs, sim_csv, similarity_func)
            if sim_df is None:
                continue
            est_df = calculate_estimated_rankings(task, sim_df, gt_df, rank_csv)
            if est_df is None:
                continue

        metrics = evaluate_task(task, est_df, gt_df)
        if metrics:
            task_results.append(metrics)

    if not task_results:
        return None

    return {
        'target': target_num,
        'source': source_num,
        'ndcg@3': np.mean([r['ndcg@3'] for r in task_results]),
        'ndcg@5': np.mean([r['ndcg@5'] for r in task_results]),
        'ndcg@7': np.mean([r['ndcg@7'] for r in task_results]),
        'tau@5': np.mean([r['tau@5'] for r in task_results]),
        'tau': np.mean([r['tau'] for r in task_results]),
    }


def run_experiment(similarity_func, sim_suffix, output_excel):
    print("=" * 60)
    print(f"Experiment: {sim_suffix.upper()} Similarity (target=1)")
    print("=" * 60)

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"[ERROR] {GROUND_TRUTH_FILE} not found")
        return

    gt_df = preprocess_model_rankings(GROUND_TRUTH_FILE)
    print(f"[OK] Loaded Ground Truth: {len(gt_df)} models, {len(gt_df.columns)} tasks")

    target = 1
    source_list = [1, 5, 10, 15, 20, 25, 50, 75, 100]
    combinations = [(target, source) for source in source_list]

    print(f"[INFO] Total {len(combinations)} combinations to process")

    all_results = []
    for idx, (target, source) in enumerate(combinations):
        print(f"\n[{idx + 1}/{len(combinations)}] Processing target={target}, source={source}...")
        result = run_single_combination(target, source, gt_df, similarity_func, sim_suffix)
        if result:
            all_results.append(result)
            print(f"   [OK] ndcg@5={result['ndcg@5']:.4f}, tau={result['tau']:.4f}")

    if all_results:
        df = pd.DataFrame(all_results)
        df = df[['target', 'source', 'ndcg@3', 'ndcg@5', 'ndcg@7', 'tau@5', 'tau']]
        df.to_excel(output_excel, index=False)
        print(f"\n[DONE] Results saved to: {output_excel}")
        print(f"   Total {len(all_results)} rows")
        return df
    else:
        print("\n[WARN] No results generated")
        return None


def main():
    print("\n" + "=" * 70)
    print("Symmetric Similarity Experiment (target=1)")
    print("=" * 70 + "\n")
    
    print("\n>>> Running Cosine Similarity Experiment <<<\n")
    cos_results = run_experiment(cosine_similarity, 'cos', OUTPUT_EXCEL_COS)
    
    print("\n>>> Running Symmetric Soft-KL Experiment <<<\n")
    softkl_results = run_experiment(symmetric_soft_kl, 'softkl', OUTPUT_EXCEL_SOFTKL)
    
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    if cos_results is not None and softkl_results is not None:
        print("\n[Cosine Similarity - Mean over all combinations]")
        print(f"  NDCG@3: {cos_results['ndcg@3'].mean():.4f}")
        print(f"  NDCG@5: {cos_results['ndcg@5'].mean():.4f}")
        print(f"  NDCG@7: {cos_results['ndcg@7'].mean():.4f}")
        print(f"  Tau@5:  {cos_results['tau@5'].mean():.4f}")
        print(f"  Tau:    {cos_results['tau'].mean():.4f}")
        
        print("\n[Symmetric Soft-KL - Mean over all combinations]")
        print(f"  NDCG@3: {softkl_results['ndcg@3'].mean():.4f}")
        print(f"  NDCG@5: {softkl_results['ndcg@5'].mean():.4f}")
        print(f"  NDCG@7: {softkl_results['ndcg@7'].mean():.4f}")
        print(f"  Tau@5:  {softkl_results['tau@5'].mean():.4f}")
        print(f"  Tau:    {softkl_results['tau'].mean():.4f}")


if __name__ == '__main__':
    main()
