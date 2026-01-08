import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# === 中文字体支持 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("📊 动漫评分校准系统：加载 ml_anime_output 中的 ML 权重")
print("=" * 60)

# === 第一步：加载原始数据集 ===
data_path = Path("top_anime_dataset.csv")
if not data_path.exists():
    print("❌ 未找到 top_anime_dataset.csv，请确保数据集存在！")
    exit(1)

df = pd.read_csv(data_path)
print(f"✅ 加载原始数据: {df.shape[0]} 条记录")

# === 第二步：复用 v3.8 的预处理逻辑（必须一致！）===
print("\n🔄 执行与 v3.8 一致的数据预处理...")

df['scored_by'] = df['scored_by'].fillna(0)
df['favorites'] = df['favorites'].fillna(0)
df['score'] = df['score'].fillna(df['score'].mean())

# 主制作公司
df['main_studio'] = df['studios'].astype(str).str.split(', ').apply(
    lambda x: x[0].strip() if isinstance(x, list) and len(x) > 0 else 'Unknown'
).replace(['nan', 'None', ''], 'Unknown')

# studio_score
studio_avg = df.groupby('main_studio')['score'].mean().to_dict()
df['studio_score'] = df['main_studio'].map(studio_avg).fillna(0.5)

# source_score
source_weights = {'Manga': 0.9, 'Novel': 0.7, 'Game': 0.5, 'Original': 0.3}
df['source_score'] = df['source'].map(source_weights).fillna(0.3)

# Heat
max_sb, max_fav = df['scored_by'].max(), df['favorites'].max()
df['Heat'] = (df['scored_by'] + df['favorites']) / (max_sb + max_fav + 1e-8)

# Rating_Score
min_sc, max_sc = df['score'].min(), df['score'].max()
df['Rating_Score'] = (df['score'] - min_sc) / (max_sc - min_sc + 1e-8)

# Rank_Score
if 'rank' in df.columns:
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank']).reset_index(drop=True)
    min_r, max_r = df['rank'].min(), df['rank'].max()
    df['Rank_Score'] = 1 - (df['rank'] - min_r) / (max_r - min_r + 1e-8)
else:
    df['Rank_Score'] = 0.5

# 原 Final_Score（用于对比）
df['Final_Score'] = (
    0.30 * df['Heat'] +
    0.25 * df['Rating_Score'] +
    0.15 * df['studio_score'] +
    0.10 * df['source_score'] +
    0.20 * df['Rank_Score']
)

# === 第三步：从 ml_anime_output 加载 ML 模型结果 ===
ml_output_dir = Path("ml_anime_output")
ml_score_path = ml_output_dir / "anime_with_ml_score.csv"
feat_imp_path = ml_output_dir / "feature_importance.csv"

if not ml_output_dir.exists():
    print(f"❌ 错误：目录 '{ml_output_dir}' 不存在！请先运行 ML 脚本生成结果。")
    exit(1)

# 尝试直接加载 ML_Score
if ml_score_path.exists():
    print("✅ 直接加载 ml_anime_output/anime_with_ml_score.csv 中的 ML_Score...")
    df_ml = pd.read_csv(ml_score_path)
    if 'name' in df.columns and 'name' in df_ml.columns:
        df = df.merge(df_ml[['name', 'ML_Score']], on='name', how='left')
    else:
        # 若无 name，按顺序合并（需确保行对齐）
        df['ML_Score'] = df_ml['ML_Score'].values[:len(df)]
else:
    print("⚠️ anime_with_ml_score.csv 不存在，将用 feature_importance.csv 重建 ML_Score...")

    if not feat_imp_path.exists():
        print(f"❌ 错误：{feat_imp_path} 不存在！无法重建 ML_Score。")
        exit(1)

    # 加载系数
    feat_df = pd.read_csv(feat_imp_path)
    coef_dict = dict(zip(feat_df['Feature'], feat_df['Coefficient']))
    intercept = 0.0  # 可扩展：从 summary.txt 读取截距

    # 处理 genres（多标签）
    df['genres_list'] = df['genres'].astype(str).apply(
        lambda x: [g.strip() for g in x.split(',')] if pd.notna(x) and x != 'nan' else []
    )
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_mat = mlb.fit_transform(df['genres_list'])
    genre_df = pd.DataFrame(genre_mat, columns=mlb.classes_, index=df.index)

    # 构建数值特征
    numeric_cols = ['Heat', 'score', 'studio_score', 'source_score']
    X_num = df[numeric_cols].copy()
    X = pd.concat([X_num, genre_df], axis=1)

    # 确保所有 ML 特征都存在
    for col in mlb.classes_:
        if col not in X.columns:
            X[col] = 0
    for col in numeric_cols:
        if col not in X.columns:
            X[col] = 0

    # 计算 ML_Score = Σ(coef * feature)
    score_series = pd.Series(intercept, index=df.index)
    for feat, coef in coef_dict.items():
        if feat in X.columns:
            score_series += coef * X[feat]
        else:
            print(f"⚠️ 警告：特征 '{feat}' 缺失，跳过（系数={coef:.4f}）")
    df['ML_Score'] = score_series

# 填充缺失值（极端情况）
df['ML_Score'] = df['ML_Score'].fillna(df['ML_Score'].mean())

print(f"✅ ML_Score 范围: [{df['ML_Score'].min():.4f}, {df['ML_Score'].max():.4f}]")
print(f"✅ Final_Score 范围: [{df['Final_Score'].min():.4f}, {df['Final_Score'].max():.4f}]")

# === 第四步：计算差异 ===
df['Score_Diff'] = df['ML_Score'] - df['Final_Score']
df['Abs_Diff'] = df['Score_Diff'].abs()

# === 第五步：保存输出（到 output/）===
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 全量对比
df[['name', 'Final_Score', 'ML_Score', 'Score_Diff', 'genres', 'main_studio']].to_csv(
    output_dir / "calibrated_comparison.csv", index=False
)

# 上升最多（被低估）
rising = df.nlargest(20, 'Score_Diff')[['name', 'Final_Score', 'ML_Score', 'Score_Diff', 'genres', 'main_studio']]
falling = df.nsmallest(20, 'Score_Diff')[['name', 'Final_Score', 'ML_Score', 'Score_Diff', 'genres', 'main_studio']]

rising.to_csv(output_dir / "rising_titles.csv", index=False)
falling.to_csv(output_dir / "falling_titles.csv", index=False)

# === 第六步：可视化 ===
# 散点图
plt.figure(figsize=(10, 6))
sc = plt.scatter(df['Final_Score'], df['ML_Score'], c=df['Heat'], cmap='viridis', alpha=0.6)
plt.plot([df['Final_Score'].min(), df['Final_Score'].max()],
         [df['Final_Score'].min(), df['Final_Score'].max()], 'r--', lw=2)
plt.xlabel('原 Final_Score')
plt.ylabel('ML 校准后 Score')
plt.title('评分校准对比（颜色 = 热度）')
plt.colorbar(sc, label='Heat')
plt.tight_layout()
plt.savefig(output_dir / "calibration_scatter.png", dpi=150)

# 差异直方图
plt.figure(figsize=(8, 5))
plt.hist(df['Score_Diff'], bins=50, color='skyblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('ML_Score - Final_Score')
plt.ylabel('频次')
plt.title('评分差异分布')
plt.tight_layout()
plt.savefig(output_dir / "score_diff_hist.png", dpi=150)

# Top 变化条形图
top_up = df.nlargest(10, 'Score_Diff')
top_down = df.nsmallest(10, 'Score_Diff')

plt.figure(figsize=(10, 8))
y = np.arange(10)
plt.barh(y + 0.2, top_up['Score_Diff'], height=0.4, label='评分上升（被低估）', color='green')
plt.barh(y - 0.2, top_down['Score_Diff'], height=0.4, label='评分下降（被高估）', color='red')
plt.yticks(y, [f"{a[:25]} / {b[:25]}" for a, b in zip(top_up['name'], top_down['name'])])
plt.xlabel('评分变化 (ML - 原)')
plt.title('Top 10 评分变化作品')
plt.legend()
plt.tight_layout()
plt.savefig(output_dir / "top_changes.png", dpi=150)

# === 第七步：打印摘要 ===
print("\n" + "=" * 60)
print("📈 校准结果摘要")
print("=" * 60)
print(f"平均 Final_Score: {df['Final_Score'].mean():.4f}")
print(f"平均 ML_Score:    {df['ML_Score'].mean():.4f}")
print(f"平均绝对差异:     {df['Abs_Diff'].mean():.4f}")
print(f"最大上升:         {df['Score_Diff'].max():+.4f}")
print(f"最大下降:         {df['Score_Diff'].min():+.4f}")

print(f"\n✅ 对比结果已保存至: {output_dir.resolve()}")
print("- calibrated_comparison.csv: 全量对比")
print("- rising_titles.csv: 被 ML 高看的作品（小众佳作？）")
print("- falling_titles.csv: 被 ML 低看的作品（热度泡沫？）")
print("- calibration_scatter.png / score_diff_hist.png / top_changes.png")

print("\n💡 建议：")
print("- 查看 rising_titles.csv，挖掘高质量冷门番剧")
print("- 结合 feature_importance.csv，理解类型偏好（如 'Slice of Life' 是否被高估？）")

print("=" * 60)