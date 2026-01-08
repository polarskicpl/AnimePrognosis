# === 第一步：检查依赖（新增 sklearn）===
print("=" * 50)
print("步骤0：检查依赖库（含机器学习）")
print("=" * 50)

required_libs = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'sklearn': 'scikit-learn'
}

for name, pkg in required_libs.items():
    try:
        __import__(name if name != 'sklearn' else 'sklearn')
        print(f"✅ {name} 已安装")
    except ImportError:
        print(f"❌ 缺少 {name}，请运行：pip install {pkg}")
        exit(1)

# === 第二步：导入库 ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# === 中文字体 ===
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# === 第三步：加载并复用 v3.8 的预处理逻辑 ===
print("\n" + "=" * 50)
print("步骤1：加载数据并执行基础预处理")
print("=" * 50)

# --- 数据加载 ---
dataset_path = Path("top_anime_dataset.csv")
if not dataset_path.exists():
    for p in ["../top_anime_dataset.csv", "data/top_anime_dataset.csv"]:
        if Path(p).exists():
            dataset_path = Path(p)
            break
df = pd.read_csv(dataset_path)
print(f"✅ 加载 {df.shape[0]} 条记录")

# --- 基础清洗 ---
df['scored_by'] = df['scored_by'].fillna(0)
df['favorites'] = df['favorites'].fillna(0)
df['score'] = df['score'].fillna(df['score'].mean())

# --- 提取主制作公司 ---
df['main_studio'] = df['studios'].astype(str).str.split(', ').apply(
    lambda x: x[0].strip() if isinstance(x, list) and len(x) > 0 else 'Unknown'
).replace(['nan', 'None', ''], 'Unknown')

# --- studio_score ---
studio_avg = df.groupby('main_studio')['score'].mean().to_dict()
df['studio_score'] = df['main_studio'].map(studio_avg).fillna(0.5)

# --- source_score ---
source_weights = {'Manga': 0.9, 'Novel': 0.7, 'Game': 0.5, 'Original': 0.3}
df['source_score'] = df['source'].map(source_weights).fillna(0.3)

# --- Heat ---
max_sb = df['scored_by'].max()
max_fav = df['favorites'].max()
df['Heat'] = (df['scored_by'] + df['favorites']) / (max_sb + max_fav + 1e-8)

# --- Rank_Score（如果存在 rank）---
if 'rank' in df.columns:
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank']).reset_index(drop=True)
    min_r, max_r = df['rank'].min(), df['rank'].max()
    df['Rank_Score'] = 1 - (df['rank'] - min_r) / (max_r - min_r + 1e-8) if max_r > min_r else 1.0
else:
    df['Rank_Score'] = 0.5

# --- Final_Score（作为监督信号 y）---
df['Final_Score'] = (
    0.30 * df['Heat'] +
    0.25 * (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min() + 1e-8) +
    0.15 * df['studio_score'] +
    0.10 * df['source_score'] +
    0.20 * df['Rank_Score']
)

# === 第四步：构建 ML 特征矩阵 ===
print("\n" + "=" * 50)
print("步骤2：构建机器学习特征")
print("=" * 50)

# 4.1 数值特征
numeric_features = df[['Heat', 'score', 'studio_score', 'source_score']].copy()

# 4.2 类型特征（genres）
print("正在处理 genres...")
df['genres_list'] = df['genres'].astype(str).apply(
    lambda x: [g.strip() for g in x.split(',')] if pd.notna(x) and x != 'nan' else []
)
mlb = MultiLabelBinarizer()
genre_features = mlb.fit_transform(df['genres_list'])
genre_df = pd.DataFrame(genre_features, columns=mlb.classes_, index=df.index)
print(f"共提取 {len(mlb.classes_)} 种动漫类型")

# 4.3 合并特征
X = pd.concat([numeric_features, genre_df], axis=1)
y = df['Final_Score']

print(f"特征矩阵形状: {X.shape} (样本数 × 特征数)")
print(f"前5个特征名: {list(X.columns[:5])}")

# === 第五步：训练 Ridge 回归模型 ===
print("\n" + "=" * 50)
print("步骤3：训练 Ridge 回归模型")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 Ridge（带 L2 正则化，防止过拟合）
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# 预测
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# 评估
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print(f"训练集 R²: {r2_train:.4f}")
print(f"测试集 R²: {r2_test:.4f}")
print(f"测试集 MAE: {mae_test:.4f}")

# === 第六步：提取新评分公式 ===
print("\n" + "=" * 50)
print("步骤4：生成机器学习优化的评分公式")
print("=" * 50)

# 获取系数
coefficients = pd.Series(model.coef_, index=X.columns)
intercept = model.intercept_

# 分离数值特征和类型特征
num_coeffs = coefficients[['Heat', 'score', 'studio_score', 'source_score']]
genre_coeffs = coefficients.drop(['Heat', 'score', 'studio_score', 'source_score']).sort_values(key=abs, ascending=False)

print("📊 新评分公式（线性组合）:")
print(f"ML_Score = {intercept:.4f}")
for feat, coef in num_coeffs.items():
    print(f"           + ({coef:.4f}) × {feat}")

print(f"\n🔍 前10个最重要的动漫类型（按系数绝对值）:")
print(genre_coeffs.head(10))

# 将 ML_Score 添加到原数据
df['ML_Score'] = model.predict(X)

# === 第七步：保存结果 ===
output_dir = Path("ml_anime_output")
output_dir.mkdir(exist_ok=True)

# 保存完整数据（含 ML_Score）
df.to_csv(output_dir / "anime_with_ml_score.csv", index=False)

# 保存特征重要性
feature_imp = pd.DataFrame({
    'Feature': coefficients.index,
    'Coefficient': coefficients.values
}).sort_values('Coefficient', key=abs, ascending=False)
feature_imp.to_csv(output_dir / "feature_importance.csv", index=False)

# 保存新公式摘要
with open(output_dir / "ml_formula_summary.txt", "w", encoding='utf-8') as f:
    f.write("机器学习优化的动漫评分公式\n")
    f.write("="*40 + "\n")
    f.write(f"ML_Score = {intercept:.6f}\n")
    for feat, coef in num_coeffs.items():
        f.write(f"         + ({coef:+.6f}) * {feat}\n")
    f.write("\n前10重要类型:\n")
    for i, (genre, coef) in enumerate(genre_coeffs.head(10).items(), 1):
        f.write(f"{i:2d}. {genre:<20} : {coef:+.6f}\n")

print(f"\n✅ 结果已保存至: {output_dir}")
print("- anime_with_ml_score.csv: 全量数据（含 ML_Score）")
print("- feature_importance.csv: 所有特征系数")
print("- ml_formula_summary.txt: 可读公式摘要")

# === 第八步：可视化 ===
print("\n" + "=" * 50)
print("步骤5：生成可视化图表")
print("=" * 50)

# 图1：真实 vs 预测
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('真实 Final_Score')
plt.ylabel('预测 ML_Score')
plt.title('模型预测效果（测试集）')
plt.tight_layout()
plt.savefig(output_dir / "ml_prediction.png", dpi=150)

# 图2：类型重要性（前15）
plt.figure(figsize=(10, 6))
top_genres = genre_coeffs.head(15)
colors = ['red' if c < 0 else 'blue' for c in top_genres]
plt.barh(range(len(top_genres)), top_genres, color=colors)
plt.yticks(range(len(top_genres)), top_genres.index)
plt.xlabel('系数（正：加分，负：减分）')
plt.title('动漫类型对评分的影响（ML 模型）')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(output_dir / "genre_importance.png", dpi=150)

print("✅ 可视化图表已保存")

# === 最终输出 ===
print("\n" + "=" * 50)
print("机器学习优化完成！")
print("=" * 50)
print("💡 核心发现：")
print(f"- 模型解释力 (R²): {r2_test:.2%}")
print(f"- 平均预测误差 (MAE): {mae_test:.4f}")
print(f"- 关键正向类型: {', '.join(genre_coeffs.head(3).index)}")
print(f"- 关键负向类型: {', '.join(genre_coeffs.tail(3).index)}")
print("\n📌 使用建议：")
print("- 可直接使用 ML_Score 作为新排名依据")
print("- 公式可嵌入业务系统进行实时评分")
print("- 类型系数可用于内容推荐或创作指导")
print("=====================================")