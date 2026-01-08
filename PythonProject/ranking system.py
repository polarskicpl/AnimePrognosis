# === 第一步：检查并安装必要依赖库 ===
print("=" * 50)
print("步骤0：检查并安装必要依赖库")
print("=" * 50)

try:
    import matplotlib
    print("✅ matplotlib 已安装")
except ImportError:
    print("❌ 缺少 matplotlib 库，请运行以下命令安装：")
    print("pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple")
    exit(1)

try:
    import seaborn
    print("✅ seaborn 已安装")
except ImportError:
    print("❌ 缺少 seaborn 库，请运行以下命令安装：")
    print("pip install seaborn -i https://pypi.tuna.tsinghua.edu.cn/simple")
    exit(1)

try:
    import pandas
    print("✅ pandas 已安装")
except ImportError:
    print("❌ 缺少 pandas 库，请运行以下命令安装：")
    print("pip install pandas -i https://pypi.tuna.tsinghua.edu.cn/simple")
    exit(1)

# === 第二步：导入必要库 ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

# === 中文字体支持 ===
print("\n" + "=" * 50)
print("步骤0.5：修复中文显示问题")
print("=" * 50)

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False
print("✅ 中文字体已设置成功！")
print("=" * 50)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

# === 第三步：加载数据集 ===
print("\n" + "=" * 50)
print("步骤1：加载数据集")
print("=" * 50)

dataset_path = Path("top_anime_dataset.csv")
if not dataset_path.exists():
    dataset_path = Path("../top_anime_dataset.csv")
    if not dataset_path.exists():
        dataset_path = Path("data/top_anime_dataset.csv")
        if not dataset_path.exists():
            print("\n⚠️ 警告：数据集文件未找到，创建含 rank 的示例数据...")
            sample_data = {
                'name': ['Anime1', 'Anime2', 'Anime3'],
                'score': [8.5, 9.0, 7.8],
                'scored_by': [1000, 2000, 500],
                'favorites': [200, 300, 100],
                'source': ['Manga', 'Original', 'Novel'],
                'producers': ['Aniplex', 'Bandai', 'Toho'],
                'studios': ['Madhouse', 'Kyoto Animation', 'Shaft'],
                'rank': [1, 2, 3]  # ← 关键：包含原始排名
            }
            df_sample = pd.DataFrame(sample_data)
            df_sample.to_csv(dataset_path, index=False)
            print(f"✅ 示例数据集已创建: {dataset_path}")
            df = df_sample
        else:
            df = pd.read_csv(dataset_path)
    else:
        df = pd.read_csv(dataset_path)
else:
    df = pd.read_csv(dataset_path)

print(f"数据集加载成功！共 {df.shape[0]} 条记录，{df.shape[1]} 个字段")
print("前3行预览：")
print(df.head(3))

# === 第四步：数据预处理 ===
print("\n" + "=" * 50)
print("步骤2：数据预处理（分离 studios/producers + 处理 rank）")
print("=" * 50)

# 4.1 基础缺失值处理
df['scored_by'] = df['scored_by'].fillna(0)
df['favorites'] = df['favorites'].fillna(0)
df['score'] = df['score'].fillna(df['score'].mean())

# 4.2 提取主制作公司（studios）
df['main_studio'] = df['studios'].astype(str).str.split(', ').apply(
    lambda x: x[0].strip() if isinstance(x, list) and len(x) > 0 else 'Unknown'
)
df['main_studio'] = df['main_studio'].replace(['nan', 'None', ''], 'Unknown')

# 4.3 提取主发行商（仅展示）
df['main_producer'] = df['producers'].astype(str).str.split(', ').apply(
    lambda x: x[0].strip() if isinstance(x, list) and len(x) > 0 else 'Unknown'
)
df['main_producer'] = df['main_producer'].replace(['nan', 'None', ''], 'Unknown')

# 4.4 原作类型量化
source_weights = {'Manga': 0.9, 'Novel': 0.7, 'Game': 0.5, 'Original': 0.3}
df['source_score'] = df['source'].map(source_weights).fillna(0.3)

# 4.5 计算 studio_score
all_studios = df[df['main_studio'] != 'Unknown']['main_studio'].unique()
studio_scores = {}
for studio in all_studios:
    avg = df[df['main_studio'] == studio]['score'].mean()
    studio_scores[studio] = avg
df['studio_score'] = df['main_studio'].map(studio_scores).fillna(0.5)

# 4.6 处理原始 rank（关键新增）
use_rank_score = True
if 'rank' in df.columns and not df['rank'].isnull().all():
    print("\n检测到原始 rank 列，正在计算 Rank_Score...")
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df = df.dropna(subset=['rank'])  # 移除 rank 无效行（可选）
    df = df.reset_index(drop=True)

    min_rank = df['rank'].min()
    max_rank = df['rank'].max()
    if min_rank == max_rank:
        df['Rank_Score'] = 1.0
    else:
        # 反向归一化：rank 越小，得分越高
        df['Rank_Score'] = 1 - (df['rank'] - min_rank) / (max_rank - min_rank)
    print(f"Rank_Score 范围: [{df['Rank_Score'].min():.4f}, {df['Rank_Score'].max():.4f}]")
else:
    print("\n⚠️ 未检测到有效 rank 列，Rank_Score 设为常量 0.5")
    df['Rank_Score'] = 0.5
    use_rank_score = False

# === 第五步：计算核心指标 ===
print("\n" + "=" * 50)
print("步骤3：计算核心指标（含 Rank_Score）")
print("=" * 50)

# Heat
max_sb = df['scored_by'].max()
max_fav = df['favorites'].max()
df['Heat'] = (df['scored_by'] + df['favorites']) / (max_sb + max_fav) if (max_sb + max_fav) > 0 else 0

# Rating_Score
min_sc = df['score'].min()
max_sc = df['score'].max()
df['Rating_Score'] = (df['score'] - min_sc) / (max_sc - min_sc) if max_sc != min_sc else 0.5

# Final_Score（新增 Rank_Score 权重）
df['Final_Score'] = (
        0.30 * df['Heat'] +
        0.25 * df['Rating_Score'] +
        0.15 * df['studio_score'] +
        0.10 * df['source_score'] +
        0.20 * df['Rank_Score']  # ← 新增：原始排名影响力
)

print("最终评分范围：[%.2f, %.2f]" % (df['Final_Score'].min(), df['Final_Score'].max()))

# === 第六步：生成榜单 ===
print("\n" + "=" * 50)
print("步骤4：生成四个榜单")
print("=" * 50)

heat_p = df['Heat'].quantile([0.30, 0.70])
final_p = df['Final_Score'].quantile([0.85])

# 大众榜
popular_top = df[
    (df['Final_Score'] > final_p[0.85]) &
    (df['Heat'] >= heat_p[0.70])
    ].sort_values('Final_Score', ascending=False).reset_index(drop=True)
popular_top['Rank'] = range(1, len(popular_top) + 1)

# 小众榜
niche_top = df[
    (df['Final_Score'] > final_p[0.85]) &
    (df['Heat'] < heat_p[0.30])
    ].sort_values('Final_Score', ascending=False).reset_index(drop=True)
niche_top['Rank'] = range(1, len(niche_top) + 1)

# 制作公司榜（作品>3）
studio_stats = df.groupby('main_studio').agg(
    Avg_Final_Score=('Final_Score', 'mean'),
    Count=('name', 'count')
).reset_index()
studio_rank = studio_stats[studio_stats['Count'] > 3].sort_values('Avg_Final_Score', ascending=False).reset_index(
    drop=True)
studio_rank['Rank'] = range(1, len(studio_rank) + 1)

# 导演榜（复用 studio_rank）
director_rank = studio_rank.copy()
director_rank.columns = ['Director', 'Avg_Final_Score', 'Count', 'Rank']

# === 第七步：保存结果（统一到项目目录的 output 文件夹）===
print("\n" + "=" * 50)
print("步骤5：保存结果到项目目录下的 output 文件夹")
print("=" * 50)

# 强制使用项目内的 output 目录
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 保存榜单（包含 rank 和 main_studio/main_producer）
cols_common = ['Rank', 'name', 'Final_Score', 'Heat', 'source', 'main_studio', 'main_producer']
if 'rank' in df.columns:
    cols_common.insert(2, 'rank')  # 在 Final_Score 前插入原始 rank

popular_top[cols_common].to_csv(output_dir / "popular_top.csv", index=False)
niche_top[cols_common].to_csv(output_dir / "niche_top.csv", index=False)
studio_rank[['Rank', 'main_studio', 'Avg_Final_Score', 'Count']].to_csv(output_dir / "studio_rank.csv", index=False)
director_rank.to_csv(output_dir / "director_rank.csv", index=False)

print(f"✅ 结果已保存至: {output_dir.resolve()}")

# === 第八步：可视化 ===
print("\n" + "=" * 50)
print("步骤6：生成可视化图表")
print("=" * 50)

# 图1：热度 vs 评分
plt.figure(figsize=(10, 6))
ax = sns.scatterplot(data=df, x='Heat', y='Rating_Score', hue='Final_Score',
                     size='Final_Score', palette='viridis', alpha=0.7, sizes=(50, 500))
plt.title('热度 vs 评分（含原始排名加权）', fontsize=14)
plt.xlabel('热度 (Heat)')
plt.ylabel('评分标准化 (Rating_Score)')
plt.colorbar(ax.collections[0], label='Final_Score')
plt.tight_layout()
plt.savefig(output_dir / "heat_vs_rating.png", dpi=150)

# 图2：榜单分布
plt.figure(figsize=(8, 8))
sizes = [len(popular_top), len(niche_top), len(df) - len(popular_top) - len(niche_top)]
plt.pie(sizes, labels=['大众榜', '小众榜', '其他'], autopct='%1.1f%%', startangle=90,
        colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('榜单分布（含 rank 加权）')
plt.tight_layout()
plt.savefig(output_dir / "ranking_distribution.png", dpi=150)

# 图3：制作公司数量
plt.figure(figsize=(12, 6))
counts = df['main_studio'].value_counts()
plt.bar(counts.index[:20], counts[:20], color='lightcoral')
plt.title('制作公司作品数量（前20）')
plt.xlabel('制作公司')
plt.ylabel('作品数')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(output_dir / "studio_counts.png", dpi=150)

# 图4：过滤说明
valid = len(studio_rank)
total_unique = df['main_studio'].nunique()
plt.figure(figsize=(8, 6))
plt.pie([valid, total_unique - valid],
        labels=['有效公司（>3部）', '过滤公司（≤3部）'],
        autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#d62728'])
plt.title('制作公司过滤说明')
plt.tight_layout()
plt.savefig(output_dir / "studio_filter.png", dpi=150)

print("✅ 所有图表已保存")

# === 最终输出 ===
print("\n" + "=" * 50)
print("执行完成！")
print("=" * 50)
print(f"大众榜: {len(popular_top)} 部")
print(f"小众榜: {len(niche_top)} 部")
print(f"制作公司榜: {len(studio_rank)} 家（作品>3）")
print("\n📌 关键改进：")
print("- 制作公司来自 'studios'（非 producers）")
print("- 原始 'rank' 已转换为 Rank_Score 并占 Final_Score 20% 权重")
print("- 权重分配：Heat(30%) + Rating(25%) + Studio(15%) + Source(10%) + Rank(20%)")
print("- 所有输出保存在项目目录的 ./output/ 文件夹中")
print("=====================================")