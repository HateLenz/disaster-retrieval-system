# 数据集与处理

## 数据集来源

本项目以 xBD（xView2 Building Damage Assessment）遥感灾害数据集为基础，原始数据位于 `data/raw/xbd/geotiffs/` 目录下，按 `hold`、`test`、`tier1`、`tier3` 四个划分组织。每个样本同时包含：

- 灾前遥感影像 `*_pre_disaster.tif`
- 灾后遥感影像 `*_post_disaster.tif`
- 灾前建筑标注 `*_pre_disaster.json`
- 灾后建筑标注 `*_post_disaster.json`

标注文件中给出了建筑物多边形、建筑唯一标识 `uid` 以及灾后损伤类别。项目在构建正式实验数据时，仅保留 `fire`、`flooding` 和 `wind` 三类灾害，以保证检索任务的类别一致性。

下面的说明主要依据两个处理中间文件：

- `data/processed/xbd_tile_dataset.csv`
- `data/processed/xbd_building_patches.csv`

其中，瓦片级 CSV 的统计是稳定的；建筑级 patch CSV 在重新执行 `build_building_patches.py` 时会被增量重写，因此其总量可能随构建进度变化。论文或报告中若需要最终精确数字，应在 patch 构建完成后再做一次汇总统计。

## 原始 xBD 数据特征

结合本地原始拷贝，xBD 在本项目中的基础组织形式可以概括为“灾前/灾后影像对 + 灾前/灾后标注对”。每个 tile 样本对应 4 个文件：

- `images/*_pre_disaster.tif`
- `images/*_post_disaster.tif`
- `labels/*_pre_disaster.json`
- `labels/*_post_disaster.json`

从规模上看，原始数据共包含 **11,034** 对 tile 样本，即 **44,136** 个文件，其中包括 **22,068** 个 GeoTIFF 和 **22,068** 个 JSON 标注文件。影像文件总量约 **129.57 GB**，标注文件总量约 **753.17 MB**。所有影像的空间尺寸统一为 **1024 × 1024**，这为后续进行配对裁剪、固定尺寸缩放和批量训练提供了较好的基础。

按数据划分统计，原始 xBD 本地拷贝包括：

- `hold`: 933 对
- `test`: 933 对
- `tier1`: 2,799 对
- `tier3`: 6,369 对

需要注意的是，这份本地数据中未发现 `tier2`，因此项目实际也是围绕上述 4 个 split 组织和处理数据。

## 原始标注中的 Feature

xBD 原始 JSON 标注的顶层由 `metadata` 和 `features` 两部分组成，这也是本项目进行二次加工的直接来源。

### 1. metadata 特征

`metadata` 记录的是样本级别的场景信息，典型字段包括：

- `disaster`：具体灾害事件名称，如 `santa-rosa-wildfire`
- `disaster_type`：灾害类别，如 `fire`、`flooding`、`wind`
- `sensor`：遥感传感器来源
- `capture_date`：影像采集时间
- `gsd`：地面采样距离等成像参数

这些字段中，项目当前直接使用的是 `disaster` 和 `disaster_type`，前者用于保留事件级语义，后者用于灾害类型筛选、数据统计和分组评估；`sensor`、`capture_date` 和 `gsd` 虽然暂未直接输入模型，但它们保留了重要的数据来源和成像背景信息，适合在误差分析、域偏移讨论和后续扩展实验中使用。

### 2. features 特征

`features.xy` 存储的是建筑物级别的几何与属性标注，是本项目最核心的原始 feature 来源。每个建筑实例至少包含以下几类信息：

- `wkt`：建筑物多边形的 WKT 表示
- `properties.uid`：建筑实例的唯一标识
- `properties.feature_type`：目标类型，本项目只保留 `building`
- `properties.subtype`：灾后损伤类别

其中，`pre_disaster.json` 中建筑的 `subtype` 通常为空，因为灾前不存在损伤分级；`post_disaster.json` 中则给出了 5 类损伤标记：

- `no-damage`
- `minor-damage`
- `major-damage`
- `destroyed`
- `un-classified`

在项目实现中，这些原始 feature 被映射为后续训练所需的结构化字段：`uid` 用于灾前和灾后建筑配对，`wkt` 用于解析建筑多边形和边界框，`subtype` 用于生成损伤类别标签，`feature_type` 用于过滤非建筑对象。

## 原始灾害构成与项目筛选

原始 xBD 共覆盖 **19** 个具体灾害事件、**6** 类灾害类型：

- `fire`
- `flooding`
- `wind`
- `tsunami`
- `earthquake`
- `volcano`

按 tile 对统计的原始灾害类型分布如下：

- `fire`: 6,372
- `flooding`: 2,132
- `wind`: 1,674
- `tsunami`: 344
- `volcano`: 319
- `earthquake`: 193

可以看出，原始数据在灾害类型上并不均衡，`fire` 占比显著更高。项目最终只保留 `fire`、`flooding` 和 `wind` 三类灾害，因此从原始 **11,034** 对 tile 中筛选出 **10,178** 对，剔除了 `tsunami`、`volcano` 和 `earthquake` 共 **856** 对样本。这样的筛选主要有两个原因：一是保留样本量相对充足的主干灾害类型，二是让建筑级跨时相检索任务聚焦在更稳定、更可比较的灾后损伤场景上。

从具体事件来看，原始数据包含多个典型灾害案例，例如：

- 火灾：`socal-fire`、`santa-rosa-wildfire`、`pinery-bushfire`、`portugal-wildfire`
- 洪水：`hurricane-harvey`、`hurricane-florence`、`midwest-flooding`、`nepal-flooding`
- 风灾：`hurricane-michael`、`moore-tornado`、`joplin-tornado`
- 海啸：`palu-tsunami`、`sunda-tsunami`
- 地震：`mexico-earthquake`
- 火山：`guatemala-volcano`、`lower-puna-volcano`

这说明 xBD 原始数据本质上不仅是类别标签集合，更是一个包含多事件、多区域、多灾种的遥感灾害场景库。

## 从原始 Feature 到项目特征

本项目不是直接使用原始 JSON，而是在原始 feature 基础上构建了两层特征表示。

### 1. 瓦片级特征

项目从原始 `metadata` 和建筑标注中提取并构造了以下瓦片级特征：

- 事件与类别特征：`disaster`、`disaster_type`、`split`
- 路径特征：`pre_image_path`、`post_image_path`、`pre_label_path`、`post_label_path`
- 统计特征：`num_features`、`num_no_damage`、`num_minor_damage`、`num_major_damage`、`num_destroyed`、`num_unclassified`
- 比例特征：`damage_ratio`、`severe_damage_ratio`
- 文本特征：`text_prompt`

这些特征使得每个 tile 不再只是原始影像对，而是带有灾害语义、建筑数量和损伤强度的结构化样本。

### 2. 建筑级特征

在建筑级数据集中，项目进一步从原始 feature 派生出：

- 配对特征：`building_id`、`positive_id`、`building_uid`
- 几何特征：`polygon_wkt`、`bbox_x1`、`bbox_y1`、`bbox_x2`、`bbox_y2`
- 时相框特征：`pre_bbox_*`、`post_bbox_*`
- 裁剪特征：`crop_x1`、`crop_y1`、`crop_x2`、`crop_y2`、`crop_width`、`crop_height`
- 监督标签：`damage_label`、`damage_id`

也就是说，项目最终使用的数据集已经从“原始遥感标注文件”转化为“可直接支持检索模型训练与评估的结构化特征集”。这一步是项目数据工程中的关键环节。

## 瓦片级数据构建

瓦片级元数据由 `scripts/build_metadata.py` 生成，每一行对应一对灾前/灾后遥感瓦片。处理流程如下：

1. 遍历 `geotiffs` 目录下各划分中的灾后标注文件。
2. 读取 `metadata.disaster` 与 `metadata.disaster_type`，过滤出 `fire`、`flooding`、`wind` 三类灾害。
3. 根据建筑多边形标注统计每个瓦片中的建筑数量及损伤分布。
4. 生成训练可直接读取的 CSV，保存影像路径、标注路径、数据划分、损伤统计量和文本提示。

瓦片级 CSV 包含以下核心字段：

- 标识与路径：`tile_id`、`pre_image_path`、`post_image_path`、`pre_label_path`、`post_label_path`
- 场景信息：`disaster`、`disaster_type`、`split`
- 损伤统计：`num_features`、`num_no_damage`、`num_minor_damage`、`num_major_damage`、`num_destroyed`、`num_unclassified`
- 派生指标：`num_classified_features`、`damage_ratio`、`severe_damage_ratio`
- 文本描述：`text_prompt`，格式为 `a post-disaster satellite tile after {disaster_type}`

当前瓦片级数据共 **10,178** 条，按划分和灾害类型统计如下：

| split | fire | flooding | wind | total |
| --- | ---: | ---: | ---: | ---: |
| hold | 350 | 300 | 203 | 853 |
| test | 381 | 296 | 171 | 848 |
| tier1 | 1049 | 917 | 581 | 2547 |
| tier3 | 4592 | 619 | 719 | 5930 |
| total | 6372 | 2132 | 1674 | 10178 |

其中，`damage_ratio` 表示轻度及以上受损建筑占已分类建筑的比例，`severe_damage_ratio` 表示重度损坏和完全毁坏建筑占已分类建筑的比例。这两个指标为后续检索和难例分析提供了场景级先验信息。

## 建筑级配对样本构建

本项目的核心检索任务在建筑级别进行，因此在瓦片级元数据基础上，进一步使用 `scripts/build_building_patches.py` 构建建筑配对 patch 数据集。其处理流程如下：

1. 读取同一瓦片对应的灾前/灾后标注文件。
2. 从 JSON 标注中提取建筑多边形、边界框、损伤标签和建筑唯一标识 `uid`。
3. 以 `uid` 为键进行灾前与灾后建筑配对，无法在两时相中同时匹配到的建筑会被跳过。
4. 对每个建筑分别取得灾前框和灾后框，并计算二者并集框。
5. 在并集框外扩 `16` 个像素的上下文边缘，得到最终裁剪区域。
6. 若裁剪框宽或高小于 `16` 像素，则该样本被剔除。
7. 对灾前和灾后影像使用同一个裁剪框裁剪，以保持空间对应关系。
8. 将裁剪结果补边并缩放为统一的 `224 x 224` patch，分别保存为 `_pre.png` 与 `_post.png`。

为保证灾前和灾后影像的显示一致性，脚本在裁剪前会先将 TIFF 影像转换为 `uint8 RGB`：

- 默认采用 `percentile` 拉伸模式
- 默认使用 `2%` 和 `98%` 分位数进行对比度拉伸
- 默认采用 `pair` 作用域，即同一对灾前/灾后影像共享拉伸统计量

这种处理能够缓解原始遥感 TIFF 动态范围不一致的问题，同时尽量保持灾前和灾后图像的亮度对比可比性。

建筑级 patch 输出路径位于 `data/processed/xbd_building_patches/`，目录层次为 `split/disaster_type/tile_id/`。当前版本默认**不包含** `un-classified` 类别，因此保留的损伤标签为 `no-damage`、`minor-damage`、`major-damage` 和 `destroyed`。

从当前构建结果可以看出，建筑级样本规模已经达到 **二十万对以上**，并且仍可能随着脚本继续运行而增加。总体上，`no-damage` 样本占比最高，约为全部建筑 patch 的三分之二到七成，`minor-damage`、`major-damage` 和 `destroyed` 则依次减少，呈现明显的类别不均衡特征。因此在训练脚本中，项目额外提供了按灾害类型进行逆频率采样的机制，以减轻训练偏置。

如果需要在论文中报告最终建筑级样本数，可以在 patch 构建结束后，对 `xbd_building_patches.csv` 执行一次 `value_counts()` 汇总，分别统计 `split`、`disaster_type` 和 `damage_label` 三个维度即可。

## 数据加载与训练时处理

项目提供了两套数据读取接口：

### 1. 瓦片级数据读取

`src/datasets/xbd_dataset.py` 对瓦片级 CSV 进行封装，可按需返回：

- 灾前与灾后整幅影像
- 建筑边界框
- 建筑损伤标签
- 建筑实例掩码 `building_mask`
- 损伤分割掩码 `damage_mask`
- 建筑多边形坐标

其中，建筑掩码和损伤掩码不是预先离线保存的，而是根据 JSON 中的建筑多边形在读取时动态生成。

### 2. 建筑级 patch 读取

`src/datasets/xbd_building_dataset.py` 负责读取建筑级 patch，可返回三种视图：

- `pair`：同时返回灾前 patch 和灾后 patch
- `pre`：只返回灾前 patch
- `post`：只返回灾后 patch

这使得同一套数据既可以用于双分支对比学习，也可以用于单视图特征提取和检索评估。

### 3. 训练阶段图像增强

`src/datasets/transforms.py` 定义了与 CLIP 视觉编码器匹配的图像预处理：

- 训练阶段：`RandomResizedCrop(224, scale=(0.8, 1.0))` 与 `RandomHorizontalFlip(0.5)`
- 评估阶段：`Resize(224)` 与 `CenterCrop(224)`
- 最终统一执行 `ToTensor()` 和 CLIP 均值方差归一化

这意味着模型在训练时会获得适度的数据增强，而评估时保持固定、可复现的中心裁剪策略。

## 小结

综上，本项目的数据处理采用了“**瓦片级统计建模 + 建筑级配对裁剪**”的两级组织方式。瓦片级数据保留了灾害场景的全局统计信息，建筑级 patch 数据则为跨时相建筑检索任务提供了严格对齐的样本对。该设计既保留了灾害场景语义，又强化了建筑实例级别的时相对应关系，适合用于灾后建筑匹配、检索和损伤分析等任务。
