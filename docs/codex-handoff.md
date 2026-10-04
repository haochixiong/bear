# 任务说明：SGE Grafana 看板 v2 实施

## 0. 先读这些

| 材料 | 位置 | 用途 |
|---|---|---|
| 静态预览页 | `grafana-v2-preview.html`（用户已下载，浏览器直接打开） | **布局和视觉以它为准**。里面的数值、容器名、PVE 节点落点是示意数据，只看结构和表现 |
| 现有看板定义及 PromQL | `D:/Magic_Photo_SaaS/local_node/prometheus_onboarding_platform.py` | 实际指标名、job 名、标签以这里为准 |
| 覆盖与限制报告 | `D:/Magic_Photo_SaaS/local_node/docs/verification/2026-10-04-prometheus-legacy-live-acceptance-report.md` | 数量与限制的依据 |

预览页里的 PromQL 指标名是设计方猜的，**一律以现有 py 文件为准**。

## 1. 环境

| 项目 | 值 |
|---|---|
| 监控服务器 | VM114，172.30.246.202 |
| Grafana | 13.2.1，`https://172.30.246.202:8441/` |
| Prometheus | 2.45.3，数据源 UID `sge-prometheus`（`http://127.0.0.1:9090`，服务端访问） |
| 现有看板 | `sge-systems`（11 面板）、`sge-containers`（14 面板） |

对象（2026-10-04 快照）：

- PVE 宿主：pve01～pve06，6 个抓取目标，**每个目标都返回整个集群，必须按 `id` 去重**。
- PVE 侧虚拟机：21 台（101～110、112～122），`id="qemu/VMID"`，带 `name`、`node`。
- 深度采集（node_exporter）：17 台，即 101、103～108、113～122。**102、109、110、112 未接入**，只有 PVE 指标。
- Docker 采集主机：10 台，容器 50 个，其中 49 个在运行。分布：101=22，103/104/105 各 3，106=4，107=7（6 运行），108=5，113=2，117=1，119=0（采集正常、无容器）。
- 老网管 7 台（101、103～108）不支持逐容器网络指标。

VM 用途与 IP（做名称映射用）：

```
101 旧网管 Web 172.30.242.14      113 运维核心服务 172.30.246.201
103 Elasticsearch-03 172.30.242.13  114 监控平台 172.30.246.202
104 Elasticsearch-02 172.30.242.12  115 GitLab 172.30.246.203
105 Elasticsearch-01 172.30.242.11  116 制品仓库 Nexus 172.30.246.204
106 Database 172.30.242.15          117 DNS 172.30.246.205
107 Collection 172.30.242.17        118 OpenSearch 172.30.246.207
108 Clickhouse / AuxData 172.30.242.16  119 GitLab Runner 172.30.246.208
                                    120 SecOps 172.30.246.209
                                    121 SecOps Collector 172.30.246.210
                                    122 Arkime 172.30.246.211
102/109/110/112：用 PVE 的 name
```

## 2. 已确认的决策

1. 结构：4 个顶层页加 2 个详情页（见第 4 节）。
2. 「已停止」默认显示蓝色，不报红。目前没有「应运行清单」。
3. 虚拟机清单里的 CPU/内存列统一用 PVE 口径（21 台都有值），系统口径放在 Top10 和详情页。
4. v2 总览设为组织首页。**旧看板不保留**：v2 验收通过后，先导出旧看板 JSON 存档，再删除。

## 3. 边界（不能做的事）

- 不修改采集器、Prometheus 告警规则、Alertmanager、网络、防火墙、TLS、业务服务。
- 不修改旧验收报告。另写一份 v2 验收记录，说明 v2 替换了 v1。
- v2 验收通过**之前**，不删除、不覆盖 `sge-systems` 和 `sge-containers`。
- 不安装 Grafana 插件，只用原生面板：Stat、Table、Bar gauge、Time series、State timeline、Text。
- 没有数据的地方不要用 `or vector(0)` 补零。

## 4. 看板清单

全部放在新文件夹「SGE 运维监控 v2」中，由 py 生成器生成并通过 API 导入。

| UID | 名称 | tag | 变量 | 默认时间 / 刷新 |
|---|---|---|---|---|
| `sge-v2-home` | 总览 | `sge-v2-nav` | 无（隐藏的 `node` 供 repeat） | 6h / 1m |
| `sge-v2-pve` | PVE 集群 | `sge-v2-nav` | `node` 单选 + All | 6h / 1m |
| `sge-v2-vms` | 虚拟机 | `sge-v2-nav` | `group` 多选，默认 All | 6h / 1m |
| `sge-v2-docker` | 容器 | `sge-v2-nav` | `dhost` 多选，默认 All，**只取 Docker 采集指标里的主机** | 6h / 1m |
| `sge-v2-vm` | 虚拟机详情 | `sge-v2-detail` | `vmid` **单选、无 All**，21 台全列，显示 `114 · name`，默认 114 | 24h / 30s |
| `sge-v2-container` | 容器详情 | `sge-v2-detail` | `dhost`、`container` 单选，container 依赖 dhost | 24h / 30s |

导航：每个看板顶部放 Dashboard links。tag `sge-v2-nav` 平铺显示；tag `sge-v2-detail` 做成「详情」下拉，勾选 keep time 和 include variables。过渡期加一个「v1」下拉指向旧看板，删除旧看板时一起移除。

钻取用 data link，例如 `/d/sge-v2-vm?var-vmid=${__data.fields.vmid}&${__url_time_range}`。路径如下：

- 总览：状态块 → 清单页；节点落点里的 VM 块 → VM 详情；告警行 → VM 详情或容器详情。
- PVE 落点表的行、虚拟机清单的行 → VM 详情。
- VM 详情：本机容器的行 → 容器详情；「所在节点」→ PVE 页。
- 容器清单：主机汇总行 → 本页过滤 `dhost`；主机列 → VM 详情；容器行 → 容器详情。
- 容器详情：「所在 VM」→ VM 详情。

## 5. 各页面板（位置 x,y，尺寸 w×h，24 列网格）

### 总览 `sge-v2-home`
- 第一排 6 个 Stat（每个 4×4），整块可点：活跃告警（按 severity 计数）｜PVE 节点在线（**待核实**，核实前显示文字「待核实」）｜虚拟机运行 x/21（按 id 去重）｜深度采集在线 x/17，副文本写「4 台未接入」｜Docker 采集正常 x/10｜容器运行 x/50。
- 当前告警表（0,4）14×6：数据源 `ALERTS{alertstate="firing"}`，列为级别、对象、告警名、持续时间、摘要，点行钻取。
- 监控链路异常表（14,4）10×6：合并 `up==0` 的目标、Docker 采集失败、textfile 陈旧三类，**只列异常行**。为空时显示「监控链路正常」。
- 节点落点（0,10）：Stat 按 `node` repeat，每个 4×7。每台 VM 一个色块，文字 `114 监控平台`，颜色取 VM 综合状态。未接入的 VM 加后缀「未接入」。
- 资源最紧张：3 个 Bar gauge（各 8×7），分别是 CPU、内存、文件系统的 Top8。用**即时查询** `sort_desc(topk(8, …))`。
- 状态图例：Text 面板，24×3。

### PVE 集群 `sge-v2-pve`
- 节点卡片：按 `node` repeat，每个 4×5，显示在线、CPU、内存。**待核实**，核实不通过就不上线。
- VM 落点表（0,5）16×13，21 行。列：节点、VM、PVE 状态、vCPU、内存分配、CPU(PVE)、内存(PVE)、深度采集状态、容器 运行/总。
- 节点内存分配 Bar gauge（16,5）8×13：显示 Σ VM 内存分配。节点物理内存核实后，改成分配率。
- 折叠行「去重校验」：3 个 Stat，分别是原始序列数、去重后数量（应为 21）、抓取目标数（6）。
- 折叠行「节点详情」：CPU/内存趋势、存储池表（**待核实**）。

### 虚拟机 `sge-v2-vms`
- 5 个 Stat：PVE 运行、深度采集在线、采集失败、未接入（4）、有告警的 VM。
- 虚拟机清单表 24×16，21 行，默认按状态严重度降序、再按 VMID 升序。可以用一个隐藏列辅助排序。列如下：
  状态｜VM（`VMID 用途`）｜IP｜节点（链接）｜深度采集（在线/采集失败/未接入）｜CPU(PVE) gauge｜内存(PVE) gauge｜最满分区（挂载点 + gauge）｜inode 最高｜网络 入/出（bit/s）｜磁盘写（B/s）｜容器 运行/总。
  未接入的 VM，深度相关列一律显示「未接入」。容器列有三种情况：非 Docker 主机显示「未采集」；119 显示「无容器」；其他显示 `运行/总`。
- 3 个 Top10 Bar gauge（各 8×8）：文件系统、内存、CPU，都用系统口径和即时查询。
- 采集在线历史：State timeline 24×9，17 行，数据是 `up`。

### 容器 `sge-v2-docker`
- Docker 主机汇总表 24×10，10 行。列：主机、采集状态（正常/数据陈旧/采集失败）、最近采集（秒）、容器 运行/总、非运行、unhealthy、曾OOM、逐容器网络（支持/不支持）、内存合计。
- 容器清单表 24×18，50 行，异常的排在前面。列：主机、容器、状态、健康检查、重启（时间窗内增量/累计）、CPU、内存/限制、PIDs、网络 入/出、块 IO、OOM、新鲜度。
- 内存 Top10、CPU Top10：Bar gauge，各 12×8，条目名格式 `容器 @ VMID`。

### 虚拟机详情 `sge-v2-vm`
- 头部 Stat：PVE 状态、所在节点（可点）、深度采集三态、运行时长、vCPU/内存、告警数，以及用途/IP/分组。
- 资源行：CPU 按 mode 堆叠、内存、负载 vs 核数（各 8×8）。
- 存储行：文件系统表（使用率、inode、`predict_linear` 预计写满）12×8；磁盘 IO 12×8。
- 网络行：收/发（排除 `lo|veth.*|docker.*|br-.*`）、错误/丢包（各 12×7）。
- 服务行：systemd 表，标题注明「仅覆盖已配置单元」。没有单元时显示「本机未配置服务监控」。
- 本机容器行：Docker 采集 Stat，四态为非 Docker 主机/正常/陈旧/失败；容器表，点行进容器详情。
- PVE 视角行：CPU、内存趋势。**放在页面最下方、常显不折叠**，因为未接入的 VM 只有这部分数据。
- 未接入的 VM：深度相关面板的无数据文案写「无数据（本机未接入，见上方）」。

### 容器详情 `sge-v2-container`
- 头部 Stat：状态、健康检查、重启、OOM、数据新鲜度、所在 VM（可点）。
- 运行状态 State timeline 24×3。
- CPU、内存（如果有限制，画一条限制线）、网络、块 IO、PIDs、累计重启。
- 网络不支持时，用 Text 面板显示「该主机不支持逐容器网络采集」，不要画空曲线。

## 6. 状态规范（所有页面统一）

按严重度编码，便于用 `max()` 取最坏状态：

| 码 | 文案 | 颜色 | 呈现 |
|---|---|---|---|
| 10 | 正常 / 运行中 | green | **只用绿字或小圆点，不铺底色** |
| 11 | 未配置（health=none） | 灰 | 文字，**绝不显示绿色** |
| 12 | 不支持 | 灰斜体 | 文字，**绝不显示 0** |
| 13 | 未接入 | 灰，虚线框 | 文字，和「采集失败」区分开 |
| 20 | 已停止 | blue | 浅底 |
| 30 | 数据陈旧 | purple | 浅底 |
| 40 | 警告 | orange | 实底 |
| 50 | 采集失败 | dark-purple | 实底 |
| 60 | 故障/严重 | red | 实底 |
| 空 | — | 灰 | 无数据 |
| — | 曾OOM | orange 文字 | 历史标记，不铺底 |

- 11、12、13 只用在覆盖/能力类的列，不参与综合状态聚合。
- 进度条、Bar gauge、时间线里「正常」的部分用中性蓝灰色，只有超阈值才变成橙色或红色。这样避免整页大面积绿色。
- 标题只能写「采集在线」，不能写「主机正常」「服务正常」。
- 阈值读现有告警规则，和规则保持一致。规则里没有的，用默认值：CPU 80/90、内存 85/95、文件系统 80/90、inode 80/90、Docker 新鲜度超过 3 倍采集周期算陈旧。没有对应告警的面板，在描述里写「看板阈值，无对应告警」。
- 单位：百分比用 `percentunit`；容量用 IEC；网络用 `bps`；磁盘用 `Bps`。

## 7. 关键 PromQL 写法（`‹›` 处替换成真实名称）

```promql
# PVE 去重并生成 vmid
label_replace(max by (id) (pve_cpu_usage_ratio{id=~"qemu/.*"}), "vmid","$1","id","qemu/(\\d+)")

# 赋状态码：先用比较过滤，再 *0+码。不要写成 码*(x==0)，那样结果是 0
# 深度采集三态 10/50/13
  ((max by (vmid) (up{job="‹node_job›"}) == 1) * 0 + 10)
or ((max by (vmid) (up{job="‹node_job›"}) == 0) * 0 + 50)
or (label_replace(max by (id) (pve_guest_info{id=~"qemu/.*"}),"vmid","$1","id","qemu/(\\d+)") * 0 + 13)

# VM 综合状态：按严重度从高到低用 or 拼接，外层 max by (vmid)
#   critical 告警 60 > 采集失败 50 > warning 告警 40 > PVE 停止 20 > 运行 10
# 如果告警没有 vmid 标签，先用 instance/hostname 做 label_replace 映射

# 不支持的能力不补 0
rate(‹dk_net_rx›[5m]) and on (hostname) (‹dk_net_cap› == 1)

# Docker 主机采集状态 50/30/10
  ((max by (hostname) (‹dk_ok›) == 0) * 0 + 50)
or (((time() - max by (hostname) (node_textfile_mtime_seconds{file=~"‹docker文件›"})) > ‹3×周期›) * 0 + 30)
or (max by (hostname) (‹dk_ok›) * 0 + 10)

# vmid 变量（显示 "114 · name"）
query_result(max by (disp) (label_join(label_replace(pve_guest_info{id=~"qemu/.*"},"vmid","$1","id","qemu/(\\d+)"),"disp"," · ","vmid","name")))
# 正则：/disp="(?<text>(?<value>\d+) · [^"]*)"/
```

中文用途和 IP 不新增采集。在 py 生成器里维护一份清单字典，生成表格的 value mapping 和详情页头部文字。

## 8. 实施步骤

1. **备份**：导出 `sge-systems`、`sge-containers` 的 JSON（含 version），记录当前组织首页设置，存档。
2. **核实**（结果写入 v2 验收记录）：
   - `count by (__name__) ({__name__=~"pve_.*", id=~"node/.*"})`：节点级指标是否存在
   - `count by (__name__) ({__name__=~"pve_.*", id=~"storage/.*"})`：存储指标是否存在
   - `count(count by (id) (pve_guest_info{id=~"qemu/.*"}))` 应为 21
   - `count(up{job="‹node_job›"})` 应为 17；Docker 采集主机数应为 10；容器数应为 50/49
   - 告警规则是否带 `vmid`、`hostname`、`container` 标签
   - 现有告警阈值
3. **生成**：在 `prometheus_onboarding_platform.py` 中新增 v2 生成函数，复用现有 PromQL 和清单，输出 6 个看板 JSON。
4. **导入**：通过 API 导入到新文件夹，旧看板不动。
5. **对照预览页自检**：逐页对比 `grafana-v2-preview.html` 的布局和状态表现。
6. **验收**：完成第 9 节检查后，交给用户确认。
7. **切换**：用户确认后，把首页设为 `sge-v2-home`；用 API 删除 `sge-systems`、`sge-containers`（JSON 已存档），并移除「v1」链接。
8. **回退方案**：从存档 JSON 重新导入旧看板，把首页设回原值。

## 9. 验收清单

- [ ] 6 个看板都在「SGE 运维监控 v2」文件夹中，UID 和 tag 正确，导航与钻取链接全部可用。
- [ ] 计数正确：VM 21（去重后）、深度采集 17、未接入 4、Docker 主机 10、容器 50/49。
- [ ] 102/109/110/112 显示「未接入」，不显示「采集失败」或「离线」。
- [ ] 119 显示「采集正常 / 无容器」，没有标成故障。
- [ ] 老网管 7 台的容器网络列显示「不支持」，不显示 0。
- [ ] `health=none` 显示「未配置」，不是绿色。
- [ ] OOM 只显示橙字「曾OOM」。
- [ ] 107 的已停止容器显示蓝色「已停止」。
- [ ] 详情页没有 All 选项；清单页没有一次画出几十条曲线。
- [ ] Docker 主机变量只包含 10 台 Docker 主机。
- [ ] 每个面板都受本页变量控制；不受控的面板，标题标了「（全局）」。
- [ ] PVE 节点级面板：核实通过才上线；不通过则显示「待核实」或不上线，并写入验收记录。
- [ ] 旧看板 JSON 已存档；旧验收报告没有被修改；新写了 v2 验收记录。

## 10. 当前不具备、不要假装已有的

- PVE 节点级 CPU、内存、存储：待核实。
- 「应运行清单」：没有，所以停止状态不报红。
- 业务可用性探测：没有，看板中不出现「服务健康」字样。
- 容器内存限制：如果没采集，只显示绝对值。
- 102/109/110/112 的深度采集：未接入，接入属于采集变更，不在本次范围。
