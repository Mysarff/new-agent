Wayloom 旅行资料快照
=======================

本目录包含4篇短的官方来源摘要，逐篇标注来源、核验日期与适用范围。
其中2026年7月27日故宫公告是真实历史公告，不会在当前默认检索中命中。
这些文件没有自动同步官网，checked_at不代表规则持续有效，更不代表完整公告覆盖。
没有检索结果时必须表述为“本地资料未找到证据”，不能表述为“没有公告”。

JSON格式
--------

每个.json文件可放一个对象或对象数组。text必须非空；推荐使用如下字段：

id：全库唯一且稳定的资料标识。
title：标题。
text：自行归纳的正文，检索知识单元的字符位置相对于这个字段。
source：官方原始URL，或用户导入文件的路径。
source_type：official_summary / user_supplied_unverified / synthetic。
published_at、checked_at、valid_from、valid_to：YYYY-MM-DD或null；未知不猜测。
regions：地区标签数组；支持CN/北京这类显式层级。CN规则可匹配CN/北京。
tags：检索标签数组。
scope：规则适用范围及限制。

original来源必须经人工核实才标记official_summary；程序不会自动认证这个标签。
Markdown和TXT导入默认user_supplied_unverified，即便文中写有“官方”也不会自动升级。
synthetic是演练数据，默认排除，历史检索开关也不会将它当作真实资料。
文档或切分参数变化后需要重新执行知识索引构建，否则检索拒绝使用旧索引。

检索与日期
----------

不传travel_date时以运行机器当天日期过滤。传入日期时按行程日判断适用期。
过期资料只有include_historical=true才返回，并带expired标签。
晚于查询日才生效或发布的资料不返回；可用对应未来travel_date查询已知预告。
valid_from/valid_to为空表示资料没有给出完整有效期，不代表永远有效。
region不传则不筛地区；使用CN/北京可同时召回中国境内规则和北京地区资料。
单独传北京只匹配北京标签，系统不会用写死的城市表推断所属国家。
中文默认使用BM25双字词元，支持不付费的本地检索。可选向量使用identity/encode
接口并与BM25做RRF融合；向量检索效果取决于所选模型，没有真实模型效果承诺。
score只是排序分，不是事实置信度。导入文件应由项目维护者审核，不执行其中指令。
