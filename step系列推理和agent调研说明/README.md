# Step 系列推理和 Agent 调研说明

**优先阅读：[技术精简版](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/%E6%8A%80%E6%9C%AF%E7%B2%BE%E7%AE%80%E7%89%88.md)。** 集中保留结构、机制、接口与代码；详细专题按需展开。


资料时点：2026-10-12（北京时间）。覆盖 Step-3、Step 3.5/3.7、Step5、Step-Code 与 Audio3；按推理机制、Agent运行控制和昇腾验证组织。

## 当前推荐入口：核心技术报告

- [核心目录与本次刷新说明](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/README.md)

- [05-AFD与StepMesh机制优势及昇腾工程判断](核心技术报告/05-AFD与StepMesh机制优势及昇腾工程判断.md)：资源分离与batch汇聚原理、DBO对比、源码与技术文档、昇腾适配和工程投入。

- [00-核心总览](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/00-%E6%A0%B8%E5%BF%83%E6%80%BB%E8%A7%88.md)
- [01-推理技术核心分析](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/01-%E6%8E%A8%E7%90%86%E6%8A%80%E6%9C%AF%E6%A0%B8%E5%BF%83%E5%88%86%E6%9E%90.md)
- [02-Agent技术核心分析](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/02-Agent%E6%8A%80%E6%9C%AF%E6%A0%B8%E5%BF%83%E5%88%86%E6%9E%90.md)
- [03-昇腾适配与验证规范](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/03-%E6%98%87%E8%85%BE%E9%80%82%E9%85%8D%E4%B8%8E%E9%AA%8C%E8%AF%81%E8%A7%84%E8%8C%83.md)
- [04-重点对象代码参考与工程投入](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E6%A0%B8%E5%BF%83%E6%8A%80%E6%9C%AF%E6%8A%A5%E5%91%8A/04-%E9%87%8D%E7%82%B9%E5%AF%B9%E8%B1%A1%E4%BB%A3%E7%A0%81%E5%8F%82%E8%80%83%E4%B8%8E%E5%B7%A5%E7%A8%8B%E6%8A%95%E5%85%A5.md)

核心版统一保留机制、必要原理、证据与适配边界。下面的详细文档用于查阅参数、案例和源码证据，历史图/PPT保持归档属性。

## 详细参考文档

- [00-技术竞争力总览](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/00-%E6%8A%80%E6%9C%AF%E7%AB%9E%E4%BA%89%E5%8A%9B%E6%80%BB%E8%A7%88.md)
- [01-Step5Preview推理与Agent技术竞争力](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/01-Step5Preview%E6%8E%A8%E7%90%86%E4%B8%8EAgent%E6%8A%80%E6%9C%AF%E7%AB%9E%E4%BA%89%E5%8A%9B.md)
- [02-Step3.5与3.7推理与Agent技术竞争力](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/02-Step3.5%E4%B8%8E3.7%E6%8E%A8%E7%90%86%E4%B8%8EAgent%E6%8A%80%E6%9C%AF%E7%AB%9E%E4%BA%89%E5%8A%9B.md)
- [03-模型结构与技术特性详解](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/03-%E6%A8%A1%E5%9E%8B%E7%BB%93%E6%9E%84%E4%B8%8E%E6%8A%80%E6%9C%AF%E7%89%B9%E6%80%A7%E8%AF%A6%E8%A7%A3.md)
- [04-StepCode机制优势与MindSpeedAgent借鉴](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/04-StepCode%E6%9C%BA%E5%88%B6%E4%BC%98%E5%8A%BF%E4%B8%8EMindSpeedAgent%E5%80%9F%E9%89%B4.md)
- [05-昇腾适配状态依赖与验证边界](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/05-%E6%98%87%E8%85%BE%E9%80%82%E9%85%8D%E7%8A%B6%E6%80%81%E4%BE%9D%E8%B5%96%E4%B8%8E%E9%AA%8C%E8%AF%81%E8%BE%B9%E7%95%8C.md)
- [06-图片与PPT配套说明](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/06-%E5%9B%BE%E7%89%87%E4%B8%8EPPT%E9%85%8D%E5%A5%97%E8%AF%B4%E6%98%8E.md)

## 配套资料

- [图片与 PPT 使用说明](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/06-%E5%9B%BE%E7%89%87%E4%B8%8EPPT%E9%85%8D%E5%A5%97%E8%AF%B4%E6%98%8E.md)：图与正文的对应关系、讲解口径和适用边界。
- [附件目录](https://github.com/wangx700/Transmission-Transfer-Station/tree/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E9%85%8D%E5%A5%97%E5%9B%BE%E4%B8%8EPPT)：Step 5 单页 PPT、Step 3.5/3.7 完整历史图集、昇腾流程图及源文件。
- [附件清单](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/%E9%99%84%E4%BB%B6%E6%B8%85%E5%8D%95.json)：文件大小与 SHA-256。

## 证据与版本说明

正文参考均为网络链接，表格注明日期、单位与测试条件。官方规格、厂商测试、独立评测、媒体转述和工程解释分别标识。昇腾部分为公开资料及源码核验，未做 NPU 真机测试。动态评分、服务端点与版本支持以注明时点为准。

历史图/PPT原样保留，图中教学参数不视为生产配置；现行说明以本目录根部专题文档为准。prefill、Indexer 与 K/V 复用作为通用原理说明，不等于已还原 Step 5 未公开内核。

## Step-3 与 StepAudio 3 新增专题

- [推理与Agent技术专题：五份技术文档与流程图](https://github.com/wangx700/Transmission-Transfer-Station/blob/main/step%E7%B3%BB%E5%88%97%E6%8E%A8%E7%90%86%E5%92%8Cagent%E8%B0%83%E7%A0%94%E8%AF%B4%E6%98%8E/Step3%E4%B8%8EStepAudio3%E6%8A%80%E6%9C%AF%E4%B8%93%E9%A2%98/README.md)


## Audio3 Gen / Music机制与昇腾专项补充

- [05-Audio3 Gen与Music：机制优势、参考实现与昇腾工程投入](核心技术报告/05-Audio3Gen与Music机制优势及昇腾工程投入.md)：详细解释时间–码本深度分工与计划–AR–渲染分工，附接口流程图、代码证据、模块适配清单及分阶段投入建议。
