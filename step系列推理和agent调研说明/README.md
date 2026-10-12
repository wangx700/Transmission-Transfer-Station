# Step 系列推理和 Agent 调研说明

资料整理日期：2026-10-12（北京时间）。分别讨论 Step 5 Preview 与 Step 3.5/3.7 的推理和 Agent 技术竞争力，并提供模型结构、Step-Code、昇腾适配及配图专题。

## 阅读导航

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
