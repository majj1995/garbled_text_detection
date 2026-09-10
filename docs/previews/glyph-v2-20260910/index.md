# 字形 V2 人工校准预览

候选 50/50；全部为 REVIEW，不是训练集。

用浏览器打开同目录的 [index.html](index.html)，先判断 96×96 字形，再展开原字和 mask。

[总览](overview.png) · [审阅模板](review-template.jsonl) · [配置及跳过统计](run.json)

明显结构异常可记录 BLOCK；合法可接受记 PASS；不确定或不可读保留 REVIEW。若变成另一个合法字，记录 confirmed_char，不能继续把来源字当识字标签。

此处不执行标签导入或训练。背景干扰和真实业务效果需要后续单独验证。
