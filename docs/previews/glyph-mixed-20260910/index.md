# 笔画级人工校准预览

打开 [index.html](index.html)，先独立判断修改字，再展开原字及完整笔画。

全部为 REVIEW，不是训练集；明显异常可记 BLOCK，合法可接受记 PASS，不确定或不可读保持 REVIEW。若成为另一个合法字，记录 confirmed_char，不能把来源字直接作为识字标签。

本阶段不导入标签、不训练、不做跨 Noto 字体迁移。

图形源自 Make Me a Hanzi / Arphic 字体，衍生图形按 [Arphic-1999](ARPHICPL.txt) 提供，不提供担保。
