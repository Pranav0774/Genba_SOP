"""The local text model sometimes slips Simplified Chinese characters into Japanese text, such as
运 for 運 in 运転資格証. Characters that are not Japanese kanji are replaced with the Japanese form
where it is known, and can be detected where it is not."""
import re

# each pair is "simplified" then "Japanese form"
_PAIRS = (
    "运運 证証 转転 车車 动動 认認 确確 检検 设設 复復 开開 关関 钢鋼 铁鉄 电電 线線 须須 应応 对対 "
    "时時 间間 过過 进進 发発 问問 题題 个個 业業 场場 构構 护護 误誤 备備 记記 员員 师師 质質 实実 "
    "现現 规規 则則 报報 压圧 载載 负負 缩縮 无無 绳縄 紧緊 态態 输輸 监監 视視 测測 试試 验験 变変 "
    "换換 调調 节節 择択 选選 项項 严厳 营営 响響 环環 险険 难難 满満 执執 轮輪 离離 边辺 远遠 层層 "
    "总総 经経 济済 结結 论論 统統 领領 导導 门門"
).split()
_TABLE = str.maketrans({pair[0]: pair[1] for pair in _PAIRS})
_IDEOGRAPH = re.compile(r"[㐀-鿿]")


def to_japanese_kanji(text):
    return text.translate(_TABLE)


def non_japanese_characters(text):
    """Kanji in the text that Japanese does not use (Simplified Chinese forms). A character that
    Windows Japanese text encoding (code page 932) cannot hold is not a Japanese kanji."""
    bad = []
    for ch in _IDEOGRAPH.findall(text):
        try:
            ch.encode("cp932")
        except UnicodeEncodeError:
            bad.append(ch)
    return bad
