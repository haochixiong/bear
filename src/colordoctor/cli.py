"""命令行：给 Codex / Claude Code / 其他 Agent 调用，也方便批量检查。

示例：
  colordoctor analyze photo.jpg --mode skin --rect 820,400,300,260 --preset japanese --json
  colordoctor analyze photo.jpg --mode palette --style teal_orange --family 绿
  colordoctor analyze photo.jpg --mode reference --reference ref.jpg
  colordoctor analyze photo.jpg --mode point --point 1200,800 --size 5
  colordoctor presets      列出肤色风格
  colordoctor styles       列出整图风格模板

只读取图片，不修改原图；不联网。
"""

from __future__ import annotations

import argparse
import json
import sys

from colordoctor.core import analysis as an
from colordoctor.core import palette as pl
from colordoctor.core import report_json as rj
from colordoctor.core import skin_presets as sp
from colordoctor.core.color_management import ImageLoadError, load_jpeg
from colordoctor.core.sampling import SampleError, crop

MODES = ("general", "scan", "skin", "palette", "reference", "point")


def _ints(text: str, n: int) -> tuple[int, ...]:
    vals = tuple(int(float(v)) for v in text.replace("，", ",").split(","))
    if len(vals) != n:
        raise argparse.ArgumentTypeError(f"需要 {n} 个用逗号分隔的数字")
    return vals


def _text(report) -> str:
    from colordoctor.core.color_models import PointSample

    if isinstance(report, PointSample):
        v = report.values
        return (
            f"{v.hex}  RGB {v.rgb}  HSB {tuple(round(x) for x in v.hsb)}  "
            f"Lab(D50) {tuple(round(x, 1) for x in v.lab)}  LCh {tuple(round(x, 1) for x in v.lch)}"
        )
    lines = [f"[{report.mode}] 脏色风险 {report.overall_risk:.0f}/100（{report.overall_severity.value}），置信度 {report.confidence:.2f}"]
    for f in report.findings:
        lines.append(f"- [{f.severity.value}] {f.problem}")
        for e in f.evidence[:5]:
            lines.append(f"    证据：{e}")
        for a in f.adjustments[:6]:
            lines.append(f"    建议（{a.tool}）：{a.instruction}")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="colordoctor", description="人像色彩诊断（离线，只读）")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="分析一张 JPG")
    a.add_argument("image")
    a.add_argument("--mode", choices=MODES, default="general")
    a.add_argument("--rect", type=lambda t: _ints(t, 4), help="x,y,w,h（像素）；默认整张图")
    a.add_argument("--point", type=lambda t: _ints(t, 2), help="x,y（point 模式）")
    a.add_argument("--size", type=int, default=5, help="取样大小 N（N×N 平均，奇数），默认 5")
    a.add_argument("--preset", help="肤色风格 key，例如 japanese / french / deep")
    a.add_argument("--part", help="肤色部位名称，例如 脸部 / 身体")
    a.add_argument("--no-auto-mask", action="store_true", help="不自动抠皮肤")
    a.add_argument("--style", help="整图风格模板 key，例如 teal_orange / black_gold / pink_blue")
    a.add_argument("--family", help="检查哪种颜色是否统一：红/橙/黄/绿/青/蓝/紫/品红")
    a.add_argument("--reference", help="仿色参考图（reference 模式）")
    a.add_argument("--json", action="store_true", help="输出完整 JSON")
    a.add_argument("--summary", action="store_true", help="输出精简 JSON 摘要（适合给 AI）")
    sub.add_parser("presets", help="列出肤色风格")
    sub.add_parser("styles", help="列出整图风格模板")
    return p


def run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "presets":
        for pr in sp.load_presets():
            print(f"{pr.key:10s} {pr.name}  Lab {tuple(round(v, 1) for v in pr.target_lab)}  {pr.description}")
        return 0
    if args.cmd == "styles":
        for s in pl.load_styles():
            print(f"{s['key']:16s} {s['name']}  {s['description']}")
        return 0
    try:
        src = load_jpeg(args.image)
        img = src.pixels
        ctx = src.context
        rect = args.rect
        region = crop(img, rect) if rect else img
        if args.mode == "point":
            if not args.point:
                raise SampleError("point 模式需要 --point x,y")
            report = an.analyze_point(img, args.point[0], args.point[1], max(0, args.size // 2), ctx)
        elif args.mode == "skin":
            report = an.analyze_skin_region(region, ctx, preset_key=args.preset, rect=rect, part=args.part,
                                            auto_mask=not args.no_auto_mask)
        elif args.mode == "scan":
            report = an.scan_image(region, ctx, rect=rect)
        elif args.mode == "palette":
            report = an.analyze_palette_report(region, ctx, style_key=args.style, family=args.family, rect=rect)
        elif args.mode == "reference":
            if not args.reference:
                raise SampleError("reference 模式需要 --reference 参考图.jpg")
            ref = load_jpeg(args.reference).pixels
            report = an.compare_reference(region, ref, ctx, rect=rect)
        else:
            report = an.analyze_region(region, ctx, rect=rect)
    except (ImageLoadError, SampleError, KeyError, ValueError) as e:
        print(json.dumps({"error": str(e)}, ensure_ascii=False) if (args.json or args.summary) else f"错误：{e}",
              file=sys.stderr)
        return 2
    if args.json:
        print(rj.report_to_json(report))
    elif args.summary:
        print(json.dumps(rj.summary(report), ensure_ascii=False, indent=2))
    else:
        print(_text(report))
    return 0


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
