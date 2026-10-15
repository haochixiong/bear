"""命令行：给 Codex / Claude Code / 其他 Agent 调用，也方便批量检查。

示例：
  colordoctor analyze photo.jpg --mode skin --rect 820,400,300,260 --preset japanese --json
  colordoctor analyze photo.jpg --mode palette --style teal_orange --family 绿
  colordoctor analyze photo.jpg --mode palette --tone-style japanese --style japanese_fresh
  colordoctor analyze photo.jpg --mode reference --reference ref.jpg
  colordoctor analyze photo.jpg --mode point --point 1200,800 --size 5
  colordoctor presets      列出肤色风格
  colordoctor styles       列出整图颜色模板
  colordoctor tones        列出影调风格（预制的目标影调）和九种影调

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
from colordoctor.core import tone_class as tcl
from colordoctor.core.color_management import ImageLoadError, load_image
from colordoctor.core.color_spaces import HUE_SECTORS
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
    if report.mode == "palette" and report.palette is not None:
        p = report.palette
        head = [f"影调 {p.tone_class.name}" + (f" → {p.tone_target}" if p.tone_target and p.tone_target != p.tone_class.name else "")
                if p.tone_class is not None else ""]
        if p.style is not None:
            head.append(f"颜色「{p.style.name}」匹配度 {p.style.score:.0f}/100")
        head += [f"6:3:1 {p.ratio.score:.0f}", f"丰富度 {p.colorfulness.value:.0f}（{p.colorfulness.label}）"]
        lines = [f"[整图色彩] {'；'.join(h for h in head if h)}，置信度 {report.confidence:.2f}"]
    else:
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
    from colordoctor import __version__

    p.add_argument("--version", action="version", version=f"ColorDoctor {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="分析一张 JPG 或相机 RAW")
    a.add_argument("image")
    a.add_argument("--mode", choices=MODES, default="general")
    a.add_argument("--rect", type=lambda t: _ints(t, 4), help="x,y,w,h（像素）；默认整张图")
    a.add_argument("--point", type=lambda t: _ints(t, 2), help="x,y（point 模式）")
    a.add_argument("--size", type=int, default=5, help="取样大小 N（N×N 平均，奇数），默认 5")
    a.add_argument("--preset", choices=[pr.key for pr in sp.load_presets()], metavar="PRESET",
                   help="肤色风格 key，例如 japanese / french / deep；见 colordoctor presets")
    a.add_argument("--part", help="肤色部位名称，例如 脸部 / 身体")
    a.add_argument("--no-auto-mask", action="store_true", help="不自动抠皮肤")
    a.add_argument("--style", choices=[s["key"] for s in pl.load_styles()], metavar="STYLE",
                   help="整图颜色模板 key（只管颜色，palette 模式），例如 teal_orange / black_gold / pink_blue；见 colordoctor styles")
    a.add_argument("--family", choices=[name for name, _, _ in HUE_SECTORS], metavar="FAMILY",
                   help="检查哪种颜色是否统一（palette 模式）：" + " / ".join(name for name, _, _ in HUE_SECTORS))
    a.add_argument("--tone-style", default="", choices=("", *tcl.TONE_STYLES),
                   help="影调风格（只管明暗和反差，和颜色模板互不绑定），例如 japanese / moody / film；见 colordoctor tones")
    a.add_argument("--tone-target", default="", choices=("", *tcl.TONE_NAMES),
                   help="直接指定目标影调（九种之一，例如 高中调），优先于 --tone-style")
    a.add_argument("--reference", help="仿色参考图（reference 模式）")
    a.add_argument("--json", action="store_true", help="输出完整 JSON")
    a.add_argument("--raw-preview", action="store_true", help="RAW 文件改用内嵌的 JPG 预览（相机直出的颜色）")
    a.add_argument("--summary", action="store_true", help="输出精简 JSON 摘要（适合给 AI）")
    sub.add_parser("presets", help="列出肤色风格")
    sub.add_parser("styles", help="列出整图颜色模板")
    sub.add_parser("tones", help="列出影调风格和九种影调")
    sv = sub.add_parser("serve", help="启动给 Photoshop 插件用的本机服务（只监听 127.0.0.1）")
    sv.add_argument("--port", type=int, default=47321)
    return p


def run(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "presets":
        for pr in sp.load_presets():
            print(f"{pr.key:10s} {pr.name}  Lab {tuple(round(v, 1) for v in pr.target_lab)}  {pr.description}")
        return 0
    if args.cmd == "serve":
        from colordoctor.bridge.server import serve

        serve(args.port)
        return 0
    if args.cmd == "tones":
        for k, s in tcl.TONE_STYLES.items():
            also = f"（也可以是{'、'.join(s['also'])}）" if s["also"] else ""
            print(f"{k:10s} {s['name']}：目标 {s['target']}{also}。{s['look']}；常用于{s['scene']}")
        print()
        for n in tcl.TONE_NAMES:
            print(f"{n}：{tcl.TONE_DESC[n]}")
        return 0
    if args.cmd == "styles":
        for s in pl.load_styles():
            print(f"{s['key']:22s} [{s.get('category', '')}] {s['name']}  {s['description']}")
        return 0
    palette_only = [f"--{n.replace('_', '-')}" for n in ("style", "family", "tone_style", "tone_target") if getattr(args, n)]
    if palette_only and args.mode != "palette":
        print(f"错误：{'、'.join(palette_only)} 只用于整图色彩，请加 --mode palette", file=sys.stderr)
        return 2
    try:
        src = load_image(args.image, use_raw_preview=args.raw_preview)
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
            report = an.analyze_palette_report(region, ctx, style_key=args.style, family=args.family, rect=rect,
                                               tone_style=args.tone_style, tone_target=args.tone_target)
        elif args.mode == "reference":
            if not args.reference:
                raise SampleError("reference 模式需要 --reference 参考图.jpg")
            ref = load_image(args.reference).pixels
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
