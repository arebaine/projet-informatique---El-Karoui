from pathlib import Path
import fitz
import argparse

PAGE_MAP = {
    1: (13, "top"), 2: (13, "bottom"), 3: (17, "top"), 4: (17, "bottom"),
    5: (18, "full"), 6: (21, "full"), 7: (22, "full"), 8: (24, "full"),
    9: (25, "full"), 10: (26, "full"), 11: (27, "full"), 12: (33, "full"),
    13: (34, "full"), 14: (35, "full"),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=str)
    ap.add_argument("--out", type=str, default="results/reference_figures")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pdf = fitz.open(args.pdf)
    for fig_num, (pno, mode) in PAGE_MAP.items():
        page = pdf[pno]
        if mode == "full":
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2), alpha=False)
        else:
            rect = page.rect
            if mode == "top":
                clip = fitz.Rect(rect.x0, rect.y0 + rect.height*0.18, rect.x1, rect.y0 + rect.height*0.58)
            else:
                clip = fitz.Rect(rect.x0, rect.y0 + rect.height*0.54, rect.x1, rect.y0 + rect.height*0.93)
            pix = page.get_pixmap(matrix=fitz.Matrix(2,2), clip=clip, alpha=False)
        pix.save(str(out / f"figure{fig_num:02d}_reference.png"))

if __name__ == "__main__":
    main()
