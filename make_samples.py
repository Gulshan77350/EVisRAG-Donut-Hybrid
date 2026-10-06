"""Generate synthetic demo document images in samples/.

These mimic the thesis scenario: a receipt, a report page and an unrelated
filler page. Real photos can be dropped into samples/ as well.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).parent / "samples"
FONTS = Path("C:/Windows/Fonts")


def font(name, size):
    try:
        return ImageFont.truetype(str(FONTS / name), size)
    except OSError:
        return ImageFont.load_default()


def receipt():
    img = Image.new("RGB", (800, 1200), "white")
    d = ImageDraw.Draw(img)
    mono, big = font("consola.ttf", 30), font("arialbd.ttf", 44)
    d.text((400, 60), "WARUNG MAKAN SEDERHANA", font=big, fill="black", anchor="mm")
    d.text((400, 120), "Jl. Sudirman No. 12, Jakarta", font=mono, fill="black", anchor="mm")
    d.text((400, 160), "Tel 021-555-0192", font=mono, fill="black", anchor="mm")
    d.line((40, 200, 760, 200), fill="black", width=2)
    rows = [
        ("1 x Nasi Goreng Special", "45,000"),
        ("2 x Es Teh Manis", "16,000"),
        ("1 x Ayam Bakar", "58,000"),
        ("1 x Sate Ayam (10)", "50,000"),
        ("1 x Kerupuk", "5,000"),
        ("2 x Air Mineral", "20,000"),
    ]
    y = 240
    for name, price in rows:
        d.text((60, y), name, font=mono, fill="black")
        d.text((740, y), price, font=mono, fill="black", anchor="ra")
        y += 55
    d.line((40, y + 10, 760, y + 10), fill="black", width=2)
    y += 40
    for name, price in [("SUBTOTAL", "194,000"), ("DISCOUNT 10%", "-19,400")]:
        d.text((60, y), name, font=mono, fill="black")
        d.text((740, y), price, font=mono, fill="black", anchor="ra")
        y += 55
    d.line((40, y + 10, 760, y + 10), fill="black", width=2)
    y += 40
    d.text((60, y), "TOTAL", font=big, fill="black")
    d.text((740, y), "174,600", font=big, fill="black", anchor="ra")
    y += 90
    d.text((60, y), "CASH", font=mono, fill="black")
    d.text((740, y), "200,000", font=mono, fill="black", anchor="ra")
    d.text((60, y + 55), "CHANGE", font=mono, fill="black")
    d.text((740, y + 55), "25,400", font=mono, fill="black", anchor="ra")
    d.text((400, 1130), "*** THANK YOU ***", font=mono, fill="black", anchor="mm")
    return img


def report():
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    h1, h2, body = font("arialbd.ttf", 48), font("arialbd.ttf", 34), font("arial.ttf", 30)
    d.text((100, 100), "Annual Report 2023", font=h1, fill="black")
    d.text((100, 190), "Section 4: Regional Sales Performance", font=h2, fill="black")
    para = [
        "In fiscal year 2023 the company expanded into two new markets.",
        "Total revenue grew by 12% compared with the previous year,",
        "driven mainly by strong demand in the Asia-Pacific region.",
    ]
    for i, line in enumerate(para):
        d.text((100, 270 + i * 45), line, font=body, fill="black")
    # table
    x0, y0, cw, rh = 100, 480, [420, 300, 300], 70
    header = ["Region", "Revenue (USD M)", "Employees"]
    data = [["North America", "412.5", "2,310"], ["Europe", "298.1", "1,845"],
            ["Asia-Pacific", "536.9", "3,120"], ["Latin America", "87.4", "640"]]
    for r, row in enumerate([header] + data):
        x = x0
        for c, cell in enumerate(row):
            d.rectangle((x, y0 + r * rh, x + cw[c], y0 + (r + 1) * rh), outline="black", width=2,
                        fill="#dde6f0" if r == 0 else "white")
            d.text((x + 15, y0 + r * rh + 18), cell, font=h2 if r == 0 else body, fill="black")
            x += cw[c]
    d.text((100, y0 + 6 * rh), "Table 1: Revenue and headcount by region, FY2023.", font=body, fill="black")
    d.text((100, y0 + 6 * rh + 80), "The Chief Executive Officer of the company is Maria Lindqvist.", font=body, fill="black")
    return img


def filler():
    img = Image.new("RGB", (1240, 1754), "white")
    d = ImageDraw.Draw(img)
    h1, body = font("arialbd.ttf", 48), font("arial.ttf", 30)
    d.text((100, 100), "Gardening Tips for Spring", font=h1, fill="black")
    lines = [
        "Spring is the best season to prepare your soil for planting.",
        "Remove weeds early and add a layer of compost to the beds.",
        "Tomatoes and peppers should be planted after the last frost.",
        "Water seedlings in the morning so the leaves dry before night.",
        "Mulching helps keep moisture in the soil during warm weeks.",
        "Rotate crops every year to reduce pests and soil diseases.",
        "Herbs such as basil and mint grow well in small containers.",
    ]
    for i, line in enumerate(lines):
        d.text((100, 220 + i * 60), line, font=body, fill="black")
    return img


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name, fn in [("receipt", receipt), ("report", report), ("filler", filler)]:
        fn().save(OUT / f"{name}.png")
        print("wrote", OUT / f"{name}.png")
