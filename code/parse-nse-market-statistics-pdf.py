#!/usr/bin/env python3
"""
NSE Market Statistics PDF Parser
Uses OCR + bounding box coordinates to reconstruct table rows accurately.
"""

import re
import sys
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from pdf2image import convert_from_path
import pytesseract


# ── helpers ──────────────────────────────────────────────────────────────────

ISIN_RE = re.compile(r'^[A-Z]{2}[A-Z0-9]{9,11}$')
SECTOR_KEYWORDS = {
    'AGRICULTURAL', 'AUTOMOBILES', 'ACCESSORIES', 'BANKING',
    'COMMERCIAL', 'SERVICES', 'CONSTRUCTION', 'ALLIED',
    'ENERGY', 'PETROLEUM', 'INSURANCE', 'INVESTMENT',
    'MANUFACTURING', 'TELECOMMUNICATION', 'REAL', 'ESTATE',
    'EXCHANGE', 'TRADED', 'FUNDS',
}

SECTOR_HEADERS = [
    'AGRICULTURAL',
    'AUTOMOBILES & ACCESSORIES',
    'BANKING',
    'COMMERCIAL AND SERVICES',
    'CONSTRUCTION & ALLIED',
    'ENERGY & PETROLEUM',
    'INSURANCE',
    'INVESTMENT SERVICES',
    'MANUFACTURING & ALLIED',
    'TELECOMMUNICATION',
    'REAL ESTATE INVESTMENT TRUST',
    'EXCHANGE TRADED FUNDS',
]

def clean_number(s: str) -> float | None:
    """Convert OCR number string to float, handling common OCR errors."""
    if not s or s.strip() in ('-', '', 'g', 'S'):
        return None
    s = s.strip().replace(',', '').replace(' ', '')
    # OCR sometimes reads '.' as space or omits it
    try:
        return float(s)
    except ValueError:
        return None


def is_isin(token: str) -> bool:
    """Check if token looks like an ISIN (allow OCR noise on first char)."""
    t = token.strip()
    if len(t) < 10:
        return False
    # ISINs start with 2 letters but OCR sometimes mangles first char
    core = t[2:]
    return bool(re.match(r'^[A-Z0-9]{9,11}$', core))


def normalize_isin(token: str) -> str:
    """Fix common OCR mangling in ISINs (0/O, 1/I, ¥->K, D->0, L->1 etc)."""
    t = token.strip().upper()
    # First two chars should be letters (country code)
    fixed = list(t)
    for i in range(min(2, len(fixed))):
        fixed[i] = fixed[i].replace('0', 'O').replace('1', 'I')
    # Replace common OCR errors in the numeric portion
    for i in range(2, len(fixed)):
        fixed[i] = (fixed[i]
                    .replace('O', '0')
                    .replace('I', '1')
                    .replace('L', '1')
                    .replace('¥', 'K')
                    .replace('D', '0'))
    return ''.join(fixed)


# ── OCR extraction ────────────────────────────────────────────────────────────

def extract_words_with_coords(image):
    """Return list of (x, y, w, h, text) for every word tesseract finds."""
    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config='--psm 6'
    )
    words = []
    for i, text in enumerate(data['text']):
        text = text.strip()
        if not text:
            continue
        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]
        conf = int(data['conf'][i])
        if conf < 10:   # skip very low-confidence noise
            continue
        words.append({'x': x, 'y': y, 'w': w, 'h': h, 'text': text})
    return words


def group_into_lines(words, y_tolerance=8):
    """Cluster words into lines by Y coordinate proximity."""
    if not words:
        return []
    words_sorted = sorted(words, key=lambda w: (w['y'], w['x']))
    lines = []
    current_line = [words_sorted[0]]
    for word in words_sorted[1:]:
        if abs(word['y'] - current_line[-1]['y']) <= y_tolerance:
            current_line.append(word)
        else:
            lines.append(sorted(current_line, key=lambda w: w['x']))
            current_line = [word]
    lines.append(sorted(current_line, key=lambda w: w['x']))
    return lines


def line_text(line):
    return ' '.join(w['text'] for w in line)


# ── sector detection ──────────────────────────────────────────────────────────

def detect_sector(text: str, current_sector: str) -> str | None:
    """Return sector name if this line is a sector header, else None."""
    upper = text.upper().strip()
    for header in SECTOR_HEADERS:
        if header in upper or upper in header:
            return header
    # Partial match: line is ALL CAPS and contains a sector keyword
    words = set(upper.split())
    if words & SECTOR_KEYWORDS and upper == upper.upper() and len(words) <= 6:
        # Try to find the closest matching header
        for header in SECTOR_HEADERS:
            header_words = set(header.split())
            if len(words & header_words) >= 1 and not any(c.isdigit() for c in upper):
                return header
    return None


# ── row parsing ───────────────────────────────────────────────────────────────

def parse_security_row(line_words):
    """
    Given a line of words, extract:
      52wk_high, 52wk_low, name, isin, high, low, vwap, prev_price, volume
    Strategy: anchor on ISIN, then work outward left and right.
    """
    texts = [w['text'] for w in line_words]
    full = ' '.join(texts)

    # Find ISIN position
    isin_idx = None
    isin_val = None
    for i, t in enumerate(texts):
        if is_isin(t):
            isin_idx = i
            isin_val = normalize_isin(t)
            break

    if isin_idx is None:
        return None

    # Everything before ISIN: [52wk_high] [52wk_low] [... name ...]
    before = texts[:isin_idx]
    # Everything after ISIN: [high] [low] [vwap] [prev] [volume]  (+ optional flags like 'g', 'S')
    after = [t for t in texts[isin_idx+1:] if t not in ('g', 'S', '|', 'I', 'l')]

    # Parse numbers from after-ISIN tokens
    nums_after = []
    for t in after:
        n = clean_number(t)
        if n is not None:
            nums_after.append(n)

    high = low = vwap = prev = volume = None
    if len(nums_after) >= 5:
        high, low, vwap, prev, volume = nums_after[0], nums_after[1], nums_after[2], nums_after[3], nums_after[4]
    elif len(nums_after) == 4:
        high, low, vwap, prev = nums_after
    elif len(nums_after) == 3:
        high, low, vwap = nums_after

    # Parse before-ISIN: last two numbers are 52wk_high and 52wk_low
    # but 52wk_low might be missing (shown as number-only in some rows)
    num_tokens = []
    name_tokens = []
    for t in before:
        n = clean_number(t)
        if n is not None and not any(c.isalpha() for c in t.replace('.', '')):
            num_tokens.append((len(name_tokens), n, t))
        else:
            name_tokens.append(t)

    # 52wk high/low are the first two pure numbers before the name
    wk52_high = wk52_low = None
    leading_nums = []
    i = 0
    for t in before:
        if clean_number(t) is not None and not any(c.isalpha() for c in t.replace('.', '')):
            leading_nums.append(clean_number(t))
            i += 1
        else:
            break

    if len(leading_nums) >= 2:
        wk52_high, wk52_low = leading_nums[0], leading_nums[1]
        name_start = i
    elif len(leading_nums) == 1:
        wk52_high = leading_nums[0]
        name_start = i
    else:
        name_start = 0

    name = ' '.join(before[name_start:isin_idx]).strip()
    # Remove OCR noise from name
    name = re.sub(r'\s+', ' ', name).strip()

    return {
        'security_name': name,
        'isin': isin_val,
        '52wk_high': wk52_high,
        '52wk_low': wk52_low,
        'high': high,
        'low': low,
        'vwap': vwap,
        'previous_price': prev,
        'volume': int(volume) if volume is not None else None,
    }


# ── main parsing loop ─────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str, date_str: str | None = None) -> pd.DataFrame:
    print(f"Parsing: {pdf_path}")
    pages = convert_from_path(pdf_path, dpi=300)
    print(f"  {len(pages)} pages found, running OCR...")

    # Infer date from filename if not given
    if date_str is None:
        stem = Path(pdf_path).stem  # e.g. "26-FEB-26"
        try:
            date_str = datetime.strptime(stem, '%d-%b-%y').strftime('%Y-%m-%d')
        except ValueError:
            date_str = stem

    records = []
    current_sector = 'UNKNOWN'

    for page_num, page_img in enumerate(pages, 1):
        print(f"  OCR page {page_num}...")
        words = extract_words_with_coords(page_img)
        lines = group_into_lines(words, y_tolerance=10)

        for line in lines:
            text = line_text(line)

            # Skip header / footer noise
            if not text.strip():
                continue
            upper = text.upper()
            if any(skip in upper for skip in [
                'NAIROBI', 'SECURITIES EXCHANGE', '52WK', 'TRADING STATUS',
                'DISCLAIMER', 'VWAP', 'SCRIP', 'AGREEMENT', 'MATURITY',
                'HIGHEST PRICE', 'ACORN', 'ILAM', 'BATTAN', 'LINZI',
            ]):
                continue

            # Sector header detection
            new_sector = detect_sector(text, current_sector)
            if new_sector:
                current_sector = new_sector
                continue

            # Try to parse as a security row
            row = parse_security_row(line)
            if row:
                row['date'] = date_str
                row['sector'] = current_sector
                records.append(row)

    df = pd.DataFrame(records)
    if not df.empty:
        cols = ['date', 'sector', 'security_name', 'isin',
                '52wk_high', '52wk_low', 'high', 'low', 'vwap',
                'previous_price', 'volume']
        df = df[cols]

    print(f"  Extracted {len(df)} securities across {df['sector'].nunique() if not df.empty else 0} sectors")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Parse NSE Market Statistics PDF')
    parser.add_argument('--input', required=True, help='Path to PDF file')
    parser.add_argument('--output', required=True, help='Path to output CSV file')
    parser.add_argument('--date', help='Date override (YYYY-MM-DD); inferred from filename if omitted')
    args = parser.parse_args()

    df = parse_pdf(args.input, args.date)

    print(df.to_string())

    out_csv = Path(args.output)
    out_parquet = out_csv.with_suffix('.parquet')

    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)

    print(f"\nSaved CSV:     {out_csv}")
    print(f"Saved Parquet: {out_parquet}")


if __name__ == '__main__':
    main()