def tle_iter_pairs(path, validate_checksum=False):
    """Yield ``(name, line1, line2)`` for each TLE in a 2LE or 3LE text file.

    Name lines may be Space-Track style ("0 ISS (ZARYA)") or CelesTrak style
    ("ISS (ZARYA)"); the name is None for a bare two-line element set or when
    the catalog numbers of the two lines disagree.
    """
    def _tle_checksum_ok(line):
        # TLE checksum: sum(digits) + count('-') over cols 1..68, mod 10 == col 69
        if len(line) < 69 or not line[68].isdigit():
            return False
        s = 0
        for ch in line[:68]:
            if ch.isdigit():
                s += ord(ch) - 48
            elif ch == '-':
                s += 1
        return (s % 10) == (ord(line[68]) - 48)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        name, l1 = None, None
        for raw in f:
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if line.startswith("\ufeff"):
                line = line.lstrip("\ufeff")

            k = line[0]
            is_line1 = k == "1" and len(line) >= 69 and line[1] == " "
            is_line2 = k == "2" and len(line) >= 69 and line[1] == " "
            if not (is_line1 or is_line2):
                # A name line: "0 NAME" in Space-Track 3LE files, a bare name in
                # CelesTrak files. Names may themselves start with a digit.
                name = line[2:].strip() if line.startswith("0 ") else line.strip()
            elif is_line1:
                # minimal sanity: length and optional checksum
                if len(line) >= 69 and (not validate_checksum or _tle_checksum_ok(line)):
                    l1 = line
                else:
                    l1 = None  # bad line 1; skip until next pair
            elif is_line2 and l1 is not None:
                ok_len = len(line) >= 69
                ok_chk = (not validate_checksum) or _tle_checksum_ok(line)
                # catalog numbers match (cols 3..7 in 1-indexed; [2:7] 0-indexed)
                match = (l1[2:7] == line[2:7])
                if ok_len and ok_chk:
                    yield (name if match else None, l1, line)
                # reset state either way
                name, l1 = None, None
        # if file ends with a lone line-1, we just drop it (or you could warn/log)
