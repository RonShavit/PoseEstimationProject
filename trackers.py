def get_trackers_from_file(path):
    trackers = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            x,y,z,r,g,b = line.split(",")
            trackers.append([float(x), float(y), float(z), int(r), int(g), int(b)])
    return trackers

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python trackers.py <trackers_file>")
        sys.exit(1)
    trackers = get_trackers_from_file(sys.argv[1])
    for t in trackers:
        print(t)