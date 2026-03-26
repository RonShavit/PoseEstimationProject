class vertex:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class tri:
    def __init__(self, v1:vertex, v2:vertex, v3:vertex):
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
    
    def  __str__(self):
        return f"tri({self.v1.x}, {self.v1.y}, {self.v1.z}), ({self.v2.x}, {self.v2.y}, {self.v2.z}), ({self.v3.x}, {self.v3.y}, {self.v3.z})"
    
def is_number_str(s:str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False

def read_tri_map(filename:str) -> list[tri]:
    tris = []
    verts = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split(",")
            if is_number_str(parts[0]):
                verts.append(vertex(float(parts[0]), float(parts[1]), float(parts[2])))
            else:
                part0,part1,part2 = int(parts[0][1:]), int(parts[1][1:]), int(parts[2][1:])
                tris.append(tri(verts[part0], verts[part1], verts[part2]))
    return tris


        
