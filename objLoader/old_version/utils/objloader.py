from scipy.misc import imread
from os.path import dirname, join

# contents = {mtl_id: mtl}
# mtl = {"Ka":[double, double, double],
#        "Ns":[double],
#        "Ks":[double, double, double],
#        "Kd":[double, double, double],
#        "map_Kd":string
#        "image": numpy.ndarray}

def MTL(filename):
    dir_name = dirname(filename)
    contents = {}
    mtl = None
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'newmtl':
            mtl = contents[values[1]] = {}
        elif mtl is None:
            raise ValueError, "mtl file doesn't start with newmtl stmt"
        elif values[0] == 'map_Kd':
            # load the texture referred to by this declaration
            mtl[values[0]] = values[1]
            mtl['image'] = imread(join(dir_name,mtl['map_Kd']))
        else:
            mtl[values[0]] = map(float, values[1:])
    return contents
 
# mtl = MTL()
# vertices = [(double,double,double)]
# normals = [(double,double,double)]
# texcoords = [(double,double)]
# faces = [ ([vertex_ids],
#            [normal_ids],
#            [tex_ids],
#             mtl_id) 
#          ],                   0 for none, starting from 1

class OBJ:
    def __init__(self, filename, swapyz=False):
        dir_name = dirname(filename)
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
 
        material = None
        for line in open(filename, "r"):
            if line.startswith('#'): continue
            values = line.split()
            if not values: continue
            if values[0] == 'v':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = map(float, values[1:4])
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(map(float, values[1:3]))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'mtllib':
                self.mtl = MTL(join(dir_name,values[1]))
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                self.faces.append((face, norms, texcoords, material))
        print "loaded", filename
    
    # return [x,y,z] 
    def getVertex(self, vertex_id):
        return self.vertices[vertex_id-1]

    # return [u,v] 
    def getTexcoords(self, tex_id):
        return self.texcoords[tex_id-1]

    # return [nx,ny,nz] 
    def getNormal(self, normal_id):
        return self.normals[normal_id-1]    

        # self.gl_list = glGenLists(1)
        # glNewList(self.gl_list, GL_COMPILE)
        # glEnable(GL_TEXTURE_2D)
        # glFrontFace(GL_CCW)
        # for face in self.faces:
        #     vertices, normals, texture_coords, material = face
 
        #     mtl = self.mtl[material]
        #     if 'texture_Kd' in mtl:
        #         # use diffuse texmap
        #         glBindTexture(GL_TEXTURE_2D, mtl['texture_Kd'])
        #     else:
        #         # just use diffuse colour
        #         glColor(*mtl['Kd'])
 
        #     glBegin(GL_POLYGON)
        #     for i in range(len(vertices)):
        #         if normals[i] > 0:
        #             glNormal3fv(self.normals[normals[i] - 1])
        #         if texture_coords[i] > 0:
        #             glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
        #         glVertex3fv(self.vertices[vertices[i] - 1])
        #     glEnd()
        # glDisable(GL_TEXTURE_2D)
        # glEndList()
