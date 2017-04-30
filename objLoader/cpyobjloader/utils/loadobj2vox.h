#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include "tiny_obj_loader.h"


typedef std::vector<double> vectd;
typedef std::vector<int> vecti;
typedef std::vector<tinyobj::shape_t>::iterator shape_iter;

// struct for (u,v)
struct UV
{
	double u,v;
	UV(double u, double v):u(u),v(v){}
};

// struct for (x,y,z)
struct Coord
{
    double x,y,z;

    Coord(double x=0, double y=0, double z=0):x(x),y(y),z(z){ }

    Coord& operator=(const Coord& a) {
        x = a.x;
        y = a.y;
        z = a.z;
        return *this;
    }

    Coord operator+(const Coord& a) const {
        return Coord(a.x+x, a.y+y, a.z+z);
    }

    Coord operator-(const Coord& a) const {
        return Coord(-a.x+x, -a.y+y, -a.z+z);
    }

    Coord operator*(double a) const {
        return Coord(a*x, a*y, a*z);
    }
};

// (N,N,N,C)
struct Grid3D
{
	double* grid;
	int* 	grid_count;
	int N,C;
	Grid3D(int N, int C):N(N),C(C) {
		grid = new double[N*N*N*C];
		memset(grid, 0, sizeof(double)*N*N*N*C);
		grid_count = new int[N*N*N];
		memset(grid_count, 0, sizeof(int)*N*N*N);
	}
	~Grid3D() {
		delete grid;
		delete grid_count;
	}
	void addSample(int x, int y, int z, float* rgb) {
		int n = getCount(x,y,z)+1;
		for (int c=0; c<3; ++c) {
			double m_ = get(x,y,z,c);
			double m = (m_*(n-1)+(double)(rgb[c]))/n;
			set(x,y,z,c,m);
		}
		setCount(x,y,z,n);
		set(x,y,z,3,1);
	}

	double get(int i) {
		return grid[i];
	}
	double get(int x, int y, int z, int c) {
		return grid[((x*N+y)*N+z)*C+c];
	}
	void set(int x, int y, int z, int c, double value) {
		grid[((x*N+y)*N+z)*C+c] = value;	
	}
	int getCount(int x, int y, int z) {
		return grid_count[(x*N+y)*N+z];
	}
	void setCount(int x, int y, int z, int value) {
		grid_count[(x*N+y)*N+z] = value;
	}

	void print() {
		int count = 0;
		for (int i=0; i<N*N*N*C; ++i) {
			if (get(i)!=0)
				printf("%f ", get(i));
		}
	}
};


// declear of functions
double min(double d1, double d2);
double max(double d1, double d2);
Coord min(Coord v1, Coord v2);
Coord max(Coord v1, Coord v2);
void incrementGrid(Grid3D& grid, Coord v, float* difuse);
double randomDouble(double min, double max);
double areaOfTriangle(Coord v1, Coord v2, Coord v3);
void sampleTriangleInGrid(Grid3D& grid, Coord v1, Coord v2, Coord v3, float* difuse);
void LoadObj2Vox(char* filename, Grid3D& grid);



double min(double d1, double d2) {
	return (d1<d2) ? d1 : d2;
}

double max(double d1, double d2) {
	return (d1>d2) ? d1 : d2;
}

Coord min(Coord v1, Coord v2) {
	return Coord(min(v1.x,v2.x),min(v1.y,v2.y),min(v1.z,v2.z));
}

Coord max(Coord v1, Coord v2) {
	return Coord(max(v1.x,v2.x),max(v1.y,v2.y),max(v1.z,v2.z));
}

void incrementGrid(Grid3D& grid, Coord v, float* difuse) {
	int x = (int)v.x;
	int y = (int)v.y;
	int z = (int)v.z;
	grid.addSample(x,y,z,difuse);
}

// range: (0,1)
double randomDouble(double min, double max) {
	double res = (double)(rand()%10000)/10000.0;
	return min + (max-min)*res;
}

double areaOfTriangle(Coord v1, Coord v2, Coord v3) {
	Coord AB = v2-v1;
	Coord AC = v3-v1;
	double res = pow(AB.y*AC.z - AB.z*AC.y, 2);
	res += pow(AB.x*AC.z - AB.z*AC.x, 2);
	res += pow(AB.x*AC.y - AB.y*AC.x, 2);
	res = sqrt(res)/2;
	return res;
}

// diffuse : (r,g,b)
void sampleTriangleInGrid(Grid3D& grid, Coord v1, Coord v2, Coord v3, float* diffuse) {
	srand (time(NULL));
	int sampleRate = 1+10*int(areaOfTriangle(v1,v2,v3));
	for (int i=0; i<sampleRate; ++i) {
		double a = randomDouble(0,1);
		double b = randomDouble(0,1-a);
		Coord v = v1 + (v2-v1)*a + (v3-v1)*b;
		incrementGrid(grid, v, diffuse);
	}
}

// N: grid size
void LoadObj2Vox(char* filename, Grid3D& grid) {
	// load obj
	tinyobj::attrib_t attrib;
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> materials;
	std::string err;
	tinyobj::LoadObj(&attrib, &shapes, &materials, &err, filename);

	// get all vertices. Flatten index array: x,y,z => a vertex
	vectd vertices = vectd(attrib.vertices.begin(), attrib.vertices.end());
	// get all texcoords. Flatten index array: u,v => a point
	vectd texcoords = vectd(attrib.texcoords.begin(), attrib.texcoords.end());

	// load all texture images and put into a dict: {"mtl_id": image}
	//printf("mtl size %d\n", materials.size());
	//printf("vertices size %d\n", vertices.size());
	//printf("grid size %d\n", grid.N);
	//for (int i=0; i<materials.size(); ++i) {
	//	float* mtl_diffuse = materials[i].diffuse;
	//	printf("%f %f %f \n", mtl_diffuse[0], mtl_diffuse[1], mtl_diffuse[2]);
	//}

	// get boundary for all vertex
	Coord minCoord(9999999,9999999,9999999);
	Coord maxCoord(-9999999,-9999999,-9999999);
	for (int i=0; i<vertices.size()/3; ++i) {
		Coord cv(vertices[i*3+0],vertices[i*3+1],vertices[i*3+2]);
		minCoord = min(minCoord, cv);
		maxCoord = max(maxCoord, cv);
	}
	Coord origin = minCoord;
	Coord diff = maxCoord - minCoord;
	double scale = max(max(diff.x,diff.y),diff.z);
	scale = (grid.N-1)/scale;		// -1: in case of overflow

	// iterate over all faces
	for (shape_iter shape=shapes.begin(); shape!=shapes.end(); shape++) {
		tinyobj::mesh_t cm = (*shape).mesh;
		// cm.indices: index_t, index_t, index_t => a face
		// cm.material_ids: mtl_id => a face
		// 					
		int num_faces = cm.indices.size()/3;
		for (int i=0; i<num_faces; i++) {
			// get face material 						may be -1 due to path miss
			int mtl_id = cm.material_ids[i];
			tinyobj::material_t mtl = materials[mtl_id];
			std::string mtl_imname = mtl.diffuse_texname.c_str();
			float* mtl_diffuse = mtl.diffuse;

			// get face vertex
			int v1_idx = cm.indices[i*3+0].vertex_index;
			int v2_idx = cm.indices[i*3+1].vertex_index;
			int v3_idx = cm.indices[i*3+2].vertex_index;
			Coord v1(vertices[v1_idx*3+0],vertices[v1_idx*3+1],vertices[v1_idx*3+2]);
			Coord v2(vertices[v2_idx*3+0],vertices[v2_idx*3+1],vertices[v2_idx*3+2]);
			Coord v3(vertices[v3_idx*3+0],vertices[v3_idx*3+1],vertices[v3_idx*3+2]);

			// get face texcoord
			int vt1_idx = cm.indices[i*3+0].texcoord_index;
			int vt2_idx = cm.indices[i*3+1].texcoord_index;
			int vt3_idx = cm.indices[i*3+2].texcoord_index;
			if (vt1_idx>=0 && vt2_idx>=0 && vt3_idx>=0) {
				UV uv1(texcoords[vt1_idx*2+0],texcoords[vt1_idx*2+1]);
				UV uv2(texcoords[vt2_idx*2+0],texcoords[vt2_idx*2+1]);
				UV uv3(texcoords[vt3_idx*2+0],texcoords[vt3_idx*2+1]);
			}

			// map to grid world
			v1 = (v1-origin)*scale;
			v2 = (v2-origin)*scale;
			v3 = (v3-origin)*scale;
			sampleTriangleInGrid(grid, v1, v2, v3, mtl_diffuse);
		}	
	}    
	//printf("finish\n");

}

















