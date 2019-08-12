package main

import (
	"encoding/json"
)

// Poly2d datatype for 2d polygon
type Poly2d struct {
	Vertices [][]float64 `json:"vertices" yaml:"vertices"`
	Types    string      `json:"types" yaml:"types"`
	Closed   bool        `json:"closed" yaml:"closed"`
}

// ItemExport datatype
type ItemExport struct {
	Name       string            `json:"name" yaml:"name"`
	Url        string            `json:"url" yaml:"url"`
	VideoName  string            `json:"videoName" yaml:"videoName"`
	Attributes map[string]string `json:"attributes" yaml:"attributes"`
	Timestamp  int64             `json:"timestamp" yaml:"timestamp"`
	Index      int               `json:"index" yaml:"index"`
	Labels     []LabelExport     `json:"labels" yaml:"labels"`
}

// LabelExport datatype
type LabelExport struct {
	Id          int                    `json:"id" yaml:"id"`
	Category    string                 `json:"category" yaml:"category"`
	Attributes  map[string]interface{} `json:"attributes" yaml:"attributes"`
	ManualShape bool                   `json:"manualShape" yaml:"manualShape"`
	Box2d       map[string]interface{} `json:"box2d" yaml:"box2d"`
	Poly2d      []Poly2d               `json:"poly2d" yaml:"poly2d"`
	Box3d       map[string]interface{} `json:"box3d" yaml:"box3d"`
}

// structs for saved data

// VertexData for single vertex
type VertexData struct {
	Id   int     `json:"id" yaml:"id"`
	X    float64 `json:"x" yaml:"x"`
	Y    float64 `json:"y" yaml:"y"`
	Type string  `json:"type" yaml:"type"`
}

// EdgeData for single edge
type EdgeData struct {
	Id            int          `json:"id" yaml:"id"`
	Src           int          `json:"src" yaml:"src"`
	Dest          int          `json:"dest" yaml:"dest"`
	Type          string       `json:"type" yaml:"type"`
	ControlPoints []VertexData `json:"control_points" yaml:"control_points"`
}

// PolylineData for multiple edges and vertices
type PolylineData struct {
	Id       int          `json:"id" yaml:"id"`
	Vertices []VertexData `json:"vertices" yaml:"vertices"`
	Edges    []EdgeData   `json:"edges" yaml:"edges"`
}

// Box2dData for single box
type Box2dData struct {
	X float64 `json:"x" yaml:"x"`
	Y float64 `json:"y" yaml:"y"`
	W float64 `json:"w" yaml:"w"`
	H float64 `json:"h" yaml:"h"`
}

// Poly2dData for multiple polylines
type Poly2dData struct {
	Closed bool           `json:"closed" yaml:"closed"`
	Polys  []PolylineData `json:"polys" yaml:"polys"`
}

// MapToStruct checks if map can be converted into struct
func MapToStruct(m map[string]interface{}, val interface{}) {
	tmp, err := json.Marshal(m)
	if err != nil {
		Error.Println(err)
	}
	err = json.Unmarshal(tmp, val)
	if err != nil {
		Error.Println(err)
	}
}

// ParseBox2d parses map into a box2dData
func ParseBox2d(data map[string]interface{}) map[string]interface{} {
	_box2d := Box2dData{}
	MapToStruct(data, &_box2d)

	box2d := map[string]interface{}{}
	box2d["x1"] = _box2d.X
	box2d["y1"] = _box2d.Y
	box2d["x2"] = _box2d.X + _box2d.W
	box2d["y2"] = _box2d.Y + _box2d.H
	return box2d
}

// ParsePoly2d parses map into a Poly2dData
func ParsePoly2d(data map[string]interface{}) []Poly2d {
	_poly2d := Poly2dData{}
	MapToStruct(data, &_poly2d)

	poly2ds := []Poly2d{}
	for _, _poly := range _poly2d.Polys {
		poly := Poly2d{}
		types := []byte{}
		for i, vertex := range _poly.Vertices {
			vXY := []float64{vertex.X, vertex.Y}
			poly.Vertices = append(poly.Vertices, vXY)
			types = append(types, 'L')
			if i < len(_poly.Edges) && _poly.Edges[i].Type == "bezier" {
				if (i < len(_poly.Edges)-1) || (_poly2d.Closed) {
					for _, c := range _poly.Edges[i].ControlPoints {
						cXY := []float64{c.X, c.Y}
						poly.Vertices = append(poly.Vertices, cXY)
						types = append(types, 'C')
					}
				}
			}
		}
		poly.Closed = _poly2d.Closed
		poly.Types = string(types[:])
		poly2ds = append(poly2ds, poly)
	}
	return poly2ds
}

/* The following code is unused
func rotateXAxis3D(vector []float64, angle float64) error {
	if len(vector) != 3 {
		return fmt.Errorf("Input array was not 3 dimensional")
	}

	y := vector[1]
	z := vector[2]

	vector[1] = math.Cos(angle)*y - math.Sin(angle)*z
	vector[2] = math.Sin(angle)*y + math.Cos(angle)*z

	return nil
}

func rotateYAxis3D(vector []float64, angle float64) error {
	if len(vector) != 3 {
		return fmt.Errorf("Input array was not 3 dimensional")
	}

	x := vector[0]
	z := vector[2]

	vector[0] = math.Cos(angle)*x + math.Sin(angle)*z
	vector[2] = -math.Sin(angle)*x + math.Cos(angle)*z

	return nil
}

func rotateZAxis3D(vector []float64, angle float64) error {
	if len(vector) != 3 {
		return fmt.Errorf("Input array was not 3 dimensional")
	}

	x := vector[0]
	y := vector[1]

	vector[0] = math.Cos(angle)*x - math.Sin(angle)*y
	vector[1] = math.Sin(angle)*x + math.Cos(angle)*y

	return nil
} */

// ParseBox3d parses a map into a box3d
func ParseBox3d(data map[string]interface{}) map[string]interface{} {
	var box3d = map[string]interface{}{}

	box3d["location"] = data["position"]
	box3d["orientation"] = data["rotation"]
	box3d["dimension"] = data["scale"]

	return box3d
}
