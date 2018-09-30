package main

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"strconv"
)

type Poly2d struct {
	Vertices [][]float64 `json:"vertices" yaml:"vertices"`
	Types    string      `json:"types" yaml:"types"`
	Closed   bool        `json:"closed" yaml:"closed"`
}

type ItemExport struct {
	Name       string                   `json:"name" yaml:"name"`
	Url        string                   `json:"url" yaml:"url"`
	VideoName  string                   `json:"videoName" yaml:"videoName"`
	Attributes []map[string]interface{} `json:"attributes" yaml:"attributes"`
	Timestamp  int64                    `json:"timestamp" yaml:"timestamp"`
	Index      int                      `json:"index" yaml:"index"`
	Labels     []LabelExport            `json:"labels" yaml:"labels"`
}

type LabelExport struct {
	Id         int                    `json:"id" yaml:"id"`
	Category   string                 `json:"category" yaml:"category"`
	Attributes map[string]interface{} `json:"attributes" yaml:"attributes"`
	Manual     bool                   `json:"manual" yaml:"manual"`
	Box2d      map[string]interface{} `json:"box2d" yaml:"box2d"`
	Poly2d     []Poly2d               `json:"poly2d" yaml:"poly2d"`
	Box3d      map[string]interface{} `json:"box3d" yaml:"box3d"`
}

// structs for saved data
type VertexData struct {
	Id   int     `json:"id" yaml:"id"`
	X    float64 `json:"x" yaml:"x"`
	Y    float64 `json:"y" yaml:"y"`
	Type string  `json:"type" yaml:"type"`
}

type EdgeData struct {
	Id            int          `json:"id" yaml:"id"`
	Src           int          `json:"src" yaml:"src"`
	Dest          int          `json:"dest" yaml:"dest"`
	Type          string       `json:"type" yaml:"type"`
	ControlPoints []VertexData `json:"control_points" yaml:"control_points"`
}

type PolylineData struct {
	Id       int          `json:"id" yaml:"id"`
	Vertices []VertexData `json:"vertices" yaml:"vertices"`
	Edges    []EdgeData   `json:"edges" yaml:"edges"`
}

type Box2dData struct {
	X float64 `json:"x" yaml:"x"`
	Y float64 `json:"y" yaml:"y"`
	W float64 `json:"w" yaml:"w"`
	H float64 `json:"h" yaml:"h"`
}

type Poly2dData struct {
	Closed bool           `json:"closed" yaml:"closed"`
	Polys  []PolylineData `json:"polys" yaml:"polys"`
}

func MapToStruct(m map[string]interface{}, val interface{}) error {
	tmp, err := json.Marshal(m)
	if err != nil {
		return err
	}
	err = json.Unmarshal(tmp, val)
	if err != nil {
		return err
	}
	return nil
}

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

func ParsePoly2d(data map[string]interface{}) []Poly2d {
	_poly2d := Poly2dData{}
	MapToStruct(data, &_poly2d)

	poly2ds := []Poly2d{}
	for _, _poly := range _poly2d.Polys {
		poly := Poly2d{}
		types := []byte{}
		for i, vertex := range _poly.Vertices {
			v_xy := []float64{vertex.X, vertex.Y}
			poly.Vertices = append(poly.Vertices, v_xy)
			types = append(types, 'L')
			if i < len(_poly.Edges) && _poly.Edges[i].Type == "bezier" {
				if (i < len(_poly.Edges)-1) || (_poly2d.Closed) {
					for _, c := range _poly.Edges[i].ControlPoints {
						c_xy := []float64{c.X, c.Y}
						poly.Vertices = append(poly.Vertices, c_xy)
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

var floatType = reflect.TypeOf(float64(0))
var integerType = reflect.TypeOf(int(0))
var stringType = reflect.TypeOf("")

func getFloatSlice(unk interface{}) ([]float64, error) {
	if reflect.TypeOf(unk).Kind() != reflect.Slice {
		return nil, fmt.Errorf("cannot convert interface to slice")
	}

	v := reflect.ValueOf(unk)
	array := make([]float64, v.Len())

	for i := 0; i < v.Len(); i++ {
		val, ok := v.Index(i).Interface().(float64)
		if !ok {
			return nil, fmt.Errorf("cannot convert interface to slice")
		}
		array[i] = val
	}

	return array, nil
}

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
}

func ParseBox3d(data map[string]interface{}) map[string]interface{} {
	var box3d = map[string]interface{}{}
	position, err := getFloatSlice(data["position"])
	rotation, err := getFloatSlice(data["rotation"])
	scale, err := getFloatSlice(data["scale"])
	if err != nil {
		fmt.Println(err)
	}

	fmt.Println(position)
	fmt.Println(scale)

	// Initialize points
	var points = [8][]float64{}
	var ind = 0
	for x := float64(-0.5); x <= 0.5; x += 1 {
		for y := float64(-0.5); y <= 0.5; y += 1 {
			for z := float64(-0.5); z <= 0.5; z += 1 {
				points[ind] = []float64{x, y, z}
				ind++
			}
		}
	}

	// Modify scale, position, rotation and load into box3d
	for i := 0; i < len(points); i++ {
		var point = points[i]
		if scale != nil {
			point[0] *= scale[0]
			point[1] *= scale[1]
			point[2] *= scale[2]
		}
		if rotation != nil {
			rotateXAxis3D(point, rotation[0])
			rotateYAxis3D(point, rotation[1])
			rotateZAxis3D(point, rotation[2])
		}
		if position != nil {
			point[0] += position[0]
			point[1] += position[1]
			point[2] += position[2]
		}
		box3d["p"+strconv.Itoa(i)] = point
	}

	return box3d
}
