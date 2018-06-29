package export

import (
    "encoding/json"
    "math"
    "reflect"
    "fmt"
    "strconv"
)

type Seg2d struct {
    Vertices        [][]float32         `json:"vertices" yaml:"vertices"`
    Types           string              `json:"types" yaml:"types"`
    Closed          bool                `json:"closed" yaml:"closed"`
}

type Box2d struct {
    X1          float32                 `json:"x1" yaml:"x1"`
    X2          float32                 `json:"x2" yaml:"x2"`
    Y1          float32                 `json:"y1" yaml:"y1"`
    Y2          float32                 `json:"y2" yaml:"y2"`
}

// Download format specifications
type ExportFile struct {
	Name            string                     `json:"name" yaml:"name"`
	Attributes      []map[string]interface{}   `json:"attributes" yaml:"attributes"`
	Items           []Item                     `json:"items" yaml:"items"`
}

type Item struct {
    Timestamp       int64                       `json:"timestamp" yaml:"timestamp"`
    Index           int                         `json:"index" yaml:"index"`
    Labels          []Label                     `json:"labels" yaml:"labels"`
}

type Label struct {
    Id              int                         `json:"id" yaml:"id"`
    Category        string                      `json:"category" yaml:"category"`
    Attributes      map[string]interface{}      `json:"attributes" yaml:"attributes"`
    Box2d           Box2d                       `json:"box2d" yaml:"box2d"`
    Seg2d          []Seg2d                    `json:"seg2d" yaml:"seg2d"`
    Box3d           map[string]interface{}      `json:"box3d" yaml:"box3d"`
}

// structs for saved data
type VertexData struct {
    Id          int                     `json:"id" yaml:"id"`
    X           float32                 `json:"x" yaml:"x"`
    Y           float32                 `json:"y" yaml:"y"`
    Type        string                  `json:"type" yaml:"type"`
}

type EdgeData struct {
    Id              int                 `json:"id" yaml:"id"`
    Src             int                 `json:"src" yaml:"src"`
    Dest            int                 `json:"dest" yaml:"dest"`
    Type            string              `json:"type" yaml:"type"`
    ControlPoints   []VertexData        `json:"control_points" yaml:"control_points"`
}

type PolylineData struct {
    Id              int                 `json:"id" yaml:"id"`
    Vertices        []VertexData        `json:"vertices" yaml:"vertices"`
    Edges           []EdgeData          `json:"edges" yaml:"edges"`
}

type Box2dData struct {
    X           float32                 `json:"x" yaml:"x"`
    Y           float32                 `json:"y" yaml:"y"`
    W           float32                 `json:"w" yaml:"w"`
    H           float32                 `json:"h" yaml:"h"`
}

type Seg2dData struct {
    Closed      bool                     `json:"closed" yaml:"closed"`
    Polys       []PolylineData           `json:"polys" yaml:"polys"`
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

func ParseBox2d(data map[string]interface{}) (Box2d) {
    _box2d := Box2dData{}
    MapToStruct(data, &_box2d)

    box2d := Box2d{}
    box2d.X1 = _box2d.X
    box2d.Y1 = _box2d.Y
    box2d.X2 = _box2d.X + _box2d.W
    box2d.Y2 = _box2d.Y + _box2d.H
    return box2d
}

func ParseSeg2d(data map[string]interface{}) ([]Seg2d) {
    _seg2d := Seg2dData{}
    MapToStruct(data, &_seg2d)

    seg2ds := []Seg2d{}
    for _, _poly := range _seg2d.Polys {
        poly := Seg2d{}
        types := []byte{}
        for i, edge := range _poly.Edges {
            v_xy := []float32{_poly.Vertices[i].X, _poly.Vertices[i].Y}
            poly.Vertices = append(poly.Vertices, v_xy)
            types = append(types, 'L')
            if (edge.Type == "bezier") {
                for _, c := range edge.ControlPoints {
                    c_xy := []float32{c.X, c.Y}
                    poly.Vertices = append(poly.Vertices, c_xy)
                    types = append(types, 'C')
                }
            }
            if (!_seg2d.Closed && i == len(_poly.Edges) - 1) {
                xy := []float32{_poly.Vertices[i+1].X, _poly.Vertices[i+1].Y}
                poly.Vertices = append(poly.Vertices, xy)
                types = append(types, 'L')
            }
        }
        poly.Closed = _seg2d.Closed
        poly.Types = string(types[:])
        seg2ds = append(seg2ds, poly)
    }
    return seg2ds
}

var floatType = reflect.TypeOf(float64(0))
var integerType = reflect.TypeOf(int(0))
var stringType = reflect.TypeOf("")

func getFloatSlice(unk interface{}) ([]float64, error) {
    if (reflect.TypeOf(unk).Kind() != reflect.Slice) {
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

func rotateXAxis3D(vector []float64, angle float64) (error) {
    if len(vector) != 3 {
        return fmt.Errorf("Input array was not 3 dimensional")
    }

    y := vector[1]
    z := vector[2]

    vector[1] = math.Cos(angle) * y - math.Sin(angle) * z
    vector[2] = math.Sin(angle) * y + math.Cos(angle) * z

    return nil
}

func rotateYAxis3D(vector []float64, angle float64) (error) {
    if len(vector) != 3 {
        return fmt.Errorf("Input array was not 3 dimensional")
    }

    x := vector[0]
    z := vector[2]

    vector[0] = math.Cos(angle) * x + math.Sin(angle) * z
    vector[2] = -math.Sin(angle) * x + math.Cos(angle) * z

    return nil
}

func rotateZAxis3D(vector []float64, angle float64) (error) {
    if len(vector) != 3 {
        return fmt.Errorf("Input array was not 3 dimensional")
    }

    x := vector[0]
    y := vector[1]

    vector[0] = math.Cos(angle) * x - math.Sin(angle) * y
    vector[1] = math.Sin(angle) * x + math.Cos(angle) * y

    return nil
}

func ParseBox3d(data map[string]interface{}) (map[string]interface{}) {
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
    var points = [8][]float64{};
    var ind = 0
    for x := float64(-0.5); x <= 0.5; x += 1 {
        for y := float64(-0.5); y <= 0.5; y += 1 {
            for z := float64(-0.5); z <= 0.5; z += 1 {
                points[ind] = []float64{x, y, z};
                ind++;
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
        box3d["p" + strconv.Itoa(i)] = point
    }

    return box3d
}
