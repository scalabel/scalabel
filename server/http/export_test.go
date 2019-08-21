package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"path"
	"reflect"
	"testing"
)

func FloatEqual(f1 float64, f2 float64) bool {
	return math.Abs(f1-f2) < 0.001
}

func FloatArrayEqual(vertices1 []float64, vertices2 []float64) bool {
	for i := 0; i < len(vertices1); i++ {
		if !FloatEqual(vertices1[i], vertices2[i]) {
			return false
		}
	}
	return true
}

func FloatArrayOfArrayEqual(vertices1 [][]float64, vertices2 [][]float64) bool {
	for i := 0; i < len(vertices1); i++ {
		if !FloatArrayEqual(vertices1[i], vertices2[i]) {
			return false
		}
	}
	return true
}

func coords(data map[string]interface{}) []float64 {
	v := VertexData{}
	MapToStruct(data, &v)
	return []float64{v.X, v.Y}
}

// data before parsing

var Box2dDataStructs = []map[string]interface{}{
	{"x": 478.7920184573, "y": 454.5838057954, "W": 173.1446443451,
		"H": 24.6506756705},
	{"x": 910.1114888987, "y": 699.7739185242, "W": 22.6105225649,
		"H": 79.6625312940},
	{"x": 857.6822839050, "y": 62.5223033165, "W": 151.5271330284,
		"H": 174.6097113707},
	{"x": 760.7877908042, "y": 201.4584124542, "W": 1.1209293956,
		"H": 142.2690489844},
	{"x": 587.6507607175, "y": 752.4520039331, "W": 132.3780017789,
		"H": 186.3163234249},
	{"x": 801.7192731172, "y": 46.0775615369, "W": 59.0162523807,
		"H": 86.5581569625},
	{"x": 29.8084527376, "y": 571.1304971125, "W": 198.8522393520,
		"H": 110.7235428664},
	{"x": 867.4728288827, "y": 442.8921109871, "W": 157.2019004174,
		"H": 165.4684783084},
	{"x": 567.0276746260, "y": 598.5596172189, "W": 68.1508588984,
		"H": 4.1495718779},
	{"x": 965.0563346591, "y": 636.1565779043, "W": 147.9695805767,
		"H": 20.8757090088},
}

var Vertices = []map[string]interface{}{
	{"id": 100, "x": 352.9180483839, "y": 48.0398192532,
		"type": "vertex"},
	{"id": 101, "x": 438.5795776342, "y": 194.3305046791,
		"type": "vertex"},
	{"id": 102, "x": 151.8652725570, "y": 554.5075538404,
		"type": "vertex"},
	{"id": 103, "x": 765.8384403890, "y": 587.1802841891,
		"type": "vertex"},
	{"id": 104, "x": 214.6648040111, "y": 228.0759932263,
		"type": "vertex"},
	{"id": 105, "x": 318.8972106447, "y": 486.9632853285,
		"type": "vertex"},
	{"id": 106, "x": 422.9682695427, "y": 251.0250404463,
		"type": "vertex"},
	{"id": 107, "x": 525.8007402697, "y": 14.5254923717,
		"type": "vertex"},
	{"id": 108, "x": 975.7559088323, "y": 326.6811136843,
		"type": "vertex"},
	{"id": 109, "x": 9.9677230648, "y": 360.4314506879,
		"type": "vertex"},
}

var ControlPoints = []map[string]interface{}{
	{"id": 200, "x": 976.2323493915, "y": 56.0409970319,
		"type": "control_point"},
	{"id": 201, "x": 221.3166425928, "y": 448.9519881560,
		"type": "control_point"},
	{"id": 202, "x": 49.3708157221, "y": 633.6111675987,
		"type": "control_point"},
	{"id": 203, "x": 43.7650062010, "y": 134.1869273863,
		"type": "control_point"},
	{"id": 204, "x": 317.9242572284, "y": 19.5803224804,
		"type": "control_point"},
	{"id": 205, "x": 21.0162497079, "y": 710.5064390722,
		"type": "control_point"},
	{"id": 206, "x": 735.0692200197, "y": 183.2300379138,
		"type": "control_point"},
	{"id": 207, "x": 110.2520759083, "y": 106.6929068358,
		"type": "control_point"},
	{"id": 208, "x": 157.3535845731, "y": 706.0231976234,
		"type": "control_point"},
	{"id": 209, "x": 124.7660897340, "y": 92.5270952830,
		"type": "control_point"},
}

var PolylineDataStructs = []map[string]interface{}{
	{
		"id": 0,
		"Vertices": []map[string]interface{}{
			Vertices[0], Vertices[1], Vertices[2],
		},
		"Edges": []map[string]interface{}{
			{"id": 300, "src": Vertices[0]["id"], "dest": Vertices[1]["id"],
				"type": "line", "control_points": nil},
			{"id": 301, "src": Vertices[1]["id"], "dest": Vertices[2]["id"],
				"type": "line", "control_points": nil},
			{"id": 302, "src": Vertices[2]["id"], "dest": Vertices[0]["id"],
				"type": "line", "control_points": nil},
		},
	},

	{
		"id": 1,
		"Vertices": []map[string]interface{}{
			Vertices[0], Vertices[1], Vertices[2],
		},
		"Edges": []map[string]interface{}{
			{"id": 300, "src": Vertices[0]["id"], "dest": Vertices[1]["id"],
				"type": "bezier",
				"control_points": []map[string]interface{}{ControlPoints[0],
					ControlPoints[1]}},
			{"id": 301, "src": Vertices[1]["id"], "dest": Vertices[2]["id"],
				"type": "bezier",
				"control_points": []map[string]interface{}{ControlPoints[2],
					ControlPoints[3]}},
			{"id": 302, "src": Vertices[2]["id"], "dest": Vertices[0]["id"],
				"type": "bezier",
				"control_points": []map[string]interface{}{ControlPoints[4],
					ControlPoints[5]}},
		},
	},

	{
		"id": 2,
		"Vertices": []map[string]interface{}{
			Vertices[1], Vertices[4], Vertices[2], Vertices[8], Vertices[5],
			Vertices[7], Vertices[9], Vertices[6], Vertices[3], Vertices[0],
		},
		"Edges": []map[string]interface{}{
			{"id": 300, "src": Vertices[1]["id"], "dest": Vertices[4]["id"],
				"type": "line", "control_points": nil},
			{"id": 301, "src": Vertices[4]["id"], "dest": Vertices[2]["id"],
				"type": "bezier",
				"control_points": []map[string]interface{}{ControlPoints[0],
					ControlPoints[1]}},
			{"id": 302, "src": Vertices[2]["id"], "dest": Vertices[8]["id"],
				"type": "line", "control_points": nil},
			{"id": 303, "src": Vertices[8]["id"], "dest": Vertices[5]["id"],
				"type": "bezier",
				"control_points": []map[string]interface{}{ControlPoints[2],
					ControlPoints[3]}},
			{"id": 304, "src": Vertices[5]["id"], "dest": Vertices[7]["id"],
				"type": "bezier",
				"control_points": []map[string]interface{}{ControlPoints[4],
					ControlPoints[5]}},
			{"id": 305, "src": Vertices[7]["id"], "dest": Vertices[9]["id"],
				"type": "line", "control_points": nil},
			{"id": 306, "src": Vertices[9]["id"], "dest": Vertices[6]["id"],
				"type": "line", "control_points": nil},
			{"id": 307, "src": Vertices[6]["id"], "dest": Vertices[3]["id"],
				"type": "bezier",
				"control_points": []map[string]interface{}{ControlPoints[6],
					ControlPoints[7]}},
			{"id": 308, "src": Vertices[3]["id"], "dest": Vertices[0]["id"],
				"type": "line", "control_points": nil},
			{"id": 309, "src": Vertices[0]["id"], "dest": Vertices[1]["id"],
				"type": "bezier",
				"control_points": []map[string]interface{}{
					ControlPoints[8], ControlPoints[9],
				}},
		},
	},
}

var PolygonPoly2dDataStructs = []map[string]interface{}{
	{"closed": true, "polys": []map[string]interface{}{PolylineDataStructs[0]}},
	{"closed": true, "polys": []map[string]interface{}{PolylineDataStructs[1]}},
	{"closed": true, "polys": []map[string]interface{}{PolylineDataStructs[2]}},
	{"closed": true, "polys": []map[string]interface{}{PolylineDataStructs[0],
		PolylineDataStructs[1]}},
	{"closed": true, "polys": []map[string]interface{}{PolylineDataStructs[0],
		PolylineDataStructs[1], PolylineDataStructs[2]}},
}

var PathPoly2dDataStructs = []map[string]interface{}{
	{"closed": false,
		"polys": []map[string]interface{}{PolylineDataStructs[0]}},
	{"closed": false,
		"polys": []map[string]interface{}{PolylineDataStructs[1]}},
	{"closed": false,
		"polys": []map[string]interface{}{PolylineDataStructs[2]}},
	{"closed": false,
		"polys": []map[string]interface{}{PolylineDataStructs[0],
			PolylineDataStructs[1]}},
	{"closed": false, "polys": []map[string]interface{}{PolylineDataStructs[0],
		PolylineDataStructs[1], PolylineDataStructs[2]}},
}

// data after parsing

var Box2dStructs = [][]float64{
	{478.7920184573, 651.9366628024, 454.5838057954, 479.2344814658},
	{910.1114888987, 932.7220114636, 699.7739185242, 779.4364498182},
	{857.6822839050, 1009.2094169334, 62.5223033165, 237.1320146872},
	{760.7877908042, 761.9087201998, 201.4584124542, 343.7274614386},
	{587.6507607175, 720.0287624964, 752.4520039331, 938.7683273580},
	{801.7192731172, 860.7355254979, 46.0775615369, 132.6357184994},
	{29.8084527376, 228.6606920896, 571.1304971125, 681.8540399789},
	{867.4728288827, 1024.6747293001, 442.8921109871, 608.3605892955},
	{567.0276746260, 635.1785335244, 598.5596172189, 602.7091890968},
	{965.0563346591, 1113.0259152359, 636.1565779043, 657.0322869132},
}

var PolygonStructs = []Poly2d{
	{
		[][]float64{
			coords(Vertices[0]),
			coords(Vertices[1]),
			coords(Vertices[2]),
		},
		"LLL",
		true,
	},

	{
		[][]float64{
			coords(Vertices[0]),
			coords(ControlPoints[0]),
			coords(ControlPoints[1]),
			coords(Vertices[1]),
			coords(ControlPoints[2]),
			coords(ControlPoints[3]),
			coords(Vertices[2]),
			coords(ControlPoints[4]),
			coords(ControlPoints[5]),
		},
		"LCCLCCLCC",
		true,
	},

	{
		[][]float64{
			coords(Vertices[1]),
			coords(Vertices[4]),
			coords(ControlPoints[0]),
			coords(ControlPoints[1]),
			coords(Vertices[2]),
			coords(Vertices[8]),
			coords(ControlPoints[2]),
			coords(ControlPoints[3]),
			coords(Vertices[5]),
			coords(ControlPoints[4]),
			coords(ControlPoints[5]),
			coords(Vertices[7]),
			coords(Vertices[9]),
			coords(Vertices[6]),
			coords(ControlPoints[6]),
			coords(ControlPoints[7]),
			coords(Vertices[3]),
			coords(Vertices[0]),
			coords(ControlPoints[8]),
			coords(ControlPoints[9]),
		},
		"LLCCLLCCLCCLLLCCLLCC",
		true,
	},
}

var PathStructs = []Poly2d{
	{
		[][]float64{
			coords(Vertices[0]),
			coords(Vertices[1]),
			coords(Vertices[2]),
		},
		"LLL",
		false,
	},

	{
		[][]float64{
			coords(Vertices[0]),
			coords(ControlPoints[0]),
			coords(ControlPoints[1]),
			coords(Vertices[1]),
			coords(ControlPoints[2]),
			coords(ControlPoints[3]),
			coords(Vertices[2]),
		},
		"LCCLCCL",
		false,
	},

	{
		[][]float64{
			coords(Vertices[1]),
			coords(Vertices[4]),
			coords(ControlPoints[0]),
			coords(ControlPoints[1]),
			coords(Vertices[2]),
			coords(Vertices[8]),
			coords(ControlPoints[2]),
			coords(ControlPoints[3]),
			coords(Vertices[5]),
			coords(ControlPoints[4]),
			coords(ControlPoints[5]),
			coords(Vertices[7]),
			coords(Vertices[9]),
			coords(Vertices[6]),
			coords(ControlPoints[6]),
			coords(ControlPoints[7]),
			coords(Vertices[3]),
			coords(Vertices[0]),
		},
		"LLCCLLCCLCCLLLCCLL",
		false,
	},
}

var PolygonPoly2dStructs = [][]Poly2d{
	{PolygonStructs[0]},
	{PolygonStructs[1]},
	{PolygonStructs[2]},
	{PolygonStructs[0], PolygonStructs[1]},
	{PolygonStructs[0], PolygonStructs[1], PolygonStructs[2]},
}

var PathPoly2dStructs = [][]Poly2d{
	{PathStructs[0]},
	{PathStructs[1]},
	{PathStructs[2]},
	{PathStructs[0], PathStructs[1]},
	{PathStructs[0], PathStructs[1], PathStructs[2]},
}

func TestBox2d(t *testing.T) {
	for i := 0; i < len(Box2dDataStructs); i++ {
		box2dConverted := ParseBox2d(Box2dDataStructs[i])
		box2d := Box2dStructs[i]
		x1, ok := box2dConverted["x1"].(float64)
		if !ok {
			return
		}
		x2, ok := box2dConverted["x2"].(float64)
		if !ok {
			return
		}
		y1, ok := box2dConverted["y1"].(float64)
		if !ok {
			return
		}
		y2, ok := box2dConverted["y2"].(float64)
		if !ok {
			return
		}
		allEqual := FloatEqual(x1, box2d[0]) &&
			FloatEqual(x2, box2d[1]) &&
			FloatEqual(y1, box2d[2]) &&
			FloatEqual(y2, box2d[3])

		if !allEqual {
			// error!
			t.Error(
				"expected", Box2dStructs[i],
				"got", box2dConverted,
			)
		}
	}
}

func TestPolygonPoly2d(t *testing.T) {
	for i := 0; i < len(PolygonPoly2dDataStructs); i++ {
		poly2dConverted := ParsePoly2d(PolygonPoly2dDataStructs[i])
		for j := 0; j < len(poly2dConverted); j++ {
			if poly2dConverted[j].Types != PolygonPoly2dStructs[i][j].Types {
				// error for types
				t.Error(
					"expected", PolygonPoly2dStructs[i][j].Types,
					"got", poly2dConverted[j].Types,
				)
			}
			if !FloatArrayOfArrayEqual(poly2dConverted[j].Vertices,
				PolygonPoly2dStructs[i][j].Vertices) {
				// error for vertices
				t.Error(
					"expected", PolygonPoly2dStructs[i][j].Vertices,
					"got", poly2dConverted[j].Vertices,
				)
			}
			if poly2dConverted[j].Closed != PolygonPoly2dStructs[i][j].Closed {
				// error for closed
				t.Error(
					"expected", PolygonPoly2dStructs[i][j].Closed,
					"got", poly2dConverted[j].Closed,
				)
			}
		}
	}
}

func TestPathPoly2d(t *testing.T) {
	for i := 0; i < len(PathPoly2dDataStructs); i++ {
		poly2dConverted := ParsePoly2d(PathPoly2dDataStructs[i])
		for j := 0; j < len(poly2dConverted); j++ {
			if poly2dConverted[j].Types != PathPoly2dStructs[i][j].Types {
				// error for types
				t.Error(
					"expected", PathPoly2dStructs[i][j].Types,
					"got", poly2dConverted[j].Types,
				)
			}
			if !FloatArrayOfArrayEqual(poly2dConverted[j].Vertices,
				PathPoly2dStructs[i][j].Vertices) {
				// error for vertices
				t.Error(
					"expected", PathPoly2dStructs[i][j].Vertices,
					"got", poly2dConverted[j].Vertices,
				)
			}
			if poly2dConverted[j].Closed != PathPoly2dStructs[i][j].Closed {
				// error for closed
				t.Error(
					"expected", PathPoly2dStructs[i][j].Closed,
					"got", poly2dConverted[j].Closed,
				)
			}
		}
	}
}

// Helper function which reads the sample sat and v2 export from storage
func readSatandExportV2() (Sat, ItemExportV2, error) {
	sampleSat := Sat{}
	sampleItemExportV2 := ItemExportV2{}

	statePath := path.Join("testdata", "sample_sat.json")
	inputBytes, err := ioutil.ReadFile(statePath)
	if err != nil {
		return sampleSat, sampleItemExportV2, err
	}
	err = json.Unmarshal(inputBytes, &sampleSat)
	if err != nil {
		return sampleSat, sampleItemExportV2, err
	}

	statePath = path.Join("testdata", "sample_item_export_v2.json")
	inputBytes, err = ioutil.ReadFile(statePath)
	if err != nil {
		return sampleSat, sampleItemExportV2, err
	}
	err = json.Unmarshal(inputBytes, &sampleItemExportV2)
	if err != nil {
		return sampleSat, sampleItemExportV2, err
	}
	return sampleSat, sampleItemExportV2, nil

}

func TestExportItemDataBox2dSimple(t *testing.T) {
	var itemExportV2 ItemExportV2

	sampleSat, sampleItemExportV2, err := readSatandExportV2()
	if err != nil {
		t.Fatal(err)
	}
	itemData := sampleSat.Task.Items[0]
	itemData.Labels = map[int]LabelData{1: itemData.Labels[1]}
	itemData.Shapes = map[int]ShapeData{1: itemData.Shapes[1]}
	sampleItemExportV2.Labels = []LabelExportV2{sampleItemExportV2.Labels[0]}

	itemExportV2 = exportItemData(
		itemData,
		sampleSat.Task.Config,
		0,
		"box2d",
		"test")

	diff := reflect.DeepEqual(itemExportV2, sampleItemExportV2)
	if !diff {
		t.Fatal(fmt.Errorf("%+v\n%+v", itemExportV2, sampleItemExportV2))
	}

}

// Tests when multiple labels share a single shape
func TestExportItemDataBox2dSharedShape(t *testing.T) {
	var itemExportV2 ItemExportV2

	sampleSat, sampleItemExportV2, err := readSatandExportV2()
	if err != nil {
		t.Fatal(err)
	}
	itemData := sampleSat.Task.Items[0]
	itemData.Labels = map[int]LabelData{1: itemData.Labels[1],
		10: itemData.Labels[1]}

	shapeWithMultipleLabels := itemData.Shapes[1]
	shapeWithMultipleLabels.Label = append(shapeWithMultipleLabels.Label, 10)
	itemData.Shapes = map[int]ShapeData{1: shapeWithMultipleLabels}
	labelCopy := sampleItemExportV2.Labels[0]
	labelCopy.Id = 10
	sampleItemExportV2.Labels = []LabelExportV2{sampleItemExportV2.Labels[0],
		labelCopy}

	itemExportV2 = exportItemData(
		itemData,
		sampleSat.Task.Config,
		0,
		"box2d",
		"test")

	if err != nil {
		t.Fatal(err)
	}
	diff := reflect.DeepEqual(itemExportV2, sampleItemExportV2)
	if !diff {
		t.Fatal(fmt.Errorf("%+v\n%+v", itemExportV2, sampleItemExportV2))
	}

}

// Test using full sat with multiple labels
func TestExportItemDataBox2dFull(t *testing.T) {
	var itemExportV2 ItemExportV2

	sampleSat, sampleItemExportV2, err := readSatandExportV2()
	if err != nil {
		t.Fatal(err)
	}

	itemExportV2 = exportItemData(
		sampleSat.Task.Items[0],
		sampleSat.Task.Config,
		0,
		"box2d",
		"test")
	if err != nil {
		t.Fatal(err)
	}
	diff := reflect.DeepEqual(itemExportV2, sampleItemExportV2)
	if !diff {
		t.Fatal(fmt.Errorf("%+v\n%+v", itemExportV2, sampleItemExportV2))
	}

}
