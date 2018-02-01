package main

import (
	"encoding/json"
	"flag"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"strings"
	"testing"
)

var (
	travis = flag.Bool("travis", false, "")
)

func TestInit(t *testing.T) {
	if *travis {
		expectedDataDir := "/home/travis/gopath/src/sat/data"
	} else {
		expectedDataDir, err := filepath.Abs("../../data")
	}
	if err != nil {
		t.Fatal(err)
	}
	Init(ioutil.Discard, os.Stdout, os.Stdout, os.Stderr)

	flag.BoolVar(travis, "t", false, "")
	flag.Parse()

	if port == nil {
		t.Errorf("got no port")
	} else if *dataDir != expectedDataDir {
		t.Errorf("got %#v, wanted %#v", *dataDir, expectedDataDir)
	} else {
		print("Passed TestInit!\n")
	}
}

func TestGetPaths(t *testing.T) {
	a := &Task{}
	str := `{"AssignmentID": "0000", "ProjectName": "testPath",
			"StartTime": 1517447292, "SubmitTime": 1517447292}`
	err := json.Unmarshal([]byte(str), a)

	expectedAssign := path.Join(*dataDir, "Assignments", a.ProjectName,
		a.AssignmentID+".json")
	expectedSubmis := path.Join(*dataDir, "Submissions", a.ProjectName,
		a.AssignmentID, formatTime(a.StartTime)+".json")
	expectedLatest := path.Join(*dataDir, "Submissions", a.ProjectName,
		a.AssignmentID, "latest.json")
	expectedLog := path.Join(GetProjPath(), "Log", a.ProjectName, a.AssignmentID,
		formatTime(a.SubmitTime)+".json")

	if err != nil {
		t.Fatal(err)
	}

	if path := a.GetAssignmentPath(); path != expectedAssign {
		t.Errorf("got %#v, wanted %#v", path, expectedAssign)
	} else {
		print("Passed TestGetAssignmentPath!\n")
	}

	if path := a.GetSubmissionPath(); path != expectedSubmis {
		t.Errorf("got %#v, wanted %#v", path, expectedSubmis)
	} else {
		print("Passed TestGetSubmissionPath!\n")
	}

	if path := a.GetLatestSubmissionPath(); path != expectedLatest {
		t.Errorf("got %#v, wanted %#v", path, expectedLatest)
	} else {
		print("Passed TestGetLatestSubmissionPath!\n")
	}

	if path := a.GetLogPath(); path != expectedLog {
		t.Errorf("got %#v, wanted %#v", path, expectedLog)
	} else {
		print("Passed TestGetLogPath!\n")
	}
}

func TestMin(t *testing.T) {
	var a, b int = 1, 2
	min := Min(a, b)
	if min != a {
		t.Errorf("Min was incorrect, got: %d, want: %d.", min, a)
	} else {
		print("Passed TestMin!\n")
	}
}

func TestGetResult(t *testing.T) {
}

func TestGetFullResult(t *testing.T) {
}

func TestIndexHandler(t *testing.T) {
	// create a Request to pass to the handler
	// pass 'nil' as the third parameter because no query parameters
	req, err := http.NewRequest("GET", "/", nil)
	if err != nil {
		t.Fatal(err)
	}

	// create a ResponseRecorder to record response
	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(parse(indexHandler))

	// call ServeHTTP method of the handler, pass in Request and ResponseRecorder
	handler.ServeHTTP(rr, req)

	// check the status code
	if status := rr.Code; status != http.StatusOK {
		t.Errorf("indexHandler returned wrong status code:",
			"got %v want %v", status, http.StatusOK)
	} else {
		print("Passed TestIndexHandler!\n")
	}
}

func TestCreateHandler(t *testing.T) {
	req, err := http.NewRequest("GET", "/create", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(createHandler)

	handler.ServeHTTP(rr, req)
	body, _ := ioutil.ReadAll(rr.Result().Body)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("createHandler returned wrong status code:",
			"got %v want %v", status, http.StatusOK)
	} else if expected := "Labeling tool"; !strings.Contains(string(body), expected) {
		t.Fatal(string(body))
		t.Errorf("createHandler does not contain expected content: %v", expected)
	} else {
		print("Passed TestCreateHandler!\n")
	}
}

func TestDashboardHandler(t *testing.T) {
	req, err := http.NewRequest("GET", "/dashboard", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(dashboardHandler)

	handler.ServeHTTP(rr, req)
	body, _ := ioutil.ReadAll(rr.Result().Body)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("dashboardHandler returned wrong status code:",
			"got %v want %v", status, http.StatusOK)
	} else if expected := "Project Dashboard"; !strings.Contains(string(body), expected) {
		t.Errorf("dashboardHandler does not contain expected content: %v", expected)
	} else {
		print("Passed TestDashboardHandler!\n")
	}
}

func TestBboxLabelingHandler(t *testing.T) {
	req, err := http.NewRequest("GET", "/2d_bbox_labeling", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(bboxLabelingHandler)

	handler.ServeHTTP(rr, req)
	body, _ := ioutil.ReadAll(rr.Result().Body)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("bboxLabelingHandler returned wrong status code:",
			"got %v want %v", status, http.StatusOK)
	} else if expected := "Bounding box labeling tool"; !strings.Contains(string(body), expected) {
		t.Errorf("bboxLabelingHandler does not contain expected content: %v", expected)
	} else {
		print("Passed TestBboxLabelingHandler!\n")
	}
}

func TestRoadLabelingHandler(t *testing.T) {
	req, err := http.NewRequest("GET", "/2d_road_labeling", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(roadLabelingHandler)

	handler.ServeHTTP(rr, req)
	body, _ := ioutil.ReadAll(rr.Result().Body)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("roadLabelingHandler returned wrong status code:",
			"got %v want %v", status, http.StatusOK)
	} else if expected := "Drivable Area Labeling Tool"; !strings.Contains(string(body), expected) {
		t.Errorf("roadLabelingHandler does not contain expected content: %v", expected)
	} else {
		print("Passed TestRoadLabelingHandler!\n")
	}
}

func TestSegLabelingHandler(t *testing.T) {
	req, err := http.NewRequest("GET", "/2d_seg_labeling", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(segLabelingHandler)

	handler.ServeHTTP(rr, req)
	body, _ := ioutil.ReadAll(rr.Result().Body)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("segLabelingHandler returned wrong status code:",
			"got %v want %v", status, http.StatusOK)
	} else if expected := "Segmentation Labeling Tool"; !strings.Contains(string(body), expected) {
		t.Errorf("segLabelingHandler does not contain expected content: %v", expected)
	} else {
		print("Passed TestSegLabelingHandler!\n")
	}
}

func TestLaneLabelingHandler(t *testing.T) {
	req, err := http.NewRequest("GET", "/2d_lane_labeling", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(laneLabelingHandler)

	handler.ServeHTTP(rr, req)
	body, _ := ioutil.ReadAll(rr.Result().Body)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("laneLabelingHandler returned wrong status code:",
			"got %v want %v", status, http.StatusOK)
	} else if expected := "Lane Edge Labeling Tool"; !strings.Contains(string(body), expected) {
		t.Errorf("laneLabelingHandler does not contain expected content: %v", expected)
	} else {
		print("Passed TestLaneLabelingHandler!\n")
	}
}

func TestImageLabelingHandler(t *testing.T) {
	req, err := http.NewRequest("GET", "/image_labeling", nil)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	handler := http.HandlerFunc(imageLabelingHandler)

	handler.ServeHTTP(rr, req)
	body, _ := ioutil.ReadAll(rr.Result().Body)

	if status := rr.Code; status != http.StatusOK {
		t.Errorf("imageLabelingHandler returned wrong status code:",
			"got %v want %v", status, http.StatusOK)
	} else if expected := "Bounding box labeling tool"; !strings.Contains(string(body), expected) {
		t.Errorf("imageLabelingHandler does not contain expected content: %v", expected)
	} else {
		print("Passed TestImageLabelingHandler!\n")
	}
}

func TestPostAssignmentHandler(t *testing.T) {
}

func TestPostSubmissionHandler(t *testing.T) {
}

func TestPostLogHandler(t *testing.T) {
}

func TestRequestAssignmentHandler(t *testing.T) {
}

func TestRequestSubmissionHandler(t *testing.T) {
}

func TestRequestInfoHandler(t *testing.T) {
}

func TestReadResultHandler(t *testing.T) {
}

func TestReadFullResultHandler(t *testing.T) {
}
