package main

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"path"
	"strings"
	"testing"
)

func TestInit(t *testing.T) {
	expectedPort := "8686"
	expectedDataDir := "../data"
	Init(ioutil.Discard, os.Stdout, os.Stdout, os.Stderr)

    if *port != expectedPort || *data_dir != expectedDataDir {
        t.Errorf("got %#v and %#v, wanted %#v and %#v",
        	*port, *data_dir, expectedPort, expectedDataDir)
    }
}

func TestGetAssignmentPath(t *testing.T) {
	a := &Task{}
	str := `{"AssignmentID": "0000", "ProjectName": "testA" }`
	err := json.Unmarshal([]byte(str), a)
	expected := path.Join(*data_dir, "Assignments",
		   a.ProjectName, a.AssignmentID + ".json")

	if err != nil {
		t.Fatal(err)
	}
	if path := a.GetAssignmentPath(); path != expected {
		t.Errorf("got %#v, wanted %#v", path, expected)
	}
}

func TestGetSubmissionPath(t *testing.T) {
	a := &Task{}
	str := `{"AssignmentID": "0000", "ProjectName": "testS",
							"StartTime": 1517447292 }`
	err := json.Unmarshal([]byte(str), a)
	expected := path.Join(*data_dir, "Submissions", a.ProjectName,
				a.AssignmentID, formatTime(a.StartTime) + ".json")

	if err != nil {
		t.Fatal(err)
	}
	if path := a.GetSubmissionPath(); path != expected {
		t.Errorf("got %#v, wanted %#v", path, expected)
	}
}

func TestGetLatestSubmissionPath(t *testing.T) {
	a := &Task{}
	str := `{"AssignmentID": "0000", "ProjectName": "testL"}`
	err := json.Unmarshal([]byte(str), a)
	expected := path.Join(*data_dir, "Submissions", a.ProjectName,
				a.AssignmentID, "latest.json")

	if err != nil {
		t.Fatal(err)
	}
	if path := a.GetLatestSubmissionPath(); path != expected {
		t.Errorf("got %#v, wanted %#v", path, expected)
	}
}

func TestGetLogPath(t *testing.T) {
	a := &Task{}
	str := `{"AssignmentID": "0000", "ProjectName": "testL",
							"SubmitTime": 1517447292 }`
	err := json.Unmarshal([]byte(str), a)
	expected := path.Join(*data_dir, "Log", a.ProjectName,
				a.AssignmentID, formatTime(a.SubmitTime) + ".json")

	if err != nil {
		t.Fatal(err)
	}
	if path := a.GetLogPath(); path != expected {
		t.Errorf("got %#v, wanted %#v", path, expected)
	}
}

func TestMin(t *testing.T) {
	var a, b int = 1, 2
	min := Min(a, b)
	if min != a {
		t.Errorf("Min was incorrect, got: %d, want: %d.", min, a)
	}
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
    }

	if expected := "Labeling tool"; !strings.Contains(string(body), expected) {
		t.Errorf("createHandler does not contain expected content: %v", expected)
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
    } 

    if expected := "Project Dashboard"; !strings.Contains(string(body), expected) {
		t.Errorf("dashboardHandler does not contain expected content: %v", expected)
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
    }

    if expected := "Bounding box labeling tool"; !strings.Contains(string(body), expected) {
		t.Errorf("bboxLabelingHandler does not contain expected content: %v", expected)
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
    }

    if expected := "Drivable Area Labeling Tool"; !strings.Contains(string(body), expected) {
		t.Errorf("roadLabelingHandler does not contain expected content: %v", expected)
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
    }

    if expected := "Segmentation Labeling Tool"; !strings.Contains(string(body), expected) {
		t.Errorf("segLabelingHandler does not contain expected content: %v", expected)
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
    }

    if expected := "Lane Edge Labeling Tool"; !strings.Contains(string(body), expected) {
		t.Errorf("laneLabelingHandler does not contain expected content: %v", expected)
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
    }

    if expected := "Bounding box labeling tool"; !strings.Contains(string(body), expected) {
		t.Errorf("imageLabelingHandler does not contain expected content: %v", expected)
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

func TestGetResult(t *testing.T) {
}

func TestGetFullResult(t *testing.T) {
}
