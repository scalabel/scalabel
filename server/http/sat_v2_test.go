package main

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"path"
	"reflect"
	"testing"
)

// Helper functions
// Save using save handler
func saveSatRequest(t *testing.T, saveData []byte) {
	buf := bytes.NewBuffer(saveData)
	req, err := http.NewRequest("POST", "postSaveV2", buf)
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	postSaveV2Handler(rr, req)
	if rr.Code != 200 {
		errString := "Save assignment handler HTTP code: %d"
		t.Fatalf(errString, rr.Code)
	}
	if rr.Body.String() != "0" {
		errString := "Response writer contains: %s"
		t.Fatalf(errString, rr.Body.String())
	}
}

// Load using load handler
func loadSatRequest(t *testing.T) Sat {
	// Request using the assignment corresponding to desired submission
	statePath := path.Join("testdata", "sample_assignment.json")
	assignmentBytes, err := ioutil.ReadFile(statePath)
	if err != nil {
		t.Fatal(err)
	}
	buf := bytes.NewBuffer(assignmentBytes)
	req, err := http.NewRequest("POST", "postLoadAssignmentV2", buf)
	if err != nil {
		t.Fatal(err)
	}

	rr := httptest.NewRecorder()
	postLoadAssignmentV2Handler(rr, req)
	if rr.Code != 200 {
		errString := "Load assignment handler HTTP code: %d"
		t.Fatalf(errString, rr.Code)
	}

	loadedSat := Sat{}
	err = json.NewDecoder(rr.Body).Decode(&loadedSat)
	if err != nil {
		t.Fatal(err)
	}
	return loadedSat
}

// Make sure loading then saving doesn't lose info in sat
func checkSatEqual(t *testing.T, sat1 Sat, sat2 Sat) {
	// Some fields can be modified in save/load process
	sat1.Task.Config.SubmitTime = sat2.Task.Config.SubmitTime
	sat1.Session.StartTime = sat2.Session.StartTime
	sat1.Session.SessionId = sat2.Session.SessionId
	if !reflect.DeepEqual(sat1, sat2) {
		t.Fatal("Loaded version is not the same as saved version")
	}
}

// Tests that Sat data can be saved using save handler
// Then loaded again without change
func TestSaveLoadV2(t *testing.T) {
	sampleSat, inputBytes, err := ReadSampleSatData()
	if err != nil {
		t.Fatal(err)
	}
	saveSatRequest(t, inputBytes)
	sat := loadSatRequest(t)
	checkSatEqual(t, sampleSat, sat)
}

// Tests that malformed input (wrong type) will throw errors
func TestSaveMalformedV2(t *testing.T) {
	req, err := http.NewRequest("POST", "postSaveV2",
		bytes.NewBuffer([]byte(`{"task": {"config": {"taskId": 50}}}`)))
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	postSaveV2Handler(rr, req)
	if rr.Body.Len() != 0 {
		errString := "Response should be nil but is: %s"
		t.Fatalf(errString, rr.Body.String())
	}
}

// Tests that demoMode being true will throw errors
func TestSaveDemoV2(t *testing.T) {
	req, err := http.NewRequest("POST", "postSaveV2",
		bytes.NewBuffer([]byte(`{"session": {"demoMode": true}}`)))
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	postSaveV2Handler(rr, req)
	if rr.Body.Len() != 0 {
		errString := "Response should be nil but is: %s"
		t.Fatalf(errString, rr.Body.String())
	}
}
