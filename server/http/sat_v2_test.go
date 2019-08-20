package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"path"
	"testing"
)

// Tests that some basic info can be saved
func TestSavePostV2(t *testing.T) {
	for i := 0; i < 10; i++ {
		taskJson := `{"config": {"assignmentId": "test", "taskSize": %d}}`
		buf := bytes.NewBuffer([]byte(fmt.Sprintf(taskJson, i)))
		req, err := http.NewRequest("POST", "postSave", buf)
		if err != nil {
			t.Fatal(err)
		}
		rr := httptest.NewRecorder()
		postSaveV2Handler(rr, req)
		if rr.Code != 200 {
			errString := "Save assignment handler HTTP code: %d"
			t.Fatal(fmt.Errorf(errString, rr.Code))
		}
		if rr.Body.String() != "0" {
			errString := "Response writer contains: %s"
			t.Fatal(fmt.Errorf(errString, rr.Body.String()))
		}
	}
}

// Tests that malformed input (wrong type) will through errors
func TestSavePostMalformedV2(t *testing.T) {
	req, err := http.NewRequest("POST", "postSave",
		bytes.NewBuffer([]byte(`{"task": {"config": {"taskId": 50}}}`)))
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	postSaveV2Handler(rr, req)
	Info.Println(rr.Body.String())
	if rr.Body.Len() != 0 {
		errString := "Response should be nil but is: %s"
		t.Fatal(fmt.Errorf(errString, rr.Body.String()))
	}
}

// Tests that demoMode being true will throw errors
func TestSavePostDemoV2(t *testing.T) {
	req, err := http.NewRequest("POST", "postSave",
		bytes.NewBuffer([]byte(`{"session": {"demoMode": true}}`)))
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	postSaveV2Handler(rr, req)
	Info.Println(rr.Body.String())
	if rr.Body.Len() != 0 {
		errString := "Response should be nil but is: %s"
		t.Fatal(fmt.Errorf(errString, rr.Body.String()))
	}
}

// Tests that real state, taken from JS console, can be saved
func TestSavePostRealInputV2(t *testing.T) {
	statePath := path.Join("testdata", "sample_state.txt")
	inputBytes, err := ioutil.ReadFile(statePath)
	if err != nil {
		t.Fatal(err)
	}

	req, err := http.NewRequest("POST", "postSave",
		bytes.NewBuffer([]byte(inputBytes)))
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	postSaveV2Handler(rr, req)
	if rr.Code != 200 {
		errString := "Save assignment handler HTTP code: %d"
		t.Fatal(fmt.Errorf(errString, rr.Code))
	}
	if rr.Body.String() != "0" {
		errString := "Response writer contains: %s"
		t.Fatal(fmt.Errorf(errString, rr.Body.String()))
	}
}
