package main

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"path"
	"testing"
)

// Tests that some basic info can be saved
func TestSavePostV2(t *testing.T) {
	for i := 0; i < 10; i += 1 {
		req, err := http.NewRequest("POST", "postSave",
			bytes.NewBuffer([]byte(fmt.Sprintf(`{"config": {"assignmentId": "test", "taskSize": %d}}`, i))))
		if err != nil {
			t.Fatal(err)
		}
		rr := httptest.NewRecorder()
		postSaveV2Handler(rr, req)
		if rr.Code != 200 {
			t.Fatal(errors.New(fmt.Sprintf("Save assignment handler HTTP code: %d", rr.Code)))
		}
		if rr.Body.String() != "0" {
			t.Fatal(errors.New(fmt.Sprintf("Response writer contains: %s", rr.Body.String())))
		}
	}
}

// Tests that malformed input (wrong type) will through errors
func TestSavePostMalformedV2(t *testing.T) {
	req, err := http.NewRequest("POST", "postSave",
		bytes.NewBuffer([]byte(`{"config": {"assignmentId": 50}}`)))
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	postSaveV2Handler(rr, req)
	Info.Println(rr.Body.String())
	if rr.Body.Len() != 0 {
		t.Fatal(errors.New(fmt.Sprintf("Response should be nil but is: %s", rr.Body.String())))
	}
}

// Tests that demoMode being true will throw errrors
func TestSavePostDemoV2(t *testing.T) {
	req, err := http.NewRequest("POST", "postSave",
		bytes.NewBuffer([]byte(`{"config": {"demoMode": true}}`)))
	if err != nil {
		t.Fatal(err)
	}
	rr := httptest.NewRecorder()
	postSaveV2Handler(rr, req)
	Info.Println(rr.Body.String())
	if rr.Body.Len() != 0 {
		t.Fatal(errors.New(fmt.Sprintf("Response should be nil but is: %s", rr.Body.String())))
	}
}

// Tests that real state, taken from JS console, can be saved
func TestSavePostRealInputV2(t *testing.T) {
	inputBytes, err := ioutil.ReadFile(path.Join("testdata", "sample_state.txt"))
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
		t.Fatal(errors.New(fmt.Sprintf("Save assignment handler HTTP code: %d", rr.Code)))
	}
	if rr.Body.String() != "0" {
		t.Fatal(errors.New(fmt.Sprintf("Response writer contains: %s", rr.Body.String())))
	}
}
