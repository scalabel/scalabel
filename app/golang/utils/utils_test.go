package main

import (
	"encoding/json"
	"path"
	"testing"
)

func TestGetPaths(t *testing.T) {
	a := &Task{}
	str := `{"AssignmentID": "0000", "ProjectName": "testPath",
			"StartTime": 1517447292, "SubmitTime": 1517447292}`
	err := json.Unmarshal([]byte(str), a)

	expectedAssign := path.Join(*data_dir, "Assignments", a.ProjectName,
		a.AssignmentID+".json")
	expectedSubmis := path.Join(*data_dir, "Submissions", a.ProjectName,
		a.AssignmentID, formatTime(a.StartTime)+".json")
	expectedLatest := path.Join(*data_dir, "Submissions", a.ProjectName,
		a.AssignmentID, "latest.json")
	expectedLog := path.Join(*data_dir, "Log", a.ProjectName, a.AssignmentID,
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
